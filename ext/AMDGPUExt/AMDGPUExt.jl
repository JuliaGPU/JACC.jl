module AMDGPUExt

using JACC, AMDGPU
using AMDGPU: HIP

const AMDGPUBackend = ROCBackend

include("array.jl")
include("multi.jl")
include("async.jl")
include("experimental/experimental.jl")

JACC.get_backend(::Val{:amdgpu}) = AMDGPUBackend()

default_stream() = AMDGPU.stream()

JACC.default_stream(::AMDGPUBackend) = default_stream()

JACC.create_stream(::AMDGPUBackend) = AMDGPU.HIPStream()

function JACC.synchronize(::AMDGPUBackend; stream = default_stream())
    AMDGPU.synchronize(stream)
end

@inline function _max_shmem_size(kernel)
    kernelShmem = Ref{Int32}()
    HIP.hipFuncGetAttribute(kernelShmem,
        HIP.HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        kernel.fun)
    return kernelShmem[]
end

@inline _kernel_args(args...) = rocconvert.((args))

@inline function _make_kernel(kernel_function, kargs, ::Nothing)
    p_tt = Tuple{Core.Typeof.(kargs)...}
    return AMDGPU.hipfunction(kernel_function, p_tt)
end

@inline function _make_kernel(kernel_function, kargs, kname::AbstractString)
    p_tt = Tuple{Core.Typeof.(kargs)...}
    return AMDGPU.hipfunction(kernel_function, p_tt; name = kname)
end

@inline function _kernel_maxshmem(kernel_function, kargs, kname)
    p_kernel = _make_kernel(kernel_function, kargs, kname)
    return (p_kernel, _max_shmem_size(p_kernel))
end

@inline function _kernel_maxthreads(kernel_function, kargs, kname)
    p_kernel = _make_kernel(kernel_function, kargs, kname)
    return (p_kernel, AMDGPU.launch_configuration(p_kernel).groupsize)
end

function JACC.parallel_for(f, ::AMDGPUBackend, N::Integer, x...; name = nothing)
    kargs = _kernel_args(N, f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu, kargs, name)
    config = AMDGPU.launch_configuration(kernel; shmem = shmem_size)
    threads = min(N, config.groupsize)
    blocks = cld(N, threads)
    kernel(kargs...; groupsize = threads, gridsize = blocks, shmem = shmem_size)
    AMDGPU.synchronize()
end

function JACC.parallel_for(
        f, spec::LaunchSpec{AMDGPUBackend}, N::Integer, x...; name = nothing)
    kargs = _kernel_args(N, f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu, kargs, name)
    if spec.shmem_size < 0
        spec.shmem_size = shmem_size
    end
    if spec.threads == 0
        config = AMDGPU.launch_configuration(kernel; shmem = spec.shmem_size)
        spec.threads = min(N, config.groupsize)
    end
    if spec.blocks == 0
        spec.blocks = cld(N, spec.threads)
    end
    kernel(kargs...; groupsize = spec.threads, gridsize = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)
    if spec.sync
        AMDGPU.synchronize(spec.stream)
    end
end

abstract type BlockIndexer2D end

struct BlockIndexerBasic <: BlockIndexer2D end

# COV_EXCL_START
function (blkIter::BlockIndexerBasic)()
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    return (i, j)
end
# COV_EXCL_STOP

struct BlockIndexerSwapped <: BlockIndexer2D end

# COV_EXCL_START
function (blkIter::BlockIndexerSwapped)()
    j = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    i = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    return (i, j)
end
# COV_EXCL_STOP

# Coalescing-aware 2D block shape for a column-major (m rows x n cols) launch.
# Consecutive workitems (workitemIdx().x, the wavefront direction) must walk the
# stride-1 (row) dimension for global-memory access to coalesce. The aspect-ratio
# heuristic gives good shapes for square/tall arrays, but starves x (down to 1
# workitem) for wide arrays — so we floor x at a full warp whenever there are
# enough rows, and otherwise use every available row.
@inline function _block_shape_2d(maxthreads, m, n)
    maxThreadsX = sqrt(maxthreads)
    y_thr = clamp(floor(Int, (n / m) * maxThreadsX), 1, maxthreads)
    x_thr = fld(maxthreads, y_thr)
    if x_thr < 32 && m >= 32
        x_thr = 32
        y_thr = fld(maxthreads, x_thr)
    elseif m < 32 && x_thr < m
        x_thr = m
        y_thr = max(1, fld(maxthreads, x_thr))
    end
    return (x_thr, y_thr)
end

# Generic launcher used for the swapped-axes fallback (uncoalesced, last resort).
function parallel_for(
        indexer::TI, f, (m, N), (M, N), x...; name = nothing) where {TI}
    kargs = _kernel_args(indexer, (M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu_MN, kargs, name)
    config = AMDGPU.launch_configuration(kernel; shmem = shmem_size)
    maxThreadsX = sqrt(config.groupsize)
    y_thr = clamp(floor(Int, (n / m) * maxThreadsX), 1, config.groupsize)
    x_thr = fld(config.groupsize, y_thr)
    threads = (x_thr, y_thr)
    blocks = (cld(m, x_thr), cld(n, y_thr))
    kernel(kargs...; groupsize = threads, gridsize = blocks, shmem = shmem_size)
    AMDGPU.synchronize()
end

function JACC.parallel_for(
        f, ::AMDGPUBackend, (M, N)::NTuple{2, Integer}, x...; name = nothing)
    dev = AMDGPU.device()
    props = AMDGPU.HIP.properties(dev)
    maxBlocks = (x = props.maxGridSize[1], y = props.maxGridSize[2])
    # Prefer the coalesced (basic) mapping and only swap axes when it is
    # *necessary* — i.e. when the basic grid would exceed the y-dimension limit.
    # Swapping moves the large extent onto the x-axis (which has a far larger
    # limit) but sacrifices memory coalescing, so it must be a last resort.
    kargs = _kernel_args(BlockIndexerBasic(), (M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu_MN, kargs, name)
    config = AMDGPU.launch_configuration(kernel; shmem = shmem_size)
    x_thr, y_thr = _block_shape_2d(config.groupsize, M, N)
    if cld(N, y_thr) > maxBlocks.y && maxBlocks.x >= maxBlocks.y
        _parallel_for(
            BlockIndexerSwapped(), f, (N, M), (M, N), x...; name = name)
    else
        blocks = (cld(M, x_thr), cld(N, y_thr))
        kernel(kargs...; groupsize = (x_thr, y_thr), gridsize = blocks,
            shmem = shmem_size)
        AMDGPU.synchronize()
    end
end

function _parallel_for(indexer::TI, f, spec::LaunchSpec{AMDGPUBackend}, (m, n),
        (M, N), x...; name = nothing) where {TI}
    kargs = _kernel_args(indexer, (M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu_MN, kargs, name)

    if spec.shmem_size < 0
        spec.shmem_size = shmem_size
    end

    if spec.threads == 0
        config = AMDGPU.launch_configuration(kernel; shmem = spec.shmem_size)
        maxThreadsX = sqrt(config.groupsize)
        y_thr = clamp(floor(Int, (n / m) * maxThreadsX), 1, config.groupsize)
        x_thr = fld(config.groupsize, y_thr)
        spec.threads = (x_thr, y_thr)
    end

    if spec.blocks == 0
        spec.blocks = (cld(m, spec.threads[1]), cld(n, spec.threads[2]))
    end

    kernel(kargs...; groupsize = spec.threads, gridsize = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)
    if spec.sync
        AMDGPU.synchronize(spec.stream)
    end
end

function JACC.parallel_for(f, spec::LaunchSpec{AMDGPUBackend},
        (M, N)::NTuple{2, Integer}, x...; name = nothing)
    dev = AMDGPU.device()
    props = AMDGPU.HIP.properties(dev)
    maxBlocks = (x = props.maxGridSize[1], y = props.maxGridSize[2])
    # Determine the y-threads the basic launch would use (from the user's block
    # if given, otherwise the coalescing-aware default), then swap axes only if
    # the basic grid would overflow the y-dimension limit.
    if spec.threads == 0
        kargs = _kernel_args(BlockIndexerBasic(), (M, N), f, x...)
        kernel,
        shmem_size = _kernel_maxshmem(_parallel_for_amdgpu_MN, kargs, name)
        if spec.shmem_size < 0
            spec.shmem_size = shmem_size
        end
        config = AMDGPU.launch_configuration(kernel; shmem = spec.shmem_size)
        x_thr, y_thr = _block_shape_2d(config.groupsize, M, N)
        if cld(N, y_thr) > maxBlocks.y && maxBlocks.x >= maxBlocks.y
            _parallel_for(BlockIndexerSwapped(), f, spec, (N, M), (M, N), x...)
        else
            spec.threads = (x_thr, y_thr)
            if spec.blocks == 0
                spec.blocks = (cld(M, x_thr), cld(N, y_thr))
            end
            kernel(kargs...; groupsize = spec.threads, gridsize = spec.blocks,
                shmem = spec.shmem_size, stream = spec.stream)
            if spec.sync
                AMDGPU.synchronize(spec.stream)
            end
        end
    else
        y_thr = spec.threads[2]
        if cld(N, y_thr) > maxBlocks.y && maxBlocks.x >= maxBlocks.y
            _parallel_for(BlockIndexerSwapped(), f, spec,
                (N, M), (M, N), x...; name = name)
        else
            _parallel_for(
                BlockIndexerBasic(), f, spec, (M, N), (M, N), x...; name = name)
        end
    end
end

function JACC.parallel_for(
        f, ::AMDGPUBackend, (L, M, N)::NTuple{3, Integer}, x...; name = nothing)
    kargs = _kernel_args((L, M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu_LMN, kargs, name)
    numThreads = 32
    Lthreads = min(L, numThreads)
    Mthreads = min(M, numThreads)
    Nthreads = 1
    Lblocks = cld(L, Lthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N, Nthreads)
    kernel(kargs...; groupsize = (Lthreads, Mthreads, Nthreads),
        gridsize = (Lblocks, Mblocks, Nblocks), shmem = shmem_size)
    AMDGPU.synchronize()
end

function JACC.parallel_for(f, spec::LaunchSpec{AMDGPUBackend},
        (L, M, N)::NTuple{3, Integer}, x...; name = nothing)
    kargs = _kernel_args((L, M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu_LMN, kargs, name)
    if spec.shmem_size < 0
        spec.shmem_size = shmem_size
    end
    if spec.threads == 0
        numThreads = 32
        Lthreads = min(L, numThreads)
        Mthreads = min(M, numThreads)
        Nthreads = 1
        spec.threads = (Lthreads, Mthreads, Nthreads)
    end
    if spec.blocks == 0
        Lblocks = cld(L, spec.threads[1])
        Mblocks = cld(M, spec.threads[2])
        Nblocks = cld(N, spec.threads[3])
        spec.blocks = (Lblocks, Mblocks, Nblocks)
    end
    kernel(kargs...; groupsize = spec.threads, gridsize = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)
    if spec.sync
        AMDGPU.synchronize(spec.stream)
    end
end

mutable struct AMDGPUReduceWorkspace{T, TP <: JACC.WkProp} <:
               JACC.ReduceWorkspace
    tmp::AMDGPU.ROCArray{T}
    ret::AMDGPU.ROCArray{T}
end

function JACC.reduce_workspace(::AMDGPUBackend, init::T) where {T}
    AMDGPUReduceWorkspace{T, JACC.Managed}(
        AMDGPU.ROCArray{T}(undef, 0), AMDGPU.ROCArray([init]))
end

function JACC.reduce_workspace(::AMDGPUBackend, tmp::AMDGPU.ROCArray{T},
        init::AMDGPU.ROCArray{T}) where {T}
    AMDGPUReduceWorkspace{T, JACC.Unmanaged}(tmp, init)
end

# Ensure the partial-results buffer is sized for this launch. `init` is seeded
# directly inside the kernels (passed as an argument), so no `fill!` is needed
# here — the reduce is a pure sequence of async kernel launches on the stream.
@inline function _init!(
        wk::AMDGPUReduceWorkspace{T, JACC.Managed}, blocks, init) where {T}
    if length(wk.tmp) != prod(blocks)
        wk.tmp = AMDGPU.ROCArray{typeof(init)}(undef, blocks)
    end
    return nothing
end

@inline function _init!(
        wk::AMDGPUReduceWorkspace{T, JACC.Unmanaged}, blocks, init) where {T}
    nothing
end

JACC.get_result(wk::AMDGPUReduceWorkspace) = Base.Array(wk.ret)[]

_make_kname(base::AbstractString, sfx::AbstractString) = base * "__" * sfx
_make_kname(::Nothing, ::AbstractString) = nothing

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{AMDGPUBackend},
        N::Integer, f, x...; name = nothing)
    wk = reducer.workspace
    op = reducer.op
    init = reducer.init

    kargs1 = _kernel_args(N, op, wk.ret, init, f, x...)
    kernel1,
    threads1 = _kernel_maxthreads(_parallel_reduce_amdgpu, kargs1,
        _make_kname(name, "block_reduce"))

    kargs2 = _kernel_args(1, op, wk.ret, init, wk.ret)
    kernel2,
    threads2 = _kernel_maxthreads(_reduce_kernel_amdgpu, kargs2,
        _make_kname(name, "grid_reduce"))

    threads = min(threads1, threads2, 512)
    blocks = cld(N, threads)
    shmem_size = threads * sizeof(init)

    _init!(wk, blocks, init)

    kargs1 = _kernel_args(N, op, wk.tmp, init, f, x...)
    kernel1(kargs1...; groupsize = threads, gridsize = blocks,
        shmem = shmem_size, stream = reducer.stream)

    kargs2 = _kernel_args(blocks, op, wk.tmp, init, wk.ret)
    kernel2(kargs2...; groupsize = threads, gridsize = 1,
        shmem = shmem_size, stream = reducer.stream)

    if reducer.sync
        AMDGPU.synchronize(reducer.stream)
    end

    return nothing
end

function JACC.parallel_reduce(f, ::AMDGPUBackend, N::Integer, x...; op, init,
        name = nothing)
    ret_inst = AMDGPU.ROCArray{typeof(init)}(undef, 0)

    kargs1 = _kernel_args(N, op, ret_inst, init, f, x...)
    kernel1,
    threads1 = _kernel_maxthreads(_parallel_reduce_amdgpu, kargs1,
        _make_kname(name, "block_reduce"))

    rret = AMDGPU.ROCArray([init])
    kargs2 = _kernel_args(1, op, ret_inst, init, rret)
    kernel2,
    threads2 = _kernel_maxthreads(_reduce_kernel_amdgpu, kargs2,
        _make_kname(name, "grid_reduce"))

    threads = min(threads1, threads2, 512)
    blocks = cld(N, threads)

    shmem_size = threads * sizeof(init)

    # no fill! — `init` is seeded inside the kernels; every block writes its slot
    ret = AMDGPU.ROCArray{typeof(init)}(undef, blocks)
    kargs1 = _kernel_args(N, op, ret, init, f, x...)
    kernel1(
        kargs1...; groupsize = threads, gridsize = blocks, shmem = shmem_size)

    kargs2 = _kernel_args(blocks, op, ret, init, rret)
    kernel2(kargs2...; groupsize = threads, gridsize = 1, shmem = shmem_size)
    AMDGPU.synchronize()

    return Base.Array(rret)[]
end

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{AMDGPUBackend},
        (M, N)::NTuple{2, Integer}, f, x...; name = nothing)
    init = reducer.init
    op = reducer.op
    numThreads = 16
    Mthreads = numThreads
    Nthreads = numThreads
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, threads[1])
    Nblocks = cld(N, threads[2])
    blocks = (Mblocks, Nblocks)
    shmem_size = 16 * 16 * sizeof(init)

    wk = reducer.workspace
    _init!(wk, blocks, init)

    kargs1 = _kernel_args((M, N), op, wk.tmp, init, f, x...)
    kernel1 = _make_kernel(_parallel_reduce_amdgpu_MN, kargs1,
        _make_kname(name, "block_reduce"))
    kernel1(kargs1...; groupsize = threads, gridsize = blocks,
        shmem = shmem_size, stream = reducer.stream)

    kargs2 = _kernel_args(blocks, op, wk.tmp, init, wk.ret)
    kernel2 = _make_kernel(_reduce_kernel_amdgpu_MN, kargs2,
        _make_kname(name, "grid_reduce"))
    kernel2(kargs2...; groupsize = threads, gridsize = (1, 1),
        shmem = shmem_size, stream = reducer.stream)

    if reducer.sync
        AMDGPU.synchronize(reducer.stream)
    end

    return nothing
end

function JACC.parallel_reduce(f, ::AMDGPUBackend, (M, N)::NTuple{2, Integer},
        x...; op, init, name = nothing)
    numThreads = 16
    Mthreads = numThreads
    Nthreads = numThreads
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N, Nthreads)
    blocks = (Mblocks, Nblocks)
    shmem_size = 16 * 16 * sizeof(init)
    # no fill! — `init` is seeded inside the kernels; every block writes its slot
    ret = AMDGPU.ROCArray{typeof(init)}(undef, blocks)
    rret = AMDGPU.ROCArray([init])

    kargs1 = _kernel_args((M, N), op, ret, init, f, x...)
    kernel1 = _make_kernel(_parallel_reduce_amdgpu_MN, kargs1,
        _make_kname(name, "block_reduce"))
    kernel1(kargs1...; groupsize = threads, gridsize = blocks, shmem = shmem_size)

    kargs2 = _kernel_args(blocks, op, ret, init, rret)
    kernel2 = _make_kernel(_reduce_kernel_amdgpu_MN, kargs2,
        _make_kname(name, "grid_reduce"))
    kernel2(kargs2...; groupsize = threads, gridsize = (1, 1), shmem = shmem_size)

    AMDGPU.synchronize()
    return Base.Array(rret)[]
end

@inline function JACC.parallel_reduce(f, ::AMDGPUBackend,
        dims::NTuple{N, Integer}, x...; op, init, kw...) where {N}
    ids = CartesianIndices(dims)
    return JACC.parallel_reduce(JACC.ReduceKernel1DND{typeof(init)}(),
        prod(dims), ids, f, x...; op = op, init = init, kw...)
end

# COV_EXCL_START
@inline function _parallel_for_amdgpu(N, f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    i > N && return nothing
    @inline f(i, x...)
    return nothing
end

@inline function _parallel_for_amdgpu_MN(
        indexer::BlockIndexer2D, (M, N), f, x...)
    i, j = indexer()
    i > M && return nothing
    j > N && return nothing
    @inline f(i, j, x...)
    return nothing
end

@inline function _parallel_for_amdgpu_LMN((L, M, N), f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    k = (workgroupIdx().z - 1) * workgroupDim().z + workitemIdx().z
    i > L && return nothing
    j > M && return nothing
    k > N && return nothing
    @inline f(i, j, k, x...)
    return nothing
end

@inline function _parallel_reduce_amdgpu(N, op, ret, init, f, x...)
    shmem_length = workgroupDim().x
    shared_mem = @ROCDynamicLocalArray(eltype(ret), shmem_length, false)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ti = workitemIdx().x
    @inbounds shared_mem[ti] = init

    if i <= N
        tmp = @inline f(i, x...)
        @inbounds shared_mem[ti] = tmp
    end

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        AMDGPU.sync_workgroup()
        tn = 2^p
        if ti <= tn
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + tn])
        end
    end

    if ti == 1
        @inbounds ret[workgroupIdx().x] = shared_mem[ti]
    end
    return nothing
end

function _reduce_kernel_amdgpu(N, op, red, init, ret)
    shmem_length = workgroupDim().x
    shared_mem = @ROCDynamicLocalArray(eltype(ret), shmem_length, false)
    i = workitemIdx().x
    ii = i
    tmp = init
    for ii in i:shmem_length:N
        tmp = op(tmp, @inbounds red[ii])
    end
    @inbounds shared_mem[i] = tmp

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        AMDGPU.sync_workgroup()
        tn = 2^p
        if i <= tn
            @inbounds shared_mem[i] = op(shared_mem[i], shared_mem[i + tn])
        end
    end

    if i == 1
        @inbounds ret[1] = shared_mem[1]
    end
    return nothing
end

function _parallel_reduce_amdgpu_MN((M, N), op, ret, init, f, x...)
    shared_mem = @ROCDynamicLocalArray(eltype(ret), (16, 16), false)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ti = workitemIdx().x
    tj = workitemIdx().y
    bi = workgroupIdx().x
    bj = workgroupIdx().y

    @inbounds shared_mem[ti, tj] = init

    if (i <= M && j <= N)
        tmp = @inline f(i, j, x...)
        @inbounds shared_mem[ti, tj] = tmp
    end

    for n in (8, 4, 2, 1)
        AMDGPU.sync_workgroup()
        if ti <= n && tj <= n
            @inbounds shared_mem[ti, tj] = op(
                shared_mem[ti, tj], shared_mem[ti + n, tj + n])
            @inbounds shared_mem[ti, tj] = op(
                shared_mem[ti, tj], shared_mem[ti, tj + n])
            @inbounds shared_mem[ti, tj] = op(
                shared_mem[ti, tj], shared_mem[ti + n, tj])
        end
    end

    if (ti == 1 && tj == 1)
        @inbounds ret[bi, bj] = shared_mem[ti, tj]
    end
    return nothing
end

function _reduce_kernel_amdgpu_MN((M, N), op, red, init, ret)
    shared_mem = @ROCDynamicLocalArray(eltype(ret), (16, 16), false)
    i = workitemIdx().x
    j = workitemIdx().y

    tmp = init
    for ci in CartesianIndices((i:16:M, j:16:N))
        tmp = op(tmp, @inbounds red[ci])
    end
    @inbounds shared_mem[i, j] = tmp

    for n in (8, 4, 2, 1)
        AMDGPU.sync_workgroup()
        if i <= n && j <= n
            @inbounds shared_mem[i, j] = op(
                shared_mem[i, j], shared_mem[i + n, j + n])
            @inbounds shared_mem[i, j] = op(
                shared_mem[i, j], shared_mem[i, j + n])
            @inbounds shared_mem[i, j] = op(
                shared_mem[i, j], shared_mem[i + n, j])
        end
    end

    if (i == 1 && j == 1)
        @inbounds ret[1] = shared_mem[i, j]
    end
    return nothing
end

function JACC.shared(::AMDGPUBackend, x::AbstractVector)
    len = length(x)
    shmem = @ROCDynamicLocalArray(eltype(x), len)
    # 1D kernel or 2D kernel at y == 1 (to avoid concurrent writes)
    if workgroupDim().y == 1 || workitemIdx().y == 1
        for i in (workitemIdx().x):(workgroupDim().x):len
            @inbounds shmem[i] = x[i]
        end
    end
    sync_workgroup()
    return shmem
end

function JACC.shared(::AMDGPUBackend, x::AbstractMatrix)
    len = length(x)
    shmem = @ROCDynamicLocalArray(eltype(x), size(x))
    num_threads = workgroupDim().x * workgroupDim().y
    if workgroupDim().y == 1
        # TODO: 1D kernel with 2D array
    else
        if len <= num_threads
            i_local = workitemIdx().x
            j_local = workitemIdx().y
            @inbounds shmem[i_local, j_local] = x[i_local, j_local]
        else
            for i in (workitemIdx().x):(workgroupDim().x):size(x, 1)
                for j in (workitemIdx().y):(workgroupDim().y):size(x, 2)
                    @inbounds shmem[i, j] = x[i, j]
                end
            end
        end
    end
    sync_workgroup()
    return shmem
end

function JACC.shared(::AMDGPUBackend, x::AbstractArray)
    len = length(x)
    shmem = @ROCDynamicLocalArray(eltype(x), size(x))
    num_threads = workgroupDim().x * workgroupDim().y
    if (len <= num_threads)
        if workgroupDim().y == 1
            ind = workitemIdx().x
            @inbounds shmem[ind] = x[ind]
        else
            i_local = workitemIdx().x
            j_local = workitemIdx().y
            ind = (i_local - 1) * workgroupDim().x + j_local
            if ndims(x) == 1
                @inbounds shmem[ind] = x[ind]
            elseif ndims(x) == 2
                @inbounds shmem[ind] = x[i_local, j_local]
            end
        end
    else
        if workgroupDim().y == 1
            ind = workgroupIdx().x
            for i in (workgroupDim().x):(workgroupDim().x):len
                @inbounds shmem[ind] = x[ind]
                ind += workgroupDim().x
            end
        else
            i_local = workgroupIdx().x
            j_local = workgroupIdx().y
            ind = (i_local - 1) * workgroupDim().x + j_local
            if ndims(x) == 1
                for i in num_threads:num_threads:len
                    @inbounds shmem[ind] = x[ind]
                    ind += num_threads
                end
            elseif ndims(x) == 2
                for i in num_threads:num_threads:len
                    @inbounds shmem[ind] = x[i_local, j_local]
                    ind += num_threads
                end
            end
        end
    end
    AMDGPU.sync_workgroup()
    return shmem
end
# COV_EXCL_STOP

JACC.sync_workgroup(::AMDGPUBackend) = AMDGPU.sync_workgroup()

JACC.array_type(::AMDGPUBackend) = AMDGPU.ROCArray

function _compute_array_storage()
    preferences = get(
        JACC.Preferences.Backend._EXT_PREFS[], "amdgpu", Dict{Symbol, Any}())
    value = get(preferences, :storage, "device")
    value isa Union{AbstractString, Symbol} || throw(ArgumentError(
        "Invalid AMDGPU array storage: $(repr(value)); " *
        "expected :device or :host"))
    storage = lowercase(String(value))
    storage in ("device", "host") || throw(ArgumentError(
        "Invalid AMDGPU array storage: $(repr(value)); " *
        "expected :device or :host"))
    return storage
end

# Same live-preference cache as ext/MetalExt/MetalExt.jl `_array_storage`:
# _EXT_PREFS_GENERATION is bumped on every set_backend write, so this is a
# cheap Int comparison on the allocation hot path.
const _ARRAY_STORAGE_CACHE = Ref{Tuple{Int, String}}((-1, ""))

function _array_storage()
    generation = JACC.Preferences.Backend._EXT_PREFS_GENERATION[]
    cached_generation, cached_value = _ARRAY_STORAGE_CACHE[]
    if cached_generation == generation
        return cached_value
    end
    value = _compute_array_storage()
    _ARRAY_STORAGE_CACHE[] = (generation, value)
    return value
end

function JACC._array(::AMDGPUBackend, x::AbstractArray)
    if _array_storage() == "host"
        # hipHostMalloc-backed pinned memory: zero-copy CPU+GPU access on
        # APUs, fast DMA on discrete GPUs.
        return AMDGPU.ROCArray{eltype(x), ndims(x), AMDGPU.Mem.HostBuffer}(x)
    end
    return AMDGPU.ROCArray{eltype(x), ndims(x), AMDGPU.Mem.HIPBuffer}(x)
end

JACC._array(::AMDGPUBackend, x::AMDGPU.ROCArray) = x
JACC.array(backend::AMDGPUBackend, x::AbstractArray) = JACC._array(backend, x)
JACC.array(::AMDGPUBackend, x::AMDGPU.ROCArray) = x

end # module AMDGPUExt
