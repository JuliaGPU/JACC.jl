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

@inline function _make_kernel(kernel_function, kargs)
    p_tt = Tuple{Core.Typeof.(kargs)...}
    p_kernel = AMDGPU.hipfunction(kernel_function, p_tt)
end

@inline function _kernel_maxshmem(kernel_function, kargs)
    p_kernel = _make_kernel(kernel_function, kargs)
    return (p_kernel, _max_shmem_size(p_kernel))
end

@inline function _kernel_maxthreads(kernel_function, kargs)
    p_kernel = _make_kernel(kernel_function, kargs)
    return (p_kernel, AMDGPU.launch_configuration(p_kernel).groupsize)
end

function JACC.parallel_for(f, ::AMDGPUBackend, N::Integer, x...)
    kargs = _kernel_args(N, f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu, kargs)
    config = AMDGPU.launch_configuration(kernel, shmem = shmem_size)
    threads = min(N, config.groupsize)
    blocks = cld(N, threads)
    kernel(kargs...; groupsize = threads, gridsize = blocks, shmem = shmem_size)
    AMDGPU.synchronize()
end

function JACC.parallel_for(
        f, spec::LaunchSpec{AMDGPUBackend}, N::Integer, x...)
    kargs = _kernel_args(N, f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu, kargs)
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

function _parallel_for(indexer::TI, f, (m, n), (M, N), x...) where {TI}
    kargs = _kernel_args(indexer, (M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu_MN, kargs)
    config = AMDGPU.launch_configuration(kernel; shmem = shmem_size)
    maxThreadsX = sqrt(config.groupsize)
    y_thr = floor(Int, (n / m) * maxThreadsX)
    x_thr = fld(config.groupsize, y_thr)
    threads = (x_thr, y_thr)
    blocks = (cld(m, x_thr), cld(n, y_thr))
    kernel(kargs...; groupsize = threads, gridsize = blocks, shmem = shmem_size)
    AMDGPU.synchronize()
end

function JACC.parallel_for(
        f, ::AMDGPUBackend, (M, N)::NTuple{2, Integer}, x...)
    dev = AMDGPU.device()
    props = AMDGPU.HIP.properties(dev)
    maxBlocks = (x = props.maxGridSize[1], y = props.maxGridSize[2])
    if M < N && maxBlocks.x > maxBlocks.y
        _parallel_for(BlockIndexerSwapped(), f, (N, M), (M, N), x...)
    else
        _parallel_for(BlockIndexerBasic(), f, (M, N), (M, N), x...)
    end
end

function _parallel_for(indexer::TI, f, spec::LaunchSpec{AMDGPUBackend}, (m, n),
        (M, N), x...) where {TI}
    kargs = _kernel_args(indexer, (M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu_MN, kargs)

    if spec.shmem_size < 0
        spec.shmem_size = shmem_size
    end

    if spec.threads == 0
        config = AMDGPU.launch_configuration(kernel; shmem = spec.shmem_size)
        maxThreadsX = sqrt(config.groupsize)
        y_thr = floor(Int, (n / m) * maxThreadsX)
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

function JACC.parallel_for(
        f, spec::LaunchSpec{AMDGPUBackend}, (M, N)::NTuple{2, Integer}, x...)
    dev = AMDGPU.device()
    props = AMDGPU.HIP.properties(dev)
    maxBlocks = (x = props.maxGridSize[1], y = props.maxGridSize[2])
    if M < N && maxBlocks.x > maxBlocks.y
        _parallel_for(BlockIndexerSwapped(), f, spec, (N, M), (M, N), x...)
    else
        _parallel_for(BlockIndexerBasic(), f, spec, (M, N), (M, N), x...)
    end
end

function JACC.parallel_for(
        f, ::AMDGPUBackend, (L, M, N)::NTuple{3, Integer}, x...)
    kargs = _kernel_args((L, M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu_LMN, kargs)
    numThreads = 32
    Lthreads = min(L, numThreads)
    Mthreads = min(M, numThreads)
    Nthreads = 1
    Lblocks = cld(L, Lthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N, Nthreads)
    kernel(kargs...; groupsize = (Lthreads, Mthreads, Nthreads),
        gridsize = (Lblocks, Mblocks, Nblocks), shmem=shmem_size)
    AMDGPU.synchronize()
end

function JACC.parallel_for(f, spec::LaunchSpec{AMDGPUBackend},
        (L, M, N)::NTuple{3, Integer}, x...)
    kargs = _kernel_args((L, M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_amdgpu_LMN, kargs)
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
        shmem=spec.shmem_size, stream=spec.stream)
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

@inline function _init!(
        wk::AMDGPUReduceWorkspace{T, JACC.Managed}, blocks, init) where {T}
    if length(wk.tmp) != prod(blocks)
        wk.tmp = AMDGPU.ROCArray{typeof(init)}(undef, blocks)
    end
    fill!(wk.tmp, init)
    fill!(wk.ret, init)
    return nothing
end

@inline function _init!(
        wk::AMDGPUReduceWorkspace{T, JACC.Unmanaged}, blocks, init) where {T}
    nothing
end

JACC.get_result(wk::AMDGPUReduceWorkspace) = Base.Array(wk.ret)[]

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{AMDGPUBackend},
        N::Integer, f, x...)
    wk = reducer.workspace
    op = reducer.op
    init = reducer.init

    kargs1 = _kernel_args(N, op, wk.ret, f, x...)
    kernel1, threads1 = _kernel_maxthreads(_parallel_reduce_amdgpu, kargs1)

    kargs2 = _kernel_args(1, op, wk.ret, wk.ret)
    kernel2, threads2 = _kernel_maxthreads(_reduce_kernel_amdgpu, kargs2)

    threads = min(threads1, threads2, 512)
    blocks = cld(N, threads)
    shmem_size = threads * sizeof(init)

    _init!(wk, blocks, init)

    kargs1 = _kernel_args(N, op, wk.tmp, f, x...)
    kernel1(kargs1...; groupsize = threads, gridsize = blocks,
        shmem = shmem_size, stream = reducer.stream)

    kargs2 = _kernel_args(blocks, op, wk.tmp, wk.ret)
    kernel2(kargs2...; groupsize = threads, gridsize = 1,
        shmem = shmem_size, stream = reducer.stream)

    if reducer.sync
        AMDGPU.synchronize(reducer.stream)
    end

    return nothing
end

function JACC.parallel_reduce(f, ::AMDGPUBackend, N::Integer, x...; op, init)
    ret_inst = AMDGPU.ROCArray{typeof(init)}(undef, 0)

    kargs1 = _kernel_args(N, op, ret_inst, f, x...)
    kernel1, threads1 = _kernel_maxthreads(_parallel_reduce_amdgpu, kargs1)

    rret = AMDGPU.ROCArray([init])
    kargs2 = _kernel_args(1, op, ret_inst, rret)
    kernel2, threads2 = _kernel_maxthreads(_reduce_kernel_amdgpu, kargs2)

    threads = min(threads1, threads2, 512)
    blocks = cld(N, threads)

    shmem_size = threads * sizeof(init)

    ret = fill!(AMDGPU.ROCArray{typeof(init)}(undef, blocks), init)
    kargs1 = _kernel_args(N, op, ret, f, x...)
    kernel1(
        kargs1...; groupsize = threads, gridsize = blocks, shmem = shmem_size)

    kargs2 = _kernel_args(blocks, op, ret, rret)
    kernel2(kargs2...; groupsize = threads, gridsize = 1, shmem = shmem_size)
    AMDGPU.synchronize()

    return Base.Array(rret)[]
end

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{AMDGPUBackend},
        (M, N)::NTuple{2, Integer}, f, x...)
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

    kargs1 = _kernel_args((M, N), op, wk.tmp, f, x...)
    kernel1 = _make_kernel(_parallel_reduce_amdgpu_MN, kargs1)
    kernel1(kargs1...; groupsize = threads, gridsize = blocks,
        shmem = shmem_size, stream = reducer.stream)

    kargs2 = _kernel_args(blocks, op, wk.tmp, wk.ret)
    kernel2 = _make_kernel(_reduce_kernel_amdgpu_MN, kargs2)
    kernel2(kargs2...; groupsize = threads, gridsize = (1, 1),
        shmem = shmem_size, stream = reducer.stream)

    if reducer.sync
        AMDGPU.synchronize(reducer.stream)
    end

    return nothing
end

function JACC.parallel_reduce(f, ::AMDGPUBackend, (M, N)::NTuple{2, Integer},
        x...; op, init)
    numThreads = 16
    Mthreads = numThreads
    Nthreads = numThreads
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N, Nthreads)
    blocks = (Mblocks, Nblocks)
    shmem_size = 16 * 16 * sizeof(init)
    ret = fill!(AMDGPU.ROCArray{typeof(init)}(undef, blocks), init)
    rret = AMDGPU.ROCArray([init])

    kargs1 = _kernel_args((M, N), op, ret, f, x...)
    kernel1 = _make_kernel(_parallel_reduce_amdgpu_MN, kargs1)
    kernel1(kargs1...; groupsize = threads, gridsize = blocks, shmem = shmem_size)

    kargs2 = _kernel_args(blocks, op, ret, rret)
    kernel2 = _make_kernel(_reduce_kernel_amdgpu_MN, kargs2)
    kernel2(kargs2...; groupsize = threads, gridsize = (1, 1), shmem = shmem_size)

    AMDGPU.synchronize()
    return Base.Array(rret)[]
end

@inline function JACC.parallel_reduce(f, ::AMDGPUBackend,
        dims::NTuple{N, Integer}, x...; op, init) where {N}
    ids = CartesianIndices(dims)
    return JACC.parallel_reduce(JACC.ReduceKernel1DND{typeof(init)}(),
        prod(dims), ids, f, x...; op = op, init = init)
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

@inline function _parallel_reduce_amdgpu(N, op, ret, f, x...)
    shmem_length = workgroupDim().x
    shared_mem = @ROCDynamicLocalArray(eltype(ret), shmem_length, false)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ti = workitemIdx().x
    @inbounds shared_mem[ti] = ret[workgroupIdx().x]

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

function _reduce_kernel_amdgpu(N, op, red, ret)
    shmem_length = workgroupDim().x
    shared_mem = @ROCDynamicLocalArray(eltype(ret), shmem_length, false)
    i = workitemIdx().x
    ii = i
    @inbounds tmp = ret[1]
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

function _parallel_reduce_amdgpu_MN((M, N), op, ret, f, x...)
    shared_mem = @ROCDynamicLocalArray(eltype(ret), (16, 16), false)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ti = workitemIdx().x
    tj = workitemIdx().y
    bi = workgroupIdx().x
    bj = workgroupIdx().y

    @inbounds shared_mem[ti, tj] = ret[bi, bj]

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

function _reduce_kernel_amdgpu_MN((M, N), op, red, ret)
    shared_mem = @ROCDynamicLocalArray(eltype(ret), (16, 16), false)
    i = workitemIdx().x
    j = workitemIdx().y

    @inbounds tmp = ret[1]
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

JACC.array(::AMDGPUBackend, x::AbstractArray) = AMDGPU.ROCArray(x)

end # module AMDGPUExt
