module CUDAExt

using JACC, CUDA

# overloaded array functions
include("array.jl")
include("multi.jl")
include("async.jl")
include("experimental/experimental.jl")

JACC.get_backend(::Val{:cuda}) = CUDABackend()

default_stream() = CUDA.stream()

JACC.default_stream(::CUDABackend) = default_stream()

JACC.create_stream(::CUDABackend) = CUDA.CuStream()

function JACC.synchronize(::CUDABackend; stream = default_stream())
    CUDA.synchronize(stream)
end

@inline function _max_shmem_size(kernel)
    maxShmem = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    shmemUsedByKernel = CUDA.memory(kernel).shared
    return maxShmem - shmemUsedByKernel
end

@inline _kernel_args(args...) = cudaconvert.((args))

@inline function _make_kernel(kernel_function, kargs)
    p_tt = Tuple{Core.Typeof.(kargs)...}
    return cufunction(kernel_function, p_tt)
end

@inline function _kernel_maxshmem(kernel_function, kargs)
    p_kernel = _make_kernel(kernel_function, kargs)
    return (p_kernel, _max_shmem_size(p_kernel))
end

@inline function _kernel_maxthreads(kernel_function, kargs)
    p_kernel = _make_kernel(kernel_function, kargs)
    return (p_kernel, CUDA.launch_configuration(p_kernel.fun).threads)
end

function JACC.parallel_for(f, ::CUDABackend, N::Integer, x...)
    kargs = _kernel_args(N, f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_cuda, kargs)
    config = CUDA.launch_configuration(kernel.fun; shmem = shmem_size)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    CUDA.@sync kernel(
        kargs...; threads = threads, blocks = blocks, shmem = shmem_size)
end

function JACC.parallel_for(f, spec::LaunchSpec{CUDABackend}, N::Integer, x...)
    kargs = _kernel_args(N, f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_cuda, kargs)
    if spec.shmem_size < 0
        spec.shmem_size = shmem_size
    end
    if spec.threads == 0
        config = CUDA.launch_configuration(kernel.fun; shmem = spec.shmem_size)
        spec.threads = min(N, config.threads)
    end
    if spec.blocks == 0
        spec.blocks = cld(N, spec.threads)
    end
    kernel(kargs...; threads = spec.threads, blocks = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)
    if spec.sync
        CUDA.synchronize(spec.stream)
    end
end

abstract type BlockIndexer2D end

struct BlockIndexerBasic <: BlockIndexer2D end

# COV_EXCL_START
function (blkIter::BlockIndexerBasic)()
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    return (i, j)
end
# COV_EXCL_STOP

struct BlockIndexerSwapped <: BlockIndexer2D end

# COV_EXCL_START
function (blkIter::BlockIndexerSwapped)()
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    return (i, j)
end
# COV_EXCL_STOP

function _parallel_for(indexer::TI, f, (m, n), (M, N), x...) where {TI}
    kargs = _kernel_args(indexer, (M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_cuda_MN, kargs)
    config = CUDA.launch_configuration(kernel.fun; shmem = shmem_size)
    maxThreadsX = sqrt(config.threads)
    y_thr = floor(Int, (n / m) * maxThreadsX)
    x_thr = fld(config.threads, y_thr)
    threads = (x_thr, y_thr)
    blocks = (cld(m, x_thr), cld(n, y_thr))
    kernel(kargs...; threads = threads, blocks = blocks, shmem = shmem_size)
    CUDA.synchronize()
end

function JACC.parallel_for(f, ::CUDABackend, (M, N)::NTuple{2, Integer}, x...)
    dev = CUDA.device()
    maxBlocks = (
        x = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
        y = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    )
    if M < N && maxBlocks.x > maxBlocks.y
        _parallel_for(BlockIndexerSwapped(), f, (N, M), (M, N), x...)
    else
        _parallel_for(BlockIndexerBasic(), f, (M, N), (M, N), x...)
    end
end

function _parallel_for(indexer::TI, f, spec::LaunchSpec{CUDABackend}, (m, n),
        (M, N), x...) where {TI}
    kargs = _kernel_args(indexer, (M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_cuda_MN, kargs)

    if spec.shmem_size < 0
        spec.shmem_size = shmem_size
    end

    if spec.threads == 0
        config = CUDA.launch_configuration(kernel.fun; shmem = spec.shmem_size)
        maxThreadsX = sqrt(config.threads)
        y_thr = floor(Int, (n / m) * maxThreadsX)
        x_thr = fld(config.threads, y_thr)
        spec.threads = (x_thr, y_thr)
    end

    if spec.blocks == 0
        spec.blocks = (cld(m, spec.threads[1]), cld(n, spec.threads[2]))
    end

    kernel(kargs...; threads = spec.threads, blocks = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)
    if spec.sync
        CUDA.synchronize(spec.stream)
    end
end

function JACC.parallel_for(
        f, spec::LaunchSpec{CUDABackend}, (M, N)::NTuple{2, Integer}, x...)
    dev = CUDA.device()
    maxBlocks = (
        x = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
        y = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    )
    if M < N && maxBlocks.x > maxBlocks.y
        _parallel_for(BlockIndexerSwapped(), f, spec, (N, M), (M, N), x...)
    else
        _parallel_for(BlockIndexerBasic(), f, spec, (M, N), (M, N), x...)
    end
end

function JACC.parallel_for(f, ::CUDABackend, (L, M, N)::NTuple{3, Integer}, x...)
    kargs = _kernel_args((L, M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_cuda_LMN, kargs)
    numThreads = 32
    Lthreads = min(L, numThreads)
    Mthreads = min(M, numThreads)
    Nthreads = 1
    Lblocks = cld(L, Lthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N, Nthreads)
    kernel(kargs...; threads = (Lthreads, Mthreads, Nthreads),
        blocks = (Lblocks, Mblocks, Nblocks), shmem = shmem_size)
    CUDA.synchronize()
end

function JACC.parallel_for(
        f, spec::LaunchSpec{CUDABackend}, (L, M, N)::NTuple{3, Integer}, x...)
    kargs = _kernel_args((L, M, N), f, x...)
    kernel, shmem_size = _kernel_maxshmem(_parallel_for_cuda_LMN, kargs)
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
    kernel(kargs...; threads = spec.threads, blocks = spec.blocks,
        shmem = spec.shmem_size)
    if spec.sync
        CUDA.synchronize(spec.stream)
    end
end

mutable struct CUDAReduceWorkspace{T, TP <: JACC.WkProp} <: JACC.ReduceWorkspace
    tmp::CUDA.CuArray{T}
    ret::CUDA.CuArray{T}
end

function JACC.reduce_workspace(::CUDABackend, init::T) where {T}
    CUDAReduceWorkspace{T, JACC.Managed}(
        CUDA.CuArray{T}(undef, 0), CUDA.CuArray([init]))
end

function JACC.reduce_workspace(::CUDABackend, tmp::CUDA.CuArray{T},
        init::CUDA.CuArray{T}) where {T}
    CUDAReduceWorkspace{T, JACC.Unmanaged}(tmp, init)
end

@inline function _init!(
        wk::CUDAReduceWorkspace{T, JACC.Managed}, blocks, init) where {T}
    if length(wk.tmp) != prod(blocks)
        wk.tmp = CUDA.CuArray{typeof(init)}(undef, blocks)
    end
    fill!(wk.tmp, init)
    fill!(wk.ret, init)
    return nothing
end

@inline function _init!(
        wk::CUDAReduceWorkspace{T, JACC.Unmanaged}, blocks, init) where {T}
    nothing
end

JACC.get_result(wk::CUDAReduceWorkspace) = Base.Array(wk.ret)[]

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{CUDABackend},
        N::Integer, f, x...)
    wk = reducer.workspace
    op = reducer.op
    init = reducer.init

    kargs_1 = _kernel_args(N, op, wk.ret, f, x...)
    kernel_1, maxThreads_1 = _kernel_maxthreads(_parallel_reduce_cuda, kargs_1)

    kargs_2 = _kernel_args(1, op, wk.ret, wk.ret)
    kernel_2, maxThreads_2 = _kernel_maxthreads(_reduce_kernel_cuda, kargs_2)

    threads = min(maxThreads_1, maxThreads_2, 512)
    blocks = cld(N, threads)
    shmem_size = threads * sizeof(init)

    _init!(wk, blocks, init)

    kargs = _kernel_args(N, op, wk.tmp, f, x...)
    kernel_1(kargs...; threads = threads, blocks = blocks,
        shmem = shmem_size, stream = reducer.stream)

    kargs = _kernel_args(blocks, op, wk.tmp, wk.ret)
    kernel_2(kargs...; threads = threads, blocks = 1,
        shmem = shmem_size, stream = reducer.stream)

    if reducer.sync
        CUDA.synchronize(reducer.stream)
    end

    return nothing
end

function JACC.parallel_reduce(f, ::CUDABackend, N::Integer, x...; op, init)
    ret_inst = CUDA.CuArray{typeof(init)}(undef, 0)

    kargs_1 = _kernel_args(N, op, ret_inst, f, x...)
    kernel_1, maxThreads_1 = _kernel_maxthreads(_parallel_reduce_cuda, kargs_1)

    rret = CUDA.CuArray([init])
    kargs_2 = _kernel_args(1, op, ret_inst, rret)
    kernel_2, maxThreads_2 = _kernel_maxthreads(_reduce_kernel_cuda, kargs_2)

    threads = min(maxThreads_1, maxThreads_2, 512)
    blocks = cld(N, threads)

    shmem_size = threads * sizeof(init)

    ret = fill!(CUDA.CuArray{typeof(init)}(undef, blocks), init)
    kargs = _kernel_args(N, op, ret, f, x...)
    kernel_1(kargs...; threads = threads, blocks = blocks, shmem = shmem_size)

    kargs = _kernel_args(blocks, op, ret, rret)
    kernel_2(kargs...; threads = threads, blocks = 1, shmem = shmem_size)

    CUDA.synchronize()

    return Base.Array(rret)[]
end

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{CUDABackend},
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
    kernel1 = _make_kernel(_parallel_reduce_cuda_MN, kargs1)
    kernel1(kargs1...; threads = threads, blocks = blocks, shmem = shmem_size,
        stream = reducer.stream)

    kargs2 = _kernel_args(blocks, op, wk.tmp, wk.ret)
    kernel2 = _make_kernel(_reduce_kernel_cuda_MN, kargs2)
    kernel2(kargs2...; threads = threads, blocks = (1, 1), shmem = shmem_size,
        stream = reducer.stream)

    if reducer.sync
        CUDA.synchronize(reducer.stream)
    end

    return nothing
end

function JACC.parallel_reduce(
        f, ::CUDABackend, (M, N)::NTuple{2, Integer}, x...; op, init)
    numThreads = 16
    Mthreads = numThreads
    Nthreads = numThreads
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N, Nthreads)
    blocks = (Mblocks, Nblocks)
    shmem_size = 16 * 16 * sizeof(init)
    ret = fill!(CUDA.CuArray{typeof(init)}(undef, (Mblocks, Nblocks)), init)
    rret = CUDA.CuArray([init])

    kargs1 = _kernel_args((M, N), op, ret, f, x...)
    kernel1 = _make_kernel(_parallel_reduce_cuda_MN, kargs1)
    kernel1(kargs1...; threads = threads, blocks = blocks, shmem = shmem_size)

    kargs2 = _kernel_args((Mblocks, Nblocks), op, ret, rret)
    kernel2 = _make_kernel(_reduce_kernel_cuda_MN, kargs2)
    kernel2(kargs2...; threads = threads, blocks = (1, 1), shmem = shmem_size)

    CUDA.synchronize()
    return Base.Array(rret)[]
end

@inline function JACC.parallel_reduce(f, ::CUDABackend,
        dims::NTuple{N, Integer}, x...; op, init) where {N}
    ids = CartesianIndices(dims)
    return JACC.parallel_reduce(JACC.ReduceKernel1DND{typeof(init)}(),
        prod(dims), ids, f, x...; op = op, init = init)
end

# COV_EXCL_START
@inline function _parallel_for_cuda(N, f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    @inline f(i, x...)
    return nothing
end

@inline function _parallel_for_cuda_MN(indexer::BlockIndexer2D, (M, N), f, x...)
    i, j = indexer()
    i > M && return nothing
    j > N && return nothing
    @inline f(i, j, x...)
    return nothing
end

@inline function _parallel_for_cuda_LMN((L, M, N), f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    i > L && return nothing
    j > M && return nothing
    k > N && return nothing
    @inline f(i, j, k, x...)
    return nothing
end

function _parallel_reduce_cuda(N, op, ret, f, x...)
    shmem_length = blockDim().x
    shared_mem = CuDynamicSharedArray(eltype(ret), shmem_length)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x
    @inbounds shared_mem[ti] = ret[blockIdx().x]

    if i <= N
        tmp = @inline f(i, x...)
        @inbounds shared_mem[ti] = tmp
    end

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        sync_threads()
        tn = 2^p
        if (ti <= tn)
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + tn])
        end
    end

    if ti == 1
        @inbounds ret[blockIdx().x] = shared_mem[ti]
    end
    return nothing
end

function _reduce_kernel_cuda(N, op, red, ret)
    shmem_length = blockDim().x
    shared_mem = CuDynamicSharedArray(eltype(ret), shmem_length)
    i = threadIdx().x
    ii = i
    @inbounds tmp = ret[1]
    for ii in i:shmem_length:N
        tmp = op(tmp, @inbounds red[ii])
    end
    @inbounds shared_mem[i] = tmp

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        sync_threads()
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

function _parallel_reduce_cuda_MN((M, N), op, ret, f, x...)
    shared_mem = CuDynamicSharedArray(eltype(ret), (16, 16))
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ti = threadIdx().x
    tj = threadIdx().y
    bi = blockIdx().x
    bj = blockIdx().y

    @inbounds shared_mem[ti, tj] = ret[bi, bj]

    if (i <= M && j <= N)
        tmp = @inline f(i, j, x...)
        @inbounds shared_mem[ti, tj] = tmp
    end

    for n in (8, 4, 2, 1)
        sync_threads()
        if (ti <= n && tj <= n)
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

function _reduce_kernel_cuda_MN((M, N), op, red, ret)
    shared_mem = CuDynamicSharedArray(eltype(ret), (16, 16))
    i = threadIdx().x
    j = threadIdx().y

    @inbounds tmp = ret[1]
    for ci in CartesianIndices((i:16:M, j:16:N))
        tmp = op(tmp, @inbounds red[ci])
    end
    @inbounds shared_mem[i, j] = tmp

    for n in (8, 4, 2, 1)
        sync_threads()
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

function JACC.shared(::CUDABackend, x::AbstractVector)
    len = length(x)
    shmem = CuDynamicSharedArray(eltype(x), len)
    # 1D kernel or 2D kernel at y == 1 (to avoid concurrent writes)
    if blockDim().y == 1 || threadIdx().y == 1
        for i in (threadIdx().x):(blockDim().x):len
            @inbounds shmem[i] = x[i]
        end
    end
    sync_threads()
    return shmem
end

function JACC.shared(::CUDABackend, x::AbstractMatrix)
    len = length(x)
    shmem = CuDynamicSharedArray(eltype(x), size(x))
    num_threads = blockDim().x * blockDim().y
    if blockDim().y == 1
        # TODO: 1D kernel with 2D array
    else
        if len <= num_threads
            i_local = threadIdx().x
            j_local = threadIdx().y
            @inbounds shmem[i_local, j_local] = x[i_local, j_local]
        else
            for i in (threadIdx().x):(blockDim().x):size(x, 1)
                for j in (threadIdx().y):(blockDim().y):size(x, 2)
                    @inbounds shmem[i, j] = x[i, j]
                end
            end
        end
    end
    sync_threads()
    return shmem
end
# COV_EXCL_STOP

JACC.sync_workgroup(::CUDABackend) = CUDA.sync_threads()

JACC.array_type(::CUDABackend) = CUDA.CuArray

JACC.array(::CUDABackend, x::AbstractArray) = CUDA.CuArray(x)

end # module CUDAExt
