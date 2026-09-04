module oneAPIExt

using JACC, oneAPI, oneAPI.oneL0

# overloaded array functions
include("array.jl")
include("async.jl")
include("experimental/experimental.jl")

JACC.get_backend(::Val{:oneapi}) = oneAPIBackend()

default_stream() = oneAPI.global_queue(oneAPI.context(), oneAPI.device())

JACC.default_stream(::oneAPIBackend) = default_stream()

function JACC.create_stream(::oneAPIBackend)
    oneAPI.oneL0.ZeCommandQueue(oneAPI.context(), oneAPI.device())
end

function JACC.synchronize(::oneAPIBackend; stream = default_stream())
    oneAPI.synchronize(stream)
end

@inline kernel_args(args...) = kernel_convert.((args))

@inline function _make_kernel(::Nothing, kernel_function, args...)
    return @oneapi launch=false kernel_function(args...)
end

@inline function _make_kernel(kname::AbstractString, kernel_function, args...)
    if isdigit(first(kname))
        kname = "k_" * kname
    end
    return @oneapi name=kname launch=false kernel_function(args...)
end

function JACC.parallel_for(f, ::oneAPIBackend, N::Integer, x...; name = nothing)
    kernel = _make_kernel(name, _parallel_for_oneapi, N, f, x...)
    config_items = div(oneAPI.launch_configuration(kernel), 2)
    items = min(N, config_items, 256)
    groups = cld(N, items)
    kernel(N, f, x...; items = items, groups = groups)
    oneAPI.synchronize()
end

function JACC.parallel_for(
        f, spec::LaunchSpec{oneAPIBackend}, N::Integer, x...; name = nothing)
    kernel = _make_kernel(name, _parallel_for_oneapi, N, f, x...)
    if spec.threads == 0
        maxItems = oneAPI.launch_configuration(kernel)
        spec.threads = min(N, maxItems)
    end
    if spec.blocks == 0
        spec.blocks = cld(N, spec.threads)
    end
    kernel(N, f, x...; items = spec.threads,
        groups = spec.blocks, queue = spec.stream)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

abstract type BlockIndexer2D end

struct BlockIndexerBasic <: BlockIndexer2D end

function (blkIter::BlockIndexerBasic)()
    i = get_global_id(1)
    j = get_global_id(2)
    return (i, j)
end

struct BlockIndexerSwapped <: BlockIndexer2D end

function (blkIter::BlockIndexerSwapped)()
    j = get_global_id(1)
    i = get_global_id(2)
    return (i, j)
end

function _parallel_for(
        indexer::TI, f, (m, n), (M, N), x...; name = nothing) where {TI}
    kernel = _make_kernel(
        name, _parallel_for_oneapi_MN, indexer, (M, N), f, x...)
    maxThreads = div(oneAPI.launch_configuration(kernel), 2)
    maxThreadsX = sqrt(maxThreads)
    y_thr = clamp(floor(Int, (n / m) * maxThreadsX), 1, maxThreads)
    x_thr = fld(maxThreads, y_thr)
    items = (x_thr, y_thr)
    groups = (cld(m, items[1]), cld(n, items[2]))

    kernel(indexer, (M, N), f, x...; items = items, groups = groups)
    oneAPI.synchronize()
end

function JACC.parallel_for(
        f, ::oneAPIBackend, (M, N)::NTuple{2, Integer}, x...; kw...)
    dev = oneAPI.device()
    props = oneAPI.compute_properties(dev)
    maxBlocks = (x = props.maxGroupCountX, y = props.maxGroupCountY)
    if M < N && maxBlocks.x >= maxBlocks.y
        _parallel_for(BlockIndexerSwapped(), f, (N, M), (M, N), x...; kw...)
    else
        _parallel_for(BlockIndexerBasic(), f, (M, N), (M, N), x...; kw...)
    end
end

function _parallel_for(indexer::TI, f, spec::LaunchSpec{oneAPIBackend}, (m, n),
        (M, N), x...; name = nothing) where {TI}
    kernel = _make_kernel(
        name, _parallel_for_oneapi_MN, indexer, (M, N), f, x...)

    if spec.threads == 0
        maxThreads = oneAPI.launch_configuration(kernel)
        maxThreadsX = sqrt(maxThreads)
        y_thr = clamp(floor(Int, (n / m) * maxThreadsX), 1, maxThreads)
        x_thr = fld(maxThreads, y_thr)
        spec.threads = (x_thr, y_thr)
    end

    if spec.blocks == 0
        spec.blocks = (cld(m, spec.threads[1]), cld(n, spec.threads[2]))
    end
    kernel(
        indexer, (M, N), f, x...; items = spec.threads, groups = spec.blocks,
        queue = spec.stream)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

function JACC.parallel_for(f, spec::LaunchSpec{oneAPIBackend},
        (M, N)::NTuple{2, Integer}, x...; kw...)
    dev = oneAPI.device()
    props = oneAPI.compute_properties(dev)
    maxBlocks = (x = props.maxGroupCountX, y = props.maxGroupCountY)
    if M < N && maxBlocks.x >= maxBlocks.y
        _parallel_for(
            BlockIndexerSwapped(), f, spec, (N, M), (M, N), x...; kw...)
    else
        _parallel_for(BlockIndexerBasic(), f, spec, (M, N), (M, N), x...; kw...)
    end
end

function JACC.parallel_for(
        f, ::oneAPIBackend, (L, M, N)::NTuple{3, Integer}, x...; name = nothing)
    kernel = _make_kernel(name, _parallel_for_oneapi_LMN, (L, M, N), f, x...)
    maxItems = 8
    Litems = min(L, maxItems)
    Mitems = min(M, maxItems)
    Nitems = 1
    items = (Litems, Mitems, Nitems)
    Lgroups = cld(L, Litems)
    Mgroups = cld(M, Mitems)
    Ngroups = cld(N, Nitems)
    groups = (Lgroups, Mgroups, Ngroups)
    kernel((L, M, N), f, x...; items = items, groups = groups)
    oneAPI.synchronize();
end

function JACC.parallel_for(f, spec::LaunchSpec{oneAPIBackend},
        (L, M, N)::NTuple{3, Integer}, x...; name = nothing)
    kernel = _make_kernel(name, _parallel_for_oneapi_LMN, (L, M, N), f, x...)
    if spec.threads == 0
        maxItems = 8
        Litems = min(L, maxItems)
        Mitems = min(M, maxItems)
        Nitems = 1
        spec.threads = (Litems, Mitems, Nitems)
    end
    if spec.blocks == 0
        Lgroups = cld(L, spec.threads[1])
        Mgroups = cld(M, spec.threads[2])
        Ngroups = cld(N, spec.threads[3])
        spec.blocks = (Lgroups, Mgroups, Ngroups)
    end
    kernel((L, M, N), f, x...; items = spec.threads,
        groups = spec.blocks, queue = spec.stream)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

mutable struct oneAPIReduceWorkspace{T, TP <: JACC.WkProp} <:
               JACC.ReduceWorkspace
    tmp::oneAPI.oneArray{T}
    ret::oneAPI.oneArray{T}
end

function JACC.reduce_workspace(::oneAPIBackend, init::T) where {T}
    oneAPIReduceWorkspace{T, JACC.Managed}(
        oneAPI.oneArray{T}(undef, 0), oneAPI.oneArray([init]))
end

function JACC.reduce_workspace(::oneAPIBackend, tmp::oneAPI.oneArray{T},
        init::oneAPI.oneArray{T}) where {T}
    oneAPIReduceWorkspace{T, JACC.Unmanaged}(tmp, init)
end

@inline function _init!(wk::oneAPIReduceWorkspace{T, JACC.Managed}, blocks, init) where {T}
    if length(wk.tmp) != prod(blocks)
        wk.tmp = oneAPI.oneArray{typeof(init)}(undef, blocks)
    end
    return nothing
end

@inline function _init!(wk::oneAPIReduceWorkspace{T, JACC.Unmanaged}, blocks, init) where {T}
    nothing
end

JACC.get_result(wk::oneAPIReduceWorkspace) = Base.Array(wk.ret)[]

_make_kname(base::AbstractString, sfx::AbstractString) = base * "__" * sfx
_make_kname(::Nothing, ::AbstractString) = nothing

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{oneAPIBackend},
        N::Integer, f, x...; name = nothing)
    wk = reducer.workspace
    op = reducer.op
    init = reducer.init

    kernel1 = _make_kernel(_make_kname(name, "block_reduce"),
        _parallel_reduce_oneapi, Val(256), N, op, wk.ret, init, f, x...)
    threads1 = oneAPI.launch_configuration(kernel1)

    kernel2 = _make_kernel(_make_kname(name, "grid_reduce"),
        _reduce_kernel_oneapi, Val(256), 1, op, wk.ret, init, wk.ret)
    threads2 = oneAPI.launch_configuration(kernel2)

    threads = min(threads1, threads2, 256)
    blocks = cld(N, threads)
    shmem_size = threads * sizeof(init)

    _init!(wk, blocks, init)

    kernel1(Val(threads), N, op, wk.tmp, init, f, x...; items = threads,
        groups = blocks, queue = reducer.stream)
    kernel2(Val(threads), blocks, op, wk.tmp, init, wk.ret; items = threads,
        groups = 1, queue = reducer.stream)

    if reducer.sync
        oneAPI.synchronize(reducer.stream)
    end

    return nothing
end

function JACC.parallel_reduce(f, ::oneAPIBackend, N::Integer, x...; op, init,
        name = nothing)
    ret_inst = oneAPI.oneArray{typeof(init)}(undef, 0)
    kernel1 = _make_kernel(_make_kname(name, "block_reduce"),
        _parallel_reduce_oneapi, Val(256), N, op, ret_inst, init, f, x...)

    rret = oneAPI.oneArray([init])
    kernel2 = _make_kernel(_make_kname(name, "grid_reduce"),
        _reduce_kernel_oneapi, Val(256), 1, op, ret_inst, init, rret)

    items = 256
    groups = cld(N, items)

    ret = oneAPI.oneArray{typeof(init)}(undef, groups)

    kernel1(
        Val(items), N, op, ret, init, f, x...; items = items, groups = groups)

    kernel2(Val(items), groups, op, ret, init, rret; items = items, groups = 1)
    oneAPI.synchronize()

    return Base.Array(rret)[]
end

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{oneAPIBackend},
        (M, N)::NTuple{2, Integer}, f, x...; name = nothing)
    init = reducer.init
    op = reducer.op
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    threads = (Mitems, Nitems)
    Mgroups = cld(M, threads[1])
    Ngroups = cld(N, threads[2])
    blocks = (Mgroups, Ngroups)

    wk = reducer.workspace
    _init!(wk, blocks, init)

    kernel1 = _make_kernel(_make_kname(name, "block_reduce"),
        _parallel_reduce_oneapi_MN, (M, N), op, wk.tmp, init, f, x...)
    kernel1(
        (M, N), op, wk.tmp, init, f, x...; items = threads, groups = blocks,
        queue = reducer.stream)

    kernel2 = _make_kernel(_make_kname(name, "grid_reduce"),
        _reduce_kernel_oneapi_MN, blocks, op, wk.tmp, init, wk.ret)
    kernel2(blocks, op, wk.tmp, init, wk.ret; items = threads, groups = (1, 1),
        queue = reducer.stream)

    if reducer.sync
        oneAPI.synchronize(reducer.stream)
    end

    return nothing
end

function JACC.parallel_reduce(f, ::oneAPIBackend, (M, N)::NTuple{2, Integer},
        x...; op, init, name = nothing)
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    items = (Mitems, Nitems)
    Mgroups = cld(M, Mitems)
    Ngroups = cld(N, Nitems)
    groups = (Mgroups, Ngroups)
    ret = oneAPI.oneArray{typeof(init)}(undef, groups)
    rret = oneAPI.oneArray([init])

    kernel1 = _make_kernel(_make_kname(name, "block_reduce"),
        _parallel_reduce_oneapi_MN, (M, N), op, ret, init, f, x...)
    kernel1((M, N), op, ret, init, f, x...; items = items, groups = groups)

    kernel2 = _make_kernel(_make_kname(name, "grid_reduce"),
        _reduce_kernel_oneapi_MN, groups, op, ret, init, rret)
    kernel2(groups, op, ret, init, rret; items = items, groups = (1, 1))

    oneAPI.synchronize()
    return Base.Array(rret)[]
end

@inline function JACC.parallel_reduce(f, ::oneAPIBackend,
        dims::NTuple{N, Integer}, x...; op, init, kw...) where {N}
    ids = CartesianIndices(dims)
    return JACC.parallel_reduce(JACC.ReduceKernel1DND{typeof(init)}(),
        prod(dims), ids, f, x...; op = op, init = init, kw...)
end

@inline function _parallel_for_oneapi(N, f, x...)
    i = get_global_id()
    i > N && return nothing
    @inline f(i, x...)
    return nothing
end

@inline function _parallel_for_oneapi_MN(indexer, (M, N), f, x...)
    i, j = indexer()
    i > M && return nothing
    j > N && return nothing
    @inline f(i, j, x...)
    return nothing
end

@inline function _parallel_for_oneapi_LMN((L, M, N), f, x...)
    i = get_global_id(1)
    j = get_global_id(2)
    k = get_global_id(3)
    i > L && return nothing
    j > M && return nothing
    k > N && return nothing
    @inline f(i, j, k, x...)
    return nothing
end

function _parallel_reduce_oneapi(
        ::Val{shmem_length}, N, op, ret, init, f, x...) where {shmem_length}
    shared_mem = oneLocalArray(eltype(ret), shmem_length)
    i = get_global_id()
    ti = get_local_id()
    @inbounds shared_mem[ti] = init

    if i <= N
        tmp = @inline f(i, x...)
        @inbounds shared_mem[ti] = tmp
    end

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        barrier(oneAPI.LOCAL_MEM_FENCE)
        tn = 2^p
        if ti <= tn
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + tn])
        end
    end

    if (ti == 1)
        @inbounds ret[get_group_id()] = shared_mem[ti]
    end
    return nothing
end

function _reduce_kernel_oneapi(
        ::Val{shmem_length}, N, op, red, init, ret) where {shmem_length}
    shared_mem = oneLocalArray(eltype(ret), shmem_length)
    i = get_global_id()
    ii = i
    tmp = init
    for ii in i:shmem_length:N
        tmp = op(tmp, @inbounds red[ii])
    end
    @inbounds shared_mem[i] = tmp

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        barrier(oneAPI.LOCAL_MEM_FENCE)
        tn = 2^p
        if i <= tn
            @inbounds shared_mem[i] = op(shared_mem[i], shared_mem[i + tn])
        end
    end

    if (i == 1)
        @inbounds ret[1] = shared_mem[1]
    end
    return nothing
end

function _parallel_reduce_oneapi_MN((M, N), op, ret, init, f, x...)
    shared_mem = oneLocalArray(eltype(ret), (16, 16))
    i = get_global_id(1)
    j = get_global_id(2)
    ti = get_local_id(1)
    tj = get_local_id(2)
    bi = get_group_id(1)
    bj = get_group_id(2)

    @inbounds shared_mem[ti, tj] = init

    if (i <= M && j <= N)
        tmp = @inline f(i, j, x...)
        @inbounds shared_mem[ti, tj] = tmp
    end

    for n in (8, 4, 2, 1)
        oneAPI.barrier(oneAPI.LOCAL_MEM_FENCE)
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

function _reduce_kernel_oneapi_MN((M, N), op, red, init, ret)
    shared_mem = oneLocalArray(eltype(ret), (16, 16))
    i = get_local_id(1)
    j = get_local_id(2)

    tmp = init
    for ci in CartesianIndices((i:16:M, j:16:N))
        tmp = op(tmp, @inbounds red[ci])
    end
    @inbounds shared_mem[i, j] = tmp

    for n in (8, 4, 2, 1)
        barrier(oneAPI.LOCAL_MEM_FENCE)
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

function _make_shared_similar(x::oneDeviceArray{T, N, A}) where {T, N, A}
    shmem_data = oneLocalArray(eltype(x), 512)
    return Base.ReshapedArray(shmem_data, size(x), ())
end

function JACC.shared(::oneAPIBackend, x::AbstractVector)
    len = length(x)
    shmem = _make_shared_similar(x)
    # 1D kernel or 2D kernel at y == 1 (to avoid concurrent writes)
    if get_local_size(2) == 1 || get_local_id(2) == 1
        for i in get_local_id(1):get_local_size(1):len
            @inbounds shmem[i] = x[i]
        end
    end
    barrier(oneAPI.LOCAL_MEM_FENCE)
    return shmem
end

function JACC.shared(::oneAPIBackend, x::AbstractMatrix)
    len = length(x)
    shmem = _make_shared_similar(x)
    num_threads = get_local_size(1) * get_local_size(2)
    if get_local_size(2) == 1
        # TODO: 1D kernel with 2D array
    else
        if len <= num_threads
            i_local = get_local_id(1)
            j_local = get_local_id(2)
            @inbounds shmem[i_local, j_local] = x[i_local, j_local]
        else
            for i in get_local_id(1):get_local_size(1):size(x, 1)
                for j in get_local_id(2):get_local_size(2):size(x, 2)
                    @inbounds shmem[i, j] = x[i, j]
                end
            end
        end
    end
    barrier(oneAPI.LOCAL_MEM_FENCE)
    return shmem
end

JACC.sync_workgroup(::oneAPIBackend) = oneAPI.barrier(oneAPI.LOCAL_MEM_FENCE)

JACC.array_type(::oneAPIBackend) = oneAPI.oneArray

JACC.array(::oneAPIBackend, x::Array) = oneAPI.oneArray(x)

DefaultFloat = Union{Type, Nothing}

function _get_default_float()
    if oneL0.module_properties(device()).fp64flags &
       oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64
        return Float64
    else
        @info """Float64 unsupported on the current device.
        Default float for JACC.jl changed to Float32.
        """
        return Float32
    end
end

function JACC.default_float(::oneAPIBackend)
    global DefaultFloat
    if isa(nothing, DefaultFloat)
        DefaultFloat = _get_default_float()
    end
    return DefaultFloat
end

include("multi.jl")

end # module oneAPIExt
