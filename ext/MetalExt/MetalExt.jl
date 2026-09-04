module MetalExt

using JACC, Metal

# overloaded array functions
include("array.jl")

JACC.get_backend(::Val{:metal}) = MetalBackend()

default_stream() = Metal.global_queue(Metal.device())

JACC.default_stream(::MetalBackend) = default_stream()

function JACC.create_stream(::MetalBackend)
    Metal.MTL.MTLCommandQueue(Metal.device())
end

function JACC.synchronize(::MetalBackend)
    Metal.synchronize()
end

@inline _kernel_args(args...) = Metal.mtlconvert.((args))

@inline function _make_kernel(kernel_function, kargs, ::Nothing)
    p_tt = Tuple{Core.Typeof.(kargs)...}
    return Metal.mtlfunction(kernel_function, p_tt)
end

@inline function _make_kernel(kernel_function, kargs, kname::AbstractString)
    p_tt = Tuple{Core.Typeof.(kargs)...}
    return Metal.mtlfunction(kernel_function, p_tt; name = kname)
end

@inline function _kernel_maxthreads(kernel_function, kargs, kname)
    p_kernel = _make_kernel(kernel_function, kargs, kname)
    return (p_kernel, p_kernel.maxthreads)
end

@inline function _launch(::Nothing, threads, groups, f, args...)
    @metal threads=threads groups=groups f(args...)
end

@inline function _launch(kname::AbstractString, threads, groups, f, args...)
    @metal name=kname threads=threads groups=groups f(args...)
end

function JACC.parallel_for(f, ::MetalBackend, N::Integer, x...; name = nothing)
    kargs = _kernel_args(N, f, x...)
    kernel, max_threads = _kernel_maxthreads(_parallel_for_metal, kargs, name)
    threads = min(N, max_threads)
    groups = cld(N, threads)
    kernel(kargs...; threads = threads, groups = groups)
    Metal.synchronize()
end

function JACC.parallel_for(
        f, ::MetalBackend, (M, N)::NTuple{2, Integer}, x...; name = nothing)
    maxItems = 32
    Mthreads = min(M, maxItems)
    Nthreads = min(N, maxItems)
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, threads[1])
    Nblocks = cld(N, threads[2])
    blocks = (Mblocks, Nblocks)

    kargs = _kernel_args(M, N, f, x...)
    kernel = _make_kernel(_parallel_for_metal_MN, kargs, name)
    kernel(kargs...; threads = threads, groups = blocks)
    Metal.synchronize()
end

function JACC.parallel_for(
        f, ::MetalBackend, (L, M, N)::NTuple{3, Integer}, x...; name = nothing)
    maxItems = 32
    Lthreads = min(L, maxItems)
    Mthreads = min(M, maxItems)
    Nthreads = 1
    threads = (Lthreads, Mthreads, Nthreads)
    Lblocks = cld(L, threads[1])
    Mblocks = cld(M, threads[2])
    Nblocks = cld(N, threads[3])
    blocks = (Lblocks, Mblocks, Nblocks)

    kargs = _kernel_args(L, M, N, f, x...)
    kernel = _make_kernel(_parallel_for_metal_LMN, kargs, name)
    kernel(kargs...; threads = threads, groups = blocks)
    Metal.synchronize()
end

function JACC.parallel_for(
        f, spec::LaunchSpec{MetalBackend}, N::Integer, x...; name = nothing)
    kargs = _kernel_args(N, f, x...)
    kernel, max_threads = _kernel_maxthreads(_parallel_for_metal, kargs, name)
    if spec.threads == 0
        maxItems = max_threads
        spec.threads = min(N, maxItems)
    end
    if spec.blocks == 0
        spec.blocks = cld(N, spec.threads)
    end

    kernel(kargs...; threads = spec.threads, groups = spec.blocks, queue = spec.stream)

    if spec.sync
        Metal.synchronize()
    end
end

function JACC.parallel_for(
        f, spec::LaunchSpec{MetalBackend}, (M, N)::NTuple{2, Integer}, x...;
        name = nothing)
    if spec.threads == 0
        maxItems = 32
        Mthreads = min(M, maxItems)
        Nthreads = min(N, maxItems)
        spec.threads = (Mthreads, Nthreads)
    end
    if spec.blocks == 0
        Mblocks = cld(M, spec.threads[1])
        Nblocks = cld(N, spec.threads[2])
        spec.blocks = (Mblocks, Nblocks)
    end

    kargs = _kernel_args(M, N, f, x...)
    kernel = _make_kernel(_parallel_for_metal_MN, kargs, name)
    kernel(kargs...; threads = spec.threads, groups = spec.blocks, queue = spec.stream)

    if spec.sync
        Metal.synchronize()
    end
end

function JACC.parallel_for(
        f, spec::LaunchSpec{MetalBackend}, (L, M, N)::NTuple{3, Integer}, x...;
        name = nothing)
    if spec.threads == 0
        maxItems = 32
        Lthreads = min(L, maxItems)
        Mthreads = min(M, maxItems)
        Nthreads = 1
        spec.threads = (Lthreads, Mthreads, Nthreads)
    end
    if spec.blocks == 0
        Lblocks = cld(L, spec.threads[1])
        Mblocks = cld(M, spec.threads[2])
        Nblocks = cld(N, spec.threads[3])
        spec.blocks = (Lblocks, Mblocks, Nblocks)
    end

    kargs = _kernel_args(L, M, N, f, x...)
    kernel = _make_kernel(_parallel_for_metal_LMN, kargs, name)
    kernel(kargs...; threads = spec.threads, groups = spec.blocks, queue = spec.stream)

    if spec.sync
        Metal.synchronize()
    end
end

mutable struct MetalReduceWorkspace{T, TP <: JACC.WkProp} <:
               JACC.ReduceWorkspace
    tmp::Metal.MtlArray{T}
    ret::Metal.MtlArray{T}
end

function JACC.reduce_workspace(::MetalBackend, init::T) where {T}
    MetalReduceWorkspace{T, JACC.Managed}(
        Metal.MtlArray{T}(undef, 0), Metal.MtlArray{T}([init]))
end

function JACC.reduce_workspace(::MetalBackend, tmp::Metal.MtlArray{T},
        init::Metal.MtlArray{T}) where {T}
    MetalReduceWorkspace{T, JACC.Unmanaged}(tmp, init)
end

@inline function _init!(
        wk::MetalReduceWorkspace{T, JACC.Managed}, groups, init) where {T}
    if length(wk.tmp) != prod(groups)
        wk.tmp = Metal.MtlArray{typeof(init)}(undef, groups)
    end
    return nothing
end

@inline function _init!(
        wk::MetalReduceWorkspace{T, JACC.Unmanaged}, groups, init) where {T}
    nothing
end

JACC.get_result(wk::MetalReduceWorkspace) = Base.Array(wk.ret)[]

_make_kname(base::AbstractString, sfx::AbstractString) = base * "__" * sfx
_make_kname(::Nothing, ::AbstractString) = nothing

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{MetalBackend},
        N::Integer, f, x...; name = nothing)
    wk = reducer.workspace
    op = reducer.op
    init = reducer.init

    kargs1 = _kernel_args(Val(512), N, op, wk.tmp, init, f, x...)
    kernel1,
    threads1 = _kernel_maxthreads(_parallel_reduce_metal, kargs1,
        _make_kname(name, "block_reduce"))

    kargs2 = _kernel_args(Val(512), 1, op, wk.tmp, init, wk.ret)
    kernel2,
    threads2 = _kernel_maxthreads(_reduce_kernel_metal, kargs2,
        _make_kname(name, "grid_reduce"))

    threads = min(threads1, threads2, 512)
    groups = cld(N, threads)

    _init!(wk, groups, init)

    kernel1(Val(threads), N, op, wk.tmp, init, f, x...; threads = threads,
        groups = groups, queue = reducer.stream)
    kernel2(Val(threads), groups, op, wk.tmp, init, wk.ret; threads = threads,
        groups = 1, queue = reducer.stream)

    if reducer.sync
        Metal.synchronize()
    end

    return nothing
end

function JACC.parallel_reduce(f, ::MetalBackend, N::Integer, x...; op, init,
        name = nothing)
    ret_inst = Metal.MtlArray{typeof(init)}(undef, 0)
    kargs1 = _kernel_args(Val(512), N, op, ret_inst, init, f, x...)
    kernel1,
    threads1 = _kernel_maxthreads(_parallel_reduce_metal, kargs1, _make_kname(name, "block_reduce"))

    rret = Metal.MtlArray([init])
    kargs2 = _kernel_args(Val(512), 1, op, ret_inst, init, rret)
    kernel2,
    threads2 = _kernel_maxthreads(_reduce_kernel_metal, kargs2, _make_kname(name, "grid_reduce"))

    threads = min(threads1, threads2, 512)
    groups = cld(N, threads)

    ret = Metal.MtlArray{typeof(init)}(undef, groups)

    kernel1(Val(threads), N, op, ret, init, f, x...;
        threads = threads, groups = groups)
    kernel2(Val(threads), groups, op, ret, init,
        rret; threads = threads, groups = 1)
    Metal.synchronize()

    return Base.Array(rret)[]
end

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{MetalBackend},
        (M, N)::NTuple{2, Integer}, f, x...; name = nothing)
    wk = reducer.workspace
    op = reducer.op
    init = reducer.init
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    threads = (Mitems, Nitems)
    Mgroups = cld(M, threads[1])
    Ngroups = cld(N, threads[2])
    groups = (Mgroups, Ngroups)

    _init!(wk, groups, init)

    println("reducer:")
    @show wk.tmp
    kargs1 = _kernel_args((M, N), op, wk.tmp, init, f, x...)
    kernel1 = _make_kernel(_parallel_reduce_metal_MN, kargs1, _make_kname(name, "block_reduce"))
    kernel1(kargs1...; threads = threads, groups = groups, queue = reducer.stream)

    @show wk.tmp
    @show wk.ret
    kargs2 = _kernel_args(groups, op, wk.tmp, init, wk.ret)
    kernel2 = _make_kernel(_reduce_kernel_metal_MN, kargs2, _make_kname(name, "grid_reduce"))
    kernel2(kargs2...; threads = threads, groups = (1, 1), queue = reducer.stream)
    @show wk.ret

    if reducer.sync
        Metal.synchronize()
    end

    return nothing
end

function JACC.parallel_reduce(f, ::MetalBackend, (M, N)::NTuple{2, Integer},
        x...; op, init, name = nothing)
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    items = (Mitems, Nitems)
    Mgroups = cld(M, Mitems)
    Ngroups = cld(N, Nitems)
    groups = (Mgroups, Ngroups)

    ret = Metal.MtlArray{typeof(init)}(undef, groups)
    rret = Metal.MtlArray([init])

    println("standard:")
    @show ret
    kargs1 = _kernel_args((M, N), op, ret, init, f, x...)
    kernel1 = _make_kernel(_parallel_reduce_metal_MN, kargs1, _make_kname(name, "block_reduce"))
    kernel1(kargs1...; threads = items, groups = groups)

    @show ret
    @show rret
    kargs2 = _kernel_args(groups, op, ret, init, rret)
    kernel2 = _make_kernel(_reduce_kernel_metal_MN, kargs2, _make_kname(name, "grid_reduce"))
    kernel2(kargs2...; threads = items, groups = (1, 1))
    @show rret

    Metal.synchronize()
    return Base.Array(rret)[]
end

@inline function JACC.parallel_reduce(f, ::MetalBackend,
        dims::NTuple{N, Integer}, x...; op, init, kw...) where {N}
    ids = CartesianIndices(dims)
    return JACC.parallel_reduce(JACC.ReduceKernel1DND{typeof(init)}(),
        prod(dims), ids, f, x...; op = op, init = init, kw...)
end

# COV_EXCL_START
function _parallel_reduce_metal(
        ::Val{shmem_length}, N, op, ret, init, f, x...) where {shmem_length}
    shared_mem = MtlThreadGroupArray(eltype(ret), shmem_length)
    i = thread_position_in_grid().x
    ti = thread_position_in_threadgroup().x

    @inbounds shared_mem[ti] = init

    if i <= N
        tmp = @inline f(i, x...)
        @inbounds shared_mem[ti] = tmp
    end

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        tn = 2^p
        if ti <= tn
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + tn])
        end
    end

    if (ti == 1)
        @inbounds ret[threadgroup_position_in_grid().x] = shared_mem[ti]
    end
    return nothing
end

function _reduce_kernel_metal(
        ::Val{shmem_length}, N, op, red, init, ret) where {shmem_length}
    shared_mem = MtlThreadGroupArray(eltype(ret), shmem_length)
    i = thread_position_in_grid().x
    ii = i
    tmp = init
    for ii in i:shmem_length:N
        tmp = op(tmp, @inbounds red[ii])
    end
    @inbounds shared_mem[i] = tmp

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
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

function _parallel_reduce_metal_MN((M, N), op, ret, init, f, x...)
    shared_mem = MtlThreadGroupArray(eltype(ret), (16, 16))
    i = thread_position_in_grid().x
    j = thread_position_in_grid().y
    ti = thread_position_in_threadgroup().x
    tj = thread_position_in_threadgroup().y
    bi = threadgroup_position_in_grid().x
    bj = threadgroup_position_in_grid().y

    @inbounds shared_mem[ti, tj] = init

    if (i <= M && j <= N)
        tmp = @inline f(i, j, x...)
        @inbounds shared_mem[ti, tj] = tmp
    end

    for n in (8, 4, 2, 1)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
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

function _reduce_kernel_metal_MN((M, N), op, red, init, ret)
    shared_mem = MtlThreadGroupArray(eltype(ret), (16, 16))
    i = thread_position_in_threadgroup().x
    j = thread_position_in_threadgroup().y

    tmp = init
    for ci in CartesianIndices((i:16:M, j:16:N))
        tmp = op(tmp, @inbounds red[ci])
    end
    @inbounds shared_mem[i, j] = tmp

    for n in (8, 4, 2, 1)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
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

function _parallel_for_metal(N, f, x...)
    i = Metal.thread_position_in_grid_1d()
    if i > N
        return nothing
    end
    f(i, x...)
    return nothing
end

function _parallel_for_metal_MN(M, N, f, x...)
    i = Metal.thread_position_in_grid().x
    j = Metal.thread_position_in_grid().y
    if i > M || j > N
        return nothing
    end
    f(i, j, x...)
    return nothing
end

function _parallel_for_metal_LMN(L, M, N, f, x...)
    i = Metal.thread_position_in_grid().x
    j = Metal.thread_position_in_grid().y
    k = Metal.thread_position_in_grid().z
    if i > L || j > M || k > N
        return nothing
    end
    f(i, j, k, x...)
    return nothing
end
# COV_EXCL_STOP

function JACC.default_float(::MetalBackend)
    return Float32
end

function JACC.shared(::MetalBackend, x::AbstractArray)
    shmem = Metal.mtl(x; storage = Metal.SharedStorage)
    # Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    return shmem
end

JACC.sync_workgroup(::MetalBackend) = Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

JACC.array_type(::MetalBackend) = Metal.MtlArray

function _compute_array_storage()
    preferences = get(
        JACC.Preferences.Backend._EXT_PREFS[], "metal", Dict{Symbol, Any}())
    value = get(preferences, :storage, "private")
    value isa Union{AbstractString, Symbol} || throw(ArgumentError(
        "Invalid Metal array storage: $(repr(value)); " *
        "expected :private or :shared"))
    storage = lowercase(String(value))
    storage in ("private", "shared") || throw(ArgumentError(
        "Invalid Metal array storage: $(repr(value)); " *
        "expected :private or :shared"))
    return storage
end

# Cached module global rather than a plain module-load-once value: `storage`
# is one of the extension preferences `set_backend` is documented (and
# tested, see array_storage_preference in test/backend/metal.jl) to apply
# live within the same Julia session, without a restart. _EXT_PREFS_GENERATION
# is bumped on every write, so this stays a cheap Int comparison on the
# array-allocation hot path instead of a Dict lookup plus revalidation on
# every call, while still tracking live changes.
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

function JACC._array(::MetalBackend, x::AbstractArray)
    if _array_storage() == "shared"
        return Metal.MtlArray{eltype(x), ndims(x), Metal.SharedStorage}(x)
    end
    return Metal.MtlArray{eltype(x), ndims(x), Metal.PrivateStorage}(x)
end

JACC._array(::MetalBackend, x::Metal.MtlArray) = x
JACC.array(backend::MetalBackend, x::Base.Array) = JACC._array(backend, x)
JACC.array(::MetalBackend, x::Metal.MtlArray) = x

end # module MetalExt
