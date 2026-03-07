module MetalExt

using JACC, Metal

# overloaded array functions
include("array.jl")

JACC.get_backend(::Val{:metal}) = MetalBackend()

default_stream() = Metal.global_queue(Metal.device())

JACC.default_stream(::MetalBackend) = default_stream()

function JACC.synchronize(::MetalBackend)
    Metal.synchronize()
end

function dummy() end

@inline function _kernel_maxthreads(N, f, x...)
    kernel = @metal launch=false dummy()
    max_threads = Int32(kernel.pipeline.maxTotalThreadsPerThreadgroup)
    return max_threads
end

function JACC.parallel_for(f, ::MetalBackend, N::Integer, x...)
    max_threads_per_group = _kernel_maxthreads(N, f, x)
    threads = min(N, max_threads_per_group)
    groups = cld(N, threads)
    @metal threads=threads groups=groups _parallel_for_metal(N, f, x...)
    Metal.synchronize()
end

function JACC.parallel_for(
        f, ::MetalBackend, (M, N)::NTuple{2, Integer}, x...)
    maxItems = 32
    Mthreads = min(M, maxItems)
    Nthreads = min(N, maxItems)
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, threads[1])
    Nblocks = cld(N, threads[2])
    blocks = (Mblocks, Nblocks)

    Metal.@sync @metal threads=threads groups=blocks _parallel_for_metal_MN(
        M, N, f, x...)
end

function JACC.parallel_for(
        f, ::MetalBackend, (L, M, N)::NTuple{3, Integer}, x...)
    maxItems = 32
    Lthreads = min(L, maxItems)
    Mthreads = min(M, maxItems)
    Nthreads = 1
    threads = (Lthreads, Mthreads, Nthreads)
    Lblocks = cld(L, threads[1])
    Mblocks = cld(M, threads[2])
    Nblocks = cld(N, threads[3])
    blocks = (Lblocks, Mblocks, Nblocks)

    Metal.@sync @metal threads=threads groups=blocks _parallel_for_metal_LMN(
        L, M, N, f, x...)
end

function JACC.parallel_for(
        f, spec::LaunchSpec{MetalBackend}, N::Integer, x...)
    max_threads_per_group = _kernel_maxthreads(N, f, x)
    if spec.threads == 0
        maxItems = max_threads_per_group
        spec.threads = min(N, maxItems)
    end
    if spec.blocks == 0
        spec.blocks = cld(N, spec.threads)
    end

    @metal threads=spec.threads groups=spec.blocks _parallel_for_metal(
        N, f, x...)

    if spec.sync
        Metal.synchronize()
    end
end

function JACC.parallel_for(
        f, spec::LaunchSpec{MetalBackend}, (M, N)::NTuple{2, Integer}, x...)
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

    @metal threads=spec.threads groups=spec.blocks _parallel_for_metal_MN(
        M, N, f, x...)

    if spec.sync
        Metal.synchronize()
    end
end

function JACC.parallel_for(
        f, spec::LaunchSpec{MetalBackend}, (L, M, N)::NTuple{3, Integer}, x...)
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

    @metal threads=spec.threads groups=spec.blocks _parallel_for_metal_LMN(
        L, M, N, f, x...)

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
    fill!(wk.tmp, init)
    fill!(wk.ret, init)
    return nothing
end

@inline function _init!(
        wk::MetalReduceWorkspace{T, JACC.Unmanaged}, groups, init) where {T}
    nothing
end

JACC.get_result(wk::MetalReduceWorkspace) = Base.Array(wk.ret)[]

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{MetalBackend},
        N::Integer, f, x...)
    wk = reducer.workspace
    op = reducer.op
    init = reducer.init

    items = _kernel_maxthreads(N, f, x)
    groups = cld(N, items)
    shmem_size = items * sizeof(init)

    _init!(wk, groups, init)

    @metal threads=items groups=groups _parallel_reduce_metal(
        Val(items), N, op, wk.tmp, f, x...)
    @metal threads=items groups=1 reduce_kernel_metal(
        Val(items), groups, op, wk.tmp, wk.ret)

    if reducer.sync
        Metal.synchronize()
    end

    return nothing
end

function JACC.parallel_reduce(f, ::MetalBackend, N::Integer, x...; op, init)
    items = _kernel_maxthreads(N, f, x)
    groups = cld(N, items)
    ret = fill!(Metal.MtlArray{typeof(init)}(undef, groups), init)
    rret = Metal.MtlArray([init])
    Metal.@sync @metal threads=items groups=groups _parallel_reduce_metal(
        Val(items), N, op, ret, f, x...)
    Metal.@sync @metal threads=items groups=1 reduce_kernel_metal(
        Val(items), groups, op, ret, rret)
    return Base.Array(rret)[]
end

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{MetalBackend},
        (M, N)::NTuple{2, Integer}, f, x...)
    init = reducer.init
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    threads = (Mitems, Nitems)
    Mgroups = cld(M, threads[1])
    Ngroups = cld(N, threads[2])
    blocks = (Mgroups, Ngroups)

    wk = reducer.workspace
    _init!(wk, blocks, init)

    @metal threads=threads groups=blocks _parallel_reduce_metal_MN(
        (M, N), reducer.op, wk.tmp, f, x...)
    @metal threads=threads groups=(1, 1) reduce_kernel_metal_MN(
        blocks, reducer.op, wk.tmp, wk.ret)

    if reducer.sync
        Metal.synchronize()
    end

    return nothing
end

function JACC.parallel_reduce(f, ::MetalBackend, (M, N)::NTuple{2, Integer},
        x...; op, init)
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    items = (Mitems, Nitems)
    Mgroups = cld(M, Mitems)
    Ngroups = cld(N, Nitems)
    groups = (Mgroups, Ngroups)
    ret = fill!(Metal.MtlArray{typeof(init)}(undef, (Mgroups, Ngroups)), init)
    rret = Metal.MtlArray([init])
    Metal.@sync @metal threads=items groups=groups _parallel_reduce_metal_MN(
        (M, N), op, ret, f, x...)
    Metal.@sync @metal threads=items groups=(1, 1) reduce_kernel_metal_MN(
        groups, op, ret, rret)
    return Base.Array(rret)[]
end

@inline function JACC.parallel_reduce(f, ::MetalBackend,
        dims::NTuple{N, Integer}, x...; op, init) where {N}
    ids = CartesianIndices(dims)
    return JACC.parallel_reduce(JACC.ReduceKernel1DND{typeof(init)}(),
        prod(dims), ids, f, x...; op = op, init = init)
end

# COV_EXCL_START
function _parallel_reduce_metal(
        ::Val{shmem_length}, N, op, ret, f, x...) where {shmem_length}
    shared_mem = MtlThreadGroupArray(eltype(ret), shmem_length)
    i = thread_position_in_grid().x
    ti = thread_position_in_threadgroup().x

    @inbounds shared_mem[ti] = ret[threadgroup_position_in_grid().x]

    if i <= N
        tmp = @inline f(i, x...)
        @inbounds shared_mem[ti] = tmp
    end

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        threadgroup_barrier()
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

function reduce_kernel_metal(
        ::Val{shmem_length}, N, op, red, ret) where {shmem_length}
    shared_mem = MtlThreadGroupArray(eltype(ret), shmem_length)
    i = thread_position_in_grid().x
    ii = i
    @inbounds tmp = ret[1]
    for ii in i:shmem_length:N
        tmp = op(tmp, @inbounds red[ii])
    end
    @inbounds shared_mem[i] = tmp

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        threadgroup_barrier()
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

function _parallel_reduce_metal_MN((M, N), op, ret, f, x...)
    shared_mem = MtlThreadGroupArray(eltype(ret), (16, 16))
    i = thread_position_in_grid().x
    j = thread_position_in_grid().y
    ti = thread_position_in_threadgroup().x
    tj = thread_position_in_threadgroup().y
    bi = threadgroup_position_in_grid().x
    bj = threadgroup_position_in_grid().y

    @inbounds shared_mem[ti, tj] = ret[bi, bj]

    if (i <= M && j <= N)
        tmp = @inline f(i, j, x...)
        @inbounds shared_mem[ti, tj] = tmp
    end

    for n in (8, 4, 2, 1)
        threadgroup_barrier()
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

function reduce_kernel_metal_MN((M, N), op, red, ret)
    shared_mem = MtlThreadGroupArray(eltype(ret), (16, 16))
    i = thread_position_in_threadgroup().x
    j = thread_position_in_threadgroup().y

    @inbounds tmp = ret[1]
    for ci in CartesianIndices((i:16:M, j:16:N))
        tmp = op(tmp, @inbounds red[ci])
    end
    @inbounds shared_mem[i, j] = tmp

    for n in (8, 4, 2, 1)
        threadgroup_barrier()
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
    # Metal.threadgroup_barrier()
    return shmem
end

JACC.sync_workgroup(::MetalBackend) = Metal.threadgroup_barrier()

JACC.array_type(::MetalBackend) = Metal.MtlArray

JACC.array(::MetalBackend, x::Base.Array) = Metal.MtlArray(x)

end # module MetalExt