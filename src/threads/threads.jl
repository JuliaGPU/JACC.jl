module ThreadsImpl

import Polyester

import JACC
import JACC: LaunchSpec

struct ThreadsBackend end

@inline JACC.get_backend(::Val{:threads}) = ThreadsBackend()

function _maybe_threaded(ex)
    quote
        if Threads.nthreads() == 1
            $ex
        else
            Polyester.@batch $ex
        end
    end
end

macro maybe_threaded(ex)
    esc(_maybe_threaded(ex))
end

include("array.jl")
include("multi.jl")
include("async.jl")
include("experimental/experimental.jl")

JACC.synchronize(::ThreadsBackend) = nothing

JACC.default_stream(::ThreadsBackend) = nothing

JACC.create_stream(::ThreadsBackend) = nothing

@inline function JACC.parallel_for(f, ::ThreadsBackend, N::Integer, x...)
    @maybe_threaded for i in 1:N
        f(i, x...)
    end
end

@inline function JACC.parallel_for(
        f, spec::LaunchSpec{ThreadsBackend}, N::Integer, x...)
    if spec.threads == 0
        JACC.parallel_for(f, ThreadsBackend(), N, x...)
    else
        _BARRIER[] = Detail.SimpleBarrier(spec.threads)
        fetch.([Threads.@spawn f(i, x...) for i in 1:N])
        _BARRIER[] = Detail.SimpleBarrier(Threads.nthreads())
    end
end

@inline function JACC.parallel_for(
        f, ::ThreadsBackend, (M, N)::NTuple{2, Integer}, x...)
    @maybe_threaded for ij in CartesianIndices((M, N))
        f(ij[1], ij[2], x...)
    end
end

@inline function JACC.parallel_for(f, spec::LaunchSpec{ThreadsBackend},
        (M, N)::NTuple{2, Integer}, x...)
    ids = CartesianIndices((M, N))
    JACC.parallel_for(i -> f(ids[i][1], ids[i][2], x...),
        JACC.launch_spec(threads = prod(spec.threads)), length(ids))
end

@inline function JACC.parallel_for(
        f, ::ThreadsBackend, (L, M, N)::NTuple{3, Integer}, x...)
    @maybe_threaded for ijk in CartesianIndices((L, M, N))
        f(ijk[1], ijk[2], ijk[3], x...)
    end
end

@inline function JACC.parallel_for(f, spec::LaunchSpec{ThreadsBackend},
        (L, M, N)::NTuple{3, Integer}, x...)
    ids = CartesianIndices((L, M, N))
    JACC.parallel_for(i -> f(ids[i][1], ids[i][2], ids[i][3], x...),
        JACC.launch_spec(threads = prod(spec.threads)), length(ids))
end

mutable struct ThreadsReduceWorkspace{T} <: JACC.ReduceWorkspace
    tmp::Vector{T}
    ret::Vector{T}
end

@inline function JACC.reduce_workspace(::ThreadsBackend, init::T) where {T}
    if Threads.nthreads() == 1
        ThreadsReduceWorkspace{T}(T[], [init])
    else
        ThreadsReduceWorkspace{T}(Vector{T}(undef, Threads.nthreads()), [init])
    end
end

@inline JACC.get_result(wk::ThreadsReduceWorkspace{T}) where {T} = wk.ret[]::T

@inline function _serial_reduce!(reducer::JACC.ParallelReduce{ThreadsBackend},
        N::Integer, f, x...)
    wk = reducer.workspace
    op = reducer.op
    tmp = reducer.init
    for i in 1:N
        tmp = op(tmp, f(i, x...))
    end
    wk.ret[] = tmp
    return nothing
end

@inline function _chunk_reduce!(reducer::JACC.ParallelReduce{ThreadsBackend},
        N::Integer, f, x...)
    wk = reducer.workspace
    op = reducer.op
    nchunks = Threads.nthreads()
    chunks = collect(Base.Iterators.partition(1:N, cld(N, nchunks)))
    nchunks = length(chunks)
    Polyester.@batch for n in 1:nchunks
        @inbounds begin
            tp = reducer.init
            for i in chunks[n]
                tp = op(tp, f(i, x...))
            end
            wk.tmp[n] = tp
        end
    end
    wk.ret[] = reduce(op, @view wk.tmp[1:nchunks])
    return nothing
end

@inline function JACC._parallel_reduce!(
        reducer::JACC.ParallelReduce{ThreadsBackend}, N::Integer, f, x...)
    if Threads.nthreads() == 1
        _serial_reduce!(reducer, N, f, x...)
    else
        _chunk_reduce!(reducer, N, f, x...)
    end
    return nothing
end

@inline function JACC.parallel_reduce(
        f, ::ThreadsBackend, N::Integer, x...; op, init)
    reducer = JACC.reducer(
        backend = ThreadsBackend(), range = N, op = op, init = init)
    reducer(f, x...)
    return JACC.get_result(reducer)
end

@inline function _serial_reduce!(reducer::JACC.ParallelReduce{ThreadsBackend},
        (M, N)::NTuple{2, Integer}, f, x...)
    wk = reducer.workspace
    op = reducer.op
    tmp = reducer.init
    for j in 1:N
        for i in 1:M
            tmp = op(tmp, f(i, j, x...))
        end
    end
    wk.ret[] = tmp
    return nothing
end

@inline function _chunk_reduce!(reducer::JACC.ParallelReduce{ThreadsBackend},
        (M, N)::NTuple{2, Integer}, f, x...)
    wk = reducer.workspace
    op = reducer.op
    ids = CartesianIndices((1:M, 1:N))
    nchunks = Threads.nthreads()
    chunks = collect(Base.Iterators.partition(ids, cld(length(ids), nchunks)))
    nchunks = length(chunks)
    Polyester.@batch for n in 1:nchunks
        @inbounds begin
            tp = reducer.init
            for ij in chunks[n]
                tp = op(tp, f(ij[1], ij[2], x...))
            end
            wk.tmp[n] = tp
        end
    end
    wk.ret[] = reduce(op, @view wk.tmp[1:nchunks])
    return nothing
end

@inline function JACC._parallel_reduce!(
        reducer::JACC.ParallelReduce{ThreadsBackend},
        (M, N)::NTuple{2, Integer}, f, x...)
    if Threads.nthreads() == 1
        _serial_reduce!(reducer, (M, N), f, x...)
    else
        _chunk_reduce!(reducer, (M, N), f, x...)
    end
    return nothing
end

@inline function JACC.parallel_reduce(f, ::ThreadsBackend,
        (M, N)::NTuple{2, Integer}, x...; op, init)
    dims = (M, N)
    reducer = JACC.reducer(
        backend = ThreadsBackend(), range = dims, op = op, init = init)
    reducer(f, x...)
    return JACC.get_result(reducer)
end

@inline function JACC.parallel_reduce(
        f, ::ThreadsBackend, dims::NTuple{N, Integer},
        x...; op, init)::typeof(init) where {N}
    ids = CartesianIndices(dims)
    return JACC.parallel_reduce(
        JACC.ReduceKernel1DND{typeof(init)}(), prod(dims), ids, f,
        x...; op = op, init = init)
end

module Detail

mutable struct SimpleBarrier
    const n::Int64
    const c::Threads.Condition
    cnt::Int64

    function SimpleBarrier(n::Integer)
        new(n, Threads.Condition(), 0)
    end
end

function Base.wait(b::SimpleBarrier)
    lock(b.c)
    try
        b.cnt += 1
        if b.cnt == b.n
            b.cnt = 0
            notify(b.c)
        else
            wait(b.c)
        end
    finally
        unlock(b.c)
    end
end

end # module Detail

const _BARRIER = Ref(Detail.SimpleBarrier(0))

JACC.sync_workgroup(::ThreadsBackend) = wait(_BARRIER[])

JACC.array_type(::ThreadsBackend) = Base.Array

JACC.array(::ThreadsBackend, x::AbstractArray) = x

JACC.shared(::ThreadsBackend, x::AbstractArray) = x

function __init__()
    _BARRIER[] = Detail.SimpleBarrier(Threads.nthreads())
end

end
