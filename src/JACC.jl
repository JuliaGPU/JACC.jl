
module JACC

import Atomix: @atomic

# module to set backend preferences
include("preferences.jl")

get_backend(backend::Symbol) = get_backend(Val(backend))

@inline default_backend() = get_backend(_backend_dispatchable)

const IDims = Union{Integer, NTuple{2, Integer}, NTuple{3, Integer}}
const AllDims = Union{Integer, NTuple{N, Integer}} where {N}

include("array.jl")
include("blas.jl")
include("multi.jl")
include("async.jl")
include("experimental/experimental.jl")

export array_type, array
export default_float
export @atomic
export parallel_for, parallel_reduce
export shared
export LaunchSpec
export synchronize

public zeros, ones, fill
public to_device, to_host, transfer!
public @parallel_for, @parallel_reduce

ilog2(n::T) where {T <: Integer} = sizeof(T) * 8 - 1 - leading_zeros(n)

default_stream() = default_stream(default_backend())

create_stream() = create_stream(default_backend())

@kwdef mutable struct LaunchSpec{Backend}
    stream = default_stream(Backend())
    threads = 0
    blocks = 0
    shmem_size::Int = -1
    sync::Bool = true
end

launch_spec(; kw...) = LaunchSpec{typeof(default_backend())}(; kw...)

default_float(::Any) = Float64

shared(x::AbstractArray) = shared(default_backend(), x)

sync_workgroup() = sync_workgroup(default_backend())

default_float() = default_float(default_backend())

synchronize(; kw...) = synchronize(default_backend(); kw...)

@inline function parallel_for(dims::AllDims, f, x...)
    parallel_for(f, dims, x...)
end

@inline function parallel_for(f, dims::NTuple{N, Integer}, x...) where {N}
    ids = CartesianIndices(dims)
    @inline function _parallel_for_kernel_1d_nd(i, ids, f, x...)::Nothing
        f(Tuple(@inbounds ids[i])..., x...)
        return nothing
    end
    parallel_for(_parallel_for_kernel_1d_nd, prod(dims), ids, f, x...)
end

@inline function parallel_for(f, dims::IDims, x...)
    parallel_for(f, default_backend(), dims, x...)
end

@inline function parallel_for(spec::LaunchSpec, dims::AllDims, f, x...)
    parallel_for(f, spec, dims, x...)
end

@inline function parallel_for(spec::LaunchSpec, dims::IDims, f, x...)
    parallel_for(f, spec, dims, x...)
end

# @inline function _range_kernel(i, range, f, x...)::Nothing
#     @inbounds f(range[i], x...)
#     return nothing
# end

@inline _range_kernel(i, range, f, x...) = @inbounds f(range[i], x...)

@inline function parallel_for(f, range::AbstractRange, x...)
    parallel_for(_range_kernel, length(range), range, f, x...)
end

@inline function parallel_for(f, spec::LaunchSpec, range::AbstractRange, x...)
    parallel_for(_range_kernel, spec, length(range), range, f, x...)
end

struct _RangesKernel{N} end

@inline function (::_RangesKernel{N})(args...) where {N}
    @inbounds rt = args[N + 1]
    @inbounds return args[N + 2](getindex.(rt, args[1:N])..., args[(N + 3):end]...)
end

@inline function parallel_for(f, range::NTuple{N, AbstractRange}, x...) where {N}
    parallel_for(_RangesKernel{N}(), length.(range), range, f, x...)
end

@inline function parallel_for(f, spec::LaunchSpec,
        range::NTuple{N, AbstractRange}, x...) where {N}
    parallel_for(_RangesKernel{N}(), spec, length.(range), range, f, x...)
end

@inline function parallel_for(; range::TR, f, args::Tuple, kw...) where {TR}
    parallel_for(f, launch_spec(; kw...), range, args...)
end

default_init(::Type{T}, ::typeof(+)) where {T} = zero(T)
default_init(::Type{T}, ::typeof(*)) where {T} = one(T)
default_init(::Type{T}, ::typeof(max)) where {T} = typemin(T)
default_init(::Type{T}, ::typeof(min)) where {T} = typemax(T)
default_init(op::Function) = default_init(default_float(), op)

abstract type ReduceWorkspace end

abstract type WkProp end
struct Managed <: WkProp end
struct Unmanaged <: WkProp end

reduce_workspace() = reduce_workspace(default_backend(), default_float()())

reduce_workspace(init::T) where {T} = reduce_workspace(default_backend(), init)

@kwdef mutable struct ParallelReduce{Backend, T, Op, R, W <: ReduceWorkspace}
    range::R = zeros(Int, R)
    op::Op = () -> nothing
    init::T = default_init(T, op)
    stream = default_stream(Backend())
    sync::Bool = true
    workspace::W = reduce_workspace(Backend(), init)
end

@inline function reducer(;
        range, backend::Backend = default_backend(), type = nothing,
        op = +, init = nothing, kw...) where {Backend}
    _init = _resolve_init_type(op, type, init)
    _workspace = reduce_workspace(default_backend(), _init)
    ParallelReduce{
        Backend, typeof(_init), typeof(op), typeof(range), typeof(_workspace)}(;
        range = range, op = op, init = _init, workspace = _workspace, kw...)
end

@inline function reducer(::Type{T}, range::AllDims, op = +;
        init = default_init(T, op)) where {T}
    reducer(; type = T, range = range, op = op, init = init)
end

struct ReduceKernel1DND{T} end

@inline function (::ReduceKernel1DND{T})(i, ids, f, x...)::T where {T}
    return f(Tuple(@inbounds ids[i])..., x...)::T
end

function _parallel_reduce!(
        reducer::ParallelReduce, dims::NTuple{N, Integer}, f, x...) where {N}
    ids = CartesianIndices(dims)
    _parallel_reduce!(reducer, prod(dims),
        ReduceKernel1DND{typeof(reducer.init)}(), ids, f, x...)
end

@inline function (reducer::ParallelReduce)(f, x...)
    _parallel_reduce!(reducer, reducer.range, f, x...)
end

@inline function (reducer::ParallelReduce)(a::AbstractArray)
    reducer(elem_access, a)
end

function set_init!(reducer::ParallelReduce{B, T}, init) where {B, T}
    reducer.init = convert(T, init)
end

@inline function get_result(reducer::ParallelReduce{B, T}) where {B, T}
    get_result(reducer.workspace)::T
end

@inline _resolve_init_type(op, type, init) = convert(type, init)
@inline _resolve_init_type(op, type, init::Nothing) = default_init(type, op)
@inline _resolve_init_type(op, type::Nothing, init) = init
@inline _resolve_init_type(op, type::Nothing, init::Nothing) = default_init(op)

@inline function parallel_reduce(f, dims::AllDims, x...;
        type = nothing, op = +, init = nothing)
    _init = _resolve_init_type(op, type, init)
    return parallel_reduce(
        f, default_backend(), dims, x...; op = op, init = _init)
end

@inline function parallel_reduce(dims::AllDims, f, x...; kw...)
    return parallel_reduce(f, dims, x...; kw...)
end

@inline function parallel_reduce(f, spec::LaunchSpec{TBackend}, dims::AllDims,
        x...; type = nothing, op = +, init = nothing) where {TBackend}
    _init = _resolve_init_type(op, type, init)
    _workspace = JACC.reduce_workspace(TBackend(), _init)
    reducer = ParallelReduce{
        TBackend, typeof(_init), typeof(op), typeof(dims), typeof(_workspace)}(;
        range = dims,
        op = op,
        init = _init,
        stream = spec.stream,
        sync = spec.sync,
        workspace = _workspace
    )
    reducer(f, x...)
    return reducer.workspace.ret
end

@inline function parallel_reduce(
        spec::LaunchSpec, dims::AllDims, f, x...; kw...)
    return parallel_reduce(f, spec, dims, x...; kw...)
end

@inline function parallel_reduce(f, range::AbstractRange, x...; kw...)
    parallel_reduce(_range_kernel, length(range), range, f, x...; kw...)
end

@inline function parallel_reduce(
        f, range::NTuple{N, AbstractRange}, x...; kw...) where {N}
    parallel_reduce(_RangesKernel{N}(), length.(range), range, f, x...; kw...)
end

@inline function parallel_reduce(f, spec::LaunchSpec{TBackend},
        range::AbstractRange, x...; kw...) where {TBackend}
    parallel_reduce(_range_kernel, spec, length(range), range, f, x...; kw...)
end

@inline function parallel_reduce(f, spec::LaunchSpec{TBackend},
        range::NTuple{N, AbstractRange}, x...; kw...) where {TBackend, N}
    parallel_reduce(
        _RangesKernel{N}(), spec, length.(range), range, f, x...; kw...)
end

@inline function parallel_reduce(; range::TR, f, args::Tuple,
        type = nothing, op = +, init = nothing, kw...) where {TR}
    return parallel_reduce(
        f, launch_spec(; kw...), range, args...; type = type,
        op = op, init = init)
end

array_size(a::AbstractArray) = size(a)
array_size(a::AbstractVector) = length(a)

elem_access(i, a::AbstractVector) = a[i]
elem_access(i, j, a::AbstractMatrix) = a[i, j]
elem_access(i, j, k, a::AbstractArray{T, 3}) where {T} = a[i, j, k]
elem_access(args...) = args[end][args[1:(end - 1)]...]

@inline function parallel_reduce(
        op, a::AbstractArray; init = default_init(eltype(a), op))
    return parallel_reduce(
        elem_access, array_size(a), a; op = op, init = init)
end

@inline parallel_reduce(a::AbstractArray; kw...) = parallel_reduce(+, a, kw...)

# TODO: range version:
# - `parallel_reduce(<r>, a)`
# - `parallel_reduce(<r>, op, a)`

@inline function parallel_reduce(spec::LaunchSpec, op, a::AbstractArray;
        init = default_init(eltype(a), op))
    return parallel_reduce(
        elem_access, spec, array_size(a), a; op = op, init = init)
end

@inline function parallel_reduce(spec::LaunchSpec, a::AbstractArray)
    return parallel_reduce(spec, +, a)
end

include("threads/threads.jl")

include("macro.jl")

end # module JACC
