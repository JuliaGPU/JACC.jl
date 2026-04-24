
array_type() = array_type(default_backend())

to_device(x::AbstractArray) = convert(array_type(), x)
to_host(x::AbstractArray) = convert(Base.Array, x)

"""
    transfer!(dst::AbstractArray{T, N}, src::AbstractArray{T, N}) where {T, N} -> dst

Transfer array `src` to array `dst`, discarding pre-existing elements in `dst`.

Throw an `ArgumentError` if arrays do not have equal axes.

!!! warning
    Behavior can be unexpected when any mutated argument shares memory with any other argument.

See also [`copyto!`](@ref).
"""
function transfer!(dst::AbstractArray{T, N}, src::AbstractArray{
        T, N}) where {T, N}
    axes(dst) == axes(src) || throw(ArgumentError(
        "arrays must have the same axes for `transfer!` (consider using `copyto!`)"))
    _transfer!(dst, IndexStyle(dst), src, IndexStyle(src))
    dst
end

function _transfer!(dst::AbstractArray{T}, ::IndexCartesian,
        src::AbstractArray{T}, ::IndexStyle) where {T}
    _cartesian_transfer!(dst, src)
end

function _transfer!(dst::AbstractArray{T}, ::IndexStyle,
        src::AbstractArray{T}, ::IndexCartesian) where {T}
    _cartesian_transfer!(dst, src)
end

function _transfer!(dst::AbstractArray{T}, ::IndexCartesian,
        src::AbstractArray{T}, ::IndexCartesian) where {T}
    _cartesian_transfer!(dst, src)
end

function _transfer!(dst::AbstractArray{T}, ::IndexLinear,
        src::AbstractArray{T}, ::IndexLinear) where {T}
    _linear_transfer!(dst, src)
end

function _cartesian_transfer!(dst::AbstractArray{T}, src::AbstractArray{T}) where {T}
    copyto!(parent(dst), CartesianIndices(parentindices(dst)),
        parent(src), CartesianIndices(parentindices(src)))
end

_prep_for_linear_copy(a::AbstractArray{T}, p::AbstractArray{T}) where {T} = a
function _prep_for_linear_copy(a::AbstractArray{T}, p::Base.Array{T}) where {T}
    unsafe_wrap(Base.Array, pointer(a), length(a))
end

function _linear_transfer!(dst::AbstractArray{T}, src::AbstractArray{T}) where {T}
    copyto!(_prep_for_linear_copy(dst, parent(dst)), _prep_for_linear_copy(src, parent(src)))
end

array(x::AbstractArray) = to_device(x)

array(::Type{T}, dims) where {T} = array_type(){T, length(dims)}(undef, dims)
array(::Type{T}, dims...) where {T} = array(T, dims)
array(dims) = array(default_float(), dims)
array(dims...) = array(dims)
array(; type = default_float(), dims = 0) = array(type, dims)

zeros(::Type{T}, dims...) where {T} = zeros(default_backend(), T, dims...)
ones(::Type{T}, dims...) where {T} = ones(default_backend(), T, dims...)

zeros(dims...) = zeros(default_float(), dims...)
ones(dims...) = ones(default_float(), dims...)

fill(value, dims...) = fill(default_backend(), value, dims...)
