import JACC: AllDims, default_backend

@inline _resolve_init_type(op, init::Nothing) = JACC.default_init(op)
@inline _resolve_init_type(op, init) = init

@inline function parallel_scan(f, dims::AllDims, x...; op = +, init = nothing)
    _init = _resolve_init_type(op, init)
    return parallel_scan(f, default_backend(), dims, x...; op = op, init = _init)
end

@inline function parallel_scan(dims::AllDims, f, x...; kw...)
    return parallel_scan(f, dims, x...; kw...)
end

# parallel_scan(LaunchSpec)

@inline function _range_scan_kernel(i, partial, is_final, range, f, x...)
    @inbounds f(range[i], partial, is_final, x...)
end

@inline function parallel_scan(f, range::AbstractRange, x...; kw...)
    parallel_scan(_range_scan_kernel, length(range), range, f, x...; kw...)
end

# @inline function parallel_scan(f, range::NTuple{N, AbstractRange}, x...; kw...) where {N}
#     parallel_scan(_RangesKernel{N}(), length.(range), range, f, x...; kw...)
# end

# parallel_scan(LaunchSpec)

# @inline function parallel_scan(; range::TR, f, args::Tuple, type = nothing,
#         op = +, init = nothing, name = nothing, kw...) where {TR}
#     parallel_scan(f, launch_spec(; kw...), range, args...; type = type, op = op,
#                   init = init, name = name)
# end

struct ElemScanInclusive{Op}
    op::Op
end

@inline function (f::ElemScanInclusive)(i, partial, is_final, a)
    @inbounds partial[] = f.op(partial[], a[i])
    if is_final
        @inbounds a[i] = partial[]
    end
end

struct ElemScanExclusive{Op}
    op::Op
end

@inline function (f::ElemScanExclusive)(i, partial, is_final, a)
    if is_final
        @inbounds a[i] = partial[]
    end
    @inbounds partial[] = f.op(partial[], a[i])
end

@inline function parallel_scan(op, a::AbstractArray;
        init = default_init(eltype(a), op), inclusive = true, kw...)
    if inclusive
        parallel_scan(ElemScanInclusive(op), array_size(a), a; op = op,
            init = init, kw...)
    else
        parallel_scan(ElemScanExclusive(op), array_size(a), a; op = op,
            init = init, kw...)
    end
end

@inline parallel_scan(a::AbstractArray; kw...) = parallel_scan(+, a; kw...)
