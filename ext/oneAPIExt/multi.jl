module Multi

import Base: Callable
using JACC, oneAPI

@inline ndevices() = length(oneAPI.devices())

JACC.Multi.ndev(::oneAPIBackend) = ndevices()

struct ArrayPart{T, N}
    a::oneAPI.oneDeviceArray{T, N, oneAPI.AS.Global}
    dev_id::Int
    ndev::Int
    ghost_dims::Int
end

@inline Base.size(p::ArrayPart) = size(p.a)
@inline Base.length(p::ArrayPart) = length(p.a)
@inline Base.getindex(p::ArrayPart, i) = getindex(p.a, i)
@inline Base.getindex(p::ArrayPart, i, j) = getindex(p.a, i, j)
@inline Base.setindex!(p::ArrayPart, v, i) = setindex!(p.a, v, i)
@inline Base.setindex!(p::ArrayPart, v, i, j) = setindex!(p.a, v, i, j)
@inline device_id(p::ArrayPart) = p.dev_id
@inline ghost_dims(p::ArrayPart) = p.ghost_dims

@inline JACC.Multi.device_id(::oneAPIBackend, p::ArrayPart) = p.dev_id

struct MultiArray{T, N, NG}
    a1::Vector{ArrayPart{T, N}}
    a2::Vector{oneArray{T, N}}
    orig_size::Any
end

JACC.to_host(x::MultiArray) = convert(Base.Array, x)

@inline ghost_dims(x::MultiArray{T, N, NG}) where {T, N, NG} = NG
@inline JACC.Multi.part_length(::oneAPIBackend, x::MultiArray) = size(x.a2[1])[end]

@inline process_param(x, dev_id) = x
@inline process_param(x::MultiArray, dev_id) = x.a1[dev_id]

JACC.Multi.multi_array_type(::oneAPIBackend) = MultiArray

# FIXME:
#   - what about ghost elements
function Base.convert(::Type{Base.Array}, x::MultiArray{T, 1}) where {T}
    oneAPI.device!(1)
    ndev = ndevices()
    ret = Base.Array{T, 1}(undef, x.orig_size)
    partlen = cld(x.orig_size, ndev)
    lastlen = x.orig_size - ((ndev - 1) * partlen)
    for i in 1:ndev
        oneAPI.device!(i)
        if i == ndev
            copyto!(ret, (((i - 1) * partlen) + 1), x.a2[i], 1, lastlen)
        else
            copyto!(ret, (((i - 1) * partlen) + 1), x.a2[i], 1, partlen)
        end
    end
    oneAPI.device!(1)
    return ret
end

function Base.convert(::Type{Base.Array}, x::MultiArray{T, 2}) where {T}
    oneAPI.device!(1)
    ndev = ndevices()
    ret = Base.Array{T, 2}(undef, x.orig_size)
    partlen = cld(x.orig_size[2], ndev)
    lastlen = x.orig_size[2] - ((ndev - 1) * partlen)
    for i in 1:ndev
        oneAPI.device!(i)
        if i == ndev
            copyto!(
                ret,
                CartesianIndices(
                    (1:size(x.a2[i], 1),
                    (((i - 1) * partlen) + 1):(i * lastlen))
                ),
                x.a2[i],
                CartesianIndices((1:size(x.a2[i], 1), 1:lastlen))
            )
        else
            copyto!(
                ret,
                CartesianIndices(
                    (1:size(x.a2[i], 1),
                    (((i - 1) * partlen) + 1):(i * partlen))
                ),
                x.a2[i],
                CartesianIndices(x.a2[i])
            )
        end
    end
    oneAPI.device!(1)
    return ret
end

function make_multi_array(x::Base.Vector{T}) where {T}
    ndev = ndevices()
    oneAPI.device!(1)
    total_length = length(x)
    partlen = cld(total_length, ndev)
    parts = Vector{oneVector{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 1}}(undef, ndev)

    for i in 1:ndev
        oneAPI.device!(i)
        parts[i] = oneArray(x[(((i - 1) * partlen) + 1):(i * partlen)])
        devparts[i] = ArrayPart(kernel_convert(parts[i]), i, ndev, 0)
    end

    oneAPI.device!(1)
    return MultiArray{T, 1, 0}(devparts, parts, total_length)
end

function make_multi_array(x::Base.Vector{T}, ghost_dims) where {T}
    ndev = ndevices()
    oneAPI.device!(1)
    total_length = length(x)
    partlen = cld(total_length, ndev)
    parts = Vector{oneVector{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 1}}(undef, ndev)
    ng = ghost_dims

    for i in 1:ndev
        oneAPI.device!(i)
        if i == 1
            parts[i] = oneArray(x[((i - 1) * partlen + 1):(i * partlen + ng)])
        elseif i == ndev
            parts[i] = oneArray(x[((i - 1) * partlen + 1 - ng):(i * partlen)])
        else
            parts[i] = oneArray(x[((i - 1) * partlen + 1 - ng):(i * partlen + ng)])
        end
        devparts[i] = ArrayPart(kernel_convert(parts[i]), i, ndev, ng)
    end

    oneAPI.device!(1)
    return MultiArray{T, 1, ng}(devparts, parts, total_length)
end

function make_multi_array(x::Base.Matrix{T}) where {T}
    ndev = ndevices()
    oneAPI.device!(1)
    total_length = size(x, 2)
    partlen = cld(total_length, ndev)
    parts = Vector{oneMatrix{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 2}}(undef, ndev)

    for i in 1:ndev
        oneAPI.device!(i)
        parts[i] = oneArray(x[:, (((i - 1) * partlen) + 1):(i * partlen)])
        devparts[i] = ArrayPart(kernel_convert(parts[i]), i, ndev, 0)
    end

    oneAPI.device!(1)
    return MultiArray{T, 2, 0}(devparts, parts, size(x))
end

function make_multi_array(x::Base.Matrix{T}, ghost_dims) where {T}
    ndev = ndevices()
    oneAPI.device!(1)
    total_length = size(x, 2)
    partlen = cld(total_length, ndev)
    parts = Vector{oneMatrix{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 2}}(undef, ndev)
    ng = ghost_dims

    for i in 1:ndev
        oneAPI.device!(i)
        if i == 1
            parts[i] = oneArray(x[:, ((i - 1) * partlen + 1):(i * partlen + ng)])
        elseif i == ndev
            parts[i] = oneArray(x[:, ((i - 1) * partlen + 1 - ng):(i * partlen)])
        else
            parts[i] = oneArray(x[:, ((i - 1) * partlen + 1 - ng):(i * partlen + ng)])
        end
        devparts[i] = ArrayPart(kernel_convert(parts[i]), i, ndev, ng)
    end

    oneAPI.device!(1)
    return MultiArray{T, 2, ng}(devparts, array_ret, size(x))
end

function JACC.Multi.array(::oneAPIBackend, x::Base.Array; ghost_dims)
    if ghost_dims == 0 || ndevices() == 1
        return make_multi_array(x)
    else
        return make_multi_array(x, ghost_dims)
    end
end

function JACC.Multi.ghost_shift(::oneAPIBackend, i::Integer, arr::ArrayPart)
    dev_id = device_id(arr)
    if dev_id == 1
        ind = i
    elseif dev_id == arr.ndev
        ind = i + ghost_dims(arr)
    else
        ind = i + ghost_dims(arr)
    end
    return ind
end

function JACC.Multi.ghost_shift(
        ::oneAPIBackend, (i, j)::NTuple{2, Integer}, arr::ArrayPart)
    ind = (0, 0)
    dev_id = device_id(arr)
    if dev_id == 1
        ind = (i, j)
    elseif dev_id == arr.ndev
        ind = (i, j + ghost_dims(arr))
    else
        ind = (i, j + ghost_dims(arr))
    end
    return ind
end

function JACC.Multi.sync_ghost_elems!(::oneAPIBackend, arr::MultiArray{
        T, 1}) where {T}
    oneAPI.device!(1)
    ndev = ndevices()
    ng = ghost_dims(arr)
    if ng == 0
        return
    end

    #Left to right swapping
    for i in 1:(ndev - 1)
        oneAPI.device!(i)
        tmp = Base.Array(arr.a2[i])
        size = length(tmp)
        oneAPI.device!(i + 1)
        ghost_lr = oneArray(tmp[(size + 1 - 2 * ng):(size - ng)])
        @oneapi items=32 groups=1 _multi_swap_ghost_lr(arr.a1[i + 1], ghost_lr)
    end

    #Right to left swapping
    for i in 2:ndev
        oneAPI.device!(i)
        tmp = Base.Array(arr.a2[i])
        size = length(tmp)
        oneAPI.device!(i - 1)
        ghost_rl = oneArray(tmp[(1 + ng):(2 * ng)])
        @oneapi items=32 groups=1 _multi_swap_ghost_rl(arr.a1[i - 1], ghost_rl)
    end

    for i in 1:ndev
        oneAPI.device!(i)
        oneAPI.synchronize()
    end

    oneAPI.device!(1)
end

function JACC.Multi.sync_ghost_elems!(::oneAPIBackend, arr::MultiArray{
        T, 2}) where {T}
    oneAPI.device!(1)
    ndev = ndevices()
    ng = ghost_dims(arr)
    if ng == 0
        return
    end

    #Left to right swapping
    for i in 1:(ndev - 1)
        oneAPI.device!(i)
        dim = size(arr.a2[i])
        tmp = Base.Array(arr.a2[i][:, (dim[2] + 1 - 2 * ng):(dim[2] - ng)])
        oneAPI.device!(i + 1)
        ghost_lr = oneArray(tmp)
        numThreads = 512
        threads = min(dim[1], numThreads)
        blocks = cld(dim[1], threads)
        @oneapi items=threads groups=blocks _multi_swap_2d_ghost_lr(
            arr.a1[i + 1], ghost_lr)
    end

    #Right to left swapping
    for i in 2:ndev
        oneAPI.device!(i)
        tmp = Base.Array(arr.a2[i][:, (1 + ng):(2 * ng)])
        oneAPI.device!(i - 1)
        dim = size(arr.a2[i - 1])
        ghost_rl = oneArray(tmp)
        numThreads = 512
        threads = min(dim[1], numThreads)
        blocks = cld(dim[1], threads)
        @oneapi items=threads groups=blocks _multi_swap_2d_ghost_rl(
            arr.a1[i - 1], ghost_rl)
    end

    for i in 1:ndev
        oneAPI.device!(i)
        oneAPI.synchronize()
    end

    oneAPI.device!(1)
end

function JACC.Multi.copy!(::oneAPIBackend, x::MultiArray, y::MultiArray)
    oneAPI.device!(1)
    ndev = ndevices()

    if ndims(x.a2[1]) == 1
        numThreads = 512
        if ghost_dims(x) == 0 && ghost_dims(y) != 0
            # "gcopytoarray"
            for i in 1:ndev
                oneAPI.device!(i)
                size = length(y.a2[i])
                threads = min(size, numThreads)
                blocks = cld(size, threads)
                @oneapi items=threads groups=blocks _multi_copy_ghosttoarray(
                    x.a1[i], y.a1[i], ndev)
            end
        elseif ghost_dims(x) != 0 && ghost_dims(y) == 0
            # "copytogarray"
            for i in 1:ndev
                oneAPI.device!(i)
                size = length(x.a2[i])
                threads = min(size, numThreads)
                blocks = cld(size, threads)
                @oneapi items=threads groups=blocks _multi_copy_arraytoghost(
                    x.a1[i], y.a1[i], ndev)
            end
        else
            for i in 1:ndev
                oneAPI.device!(i)
                size = length(x.a2[i])
                threads = min(size, numThreads)
                blocks = cld(size, threads)
                @oneapi items=threads groups=blocks _multi_copy(x.a1[i], y.a1[i])
            end
        end

        for i in 1:ndev
            oneAPI.device!(i)
            oneAPI.synchronize()
        end

    elseif ndims(x.a2[1]) == 2

        # TODO: handle arrays with ghost elements

        for i in 1:ndev
            oneAPI.device!(i)
            ssize = size(x.a2[i])
            numThreads = 16
            Mthreads = min(ssize[1], numThreads)
            Mblocks = cld(ssize[1], Mthreads)
            Nthreads = min(ssize[2], numThreads)
            Nblocks = cld(ssize[2], Nthreads)
            threads = (Mthreads, Nthreads)
            blocks = (Mblocks, Nblocks)
            @oneapi items=threads groups=blocks _multi_copy_2d(x.a1[i], y.a1[i])
        end

        for i in 1:ndev
            oneAPI.device!(i)
            oneAPI.synchronize()
        end
    end

    oneAPI.device!(1)
end

function JACC.Multi.parallel_for(::oneAPIBackend, N::Integer, f::Callable, x...)
    oneAPI.device!(1)
    ndev = ndevices()
    N_multi = cld(N, ndev)
    # numThreads = 256
    # threads = min(N_multi, numThreads)
    # blocks = cld(N_multi, threads)

    for i in 1:ndev
        oneAPI.device!(i)
        dev_id = i
        JACC.parallel_for(
            JACC.LaunchSpec{oneAPIBackend}(; sync = false), N_multi,
            f, process_param.((x), dev_id)...)
        # @oneapi items=threads groups=blocks _multi_parallel_for_amdgpu(
        #     N_multi, f, process_param.((x), dev_id)...)
    end

    for i in 1:ndev
        oneAPI.device!(i)
        oneAPI.synchronize()
    end

    oneAPI.device!(1)
end

function JACC.Multi.parallel_for(
        ::oneAPIBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = ndevices()
    N_multi = ceil(Int, N / ndev)

    for i in 1:ndev
        oneAPI.device!(i)
        dev_id = i
        JACC.parallel_for(JACC.LaunchSpec{oneAPIBackend}(; sync = false),
            (M, N_multi), f, process_param.((x), dev_id)...)
    end

    for i in 1:ndev
        oneAPI.device!(i)
        oneAPI.synchronize()
    end

    oneAPI.device!(1)
end

function JACC.Multi.parallel_reduce(
        ::oneAPIBackend, N::Integer, f::Callable, x...)
    oneAPI.device!(1)
    ndev = length(oneAPI.devices())
    rret = Vector{Any}(undef, ndev)
    N_multi = cld(N, ndev)

    for i in 1:ndev
        oneAPI.device!(i)
        dev_id = i
        reducer = JACC.reducer(;
            backend = oneAPIBackend(), type = JACC.default_float(),
            range = N_multi, op = +, sync = false)
        reducer(f, process_param.((x), dev_id)...)
        rret[i] = reducer.workspace.ret
    end

    for i in 1:ndev
        oneAPI.device!(i)
        oneAPI.synchronize()
    end

    final_rret = 0.0
    for i in 1:ndev
        oneAPI.device!(i)
        final_rret += Base.Array(rret[i])[]
    end

    oneAPI.device!(1)

    return final_rret
end

function JACC.Multi.parallel_reduce(
        ::oneAPIBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = ndevices()
    rret = Vector{Any}(undef, ndev)
    N_multi = cld(N, ndev)
    dims_multi = (M, N_multi)

    for i in 1:ndev
        oneAPI.device!(i)
        dev_id = i
        reducer = JACC.reducer(;
            backend = oneAPIBackend(), type = JACC.default_float(),
            range = dims_multi, op = +, sync = false)
        reducer(f, process_param.((x), dev_id)...)
        rret[i] = reducer.workspace.ret
    end

    for i in 1:ndev
        oneAPI.device!(i)
        oneAPI.synchronize()
    end

    final_rret = 0.0
    for i in 1:ndev
        oneAPI.device!(i)
        final_rret += Base.Array(rret[i])[]
    end

    oneAPI.device!(1)

    return final_rret
end

function _multi_copy(x, y)
    i = get_global_id()
    if i <= length(x)
        @inbounds x[i] = y[i]
    end
    return nothing
end

function _multi_copy_2d(x, y)
    i = get_global_id(1)
    j = get_global_id(2)
    dims = size(x)
    if i <= dims[1] && j <= dims[2]
        @inbounds x[i, j] = y[i, j]
    end
    return nothing
end

function _multi_copy_ghosttoarray(x::ArrayPart, y::ArrayPart, ndev::Integer)
    #x is the array and y is the ghost array
    i = get_global_id()
    dev_id = device_id(x)
    len = length(y)
    if dev_id == 1 && i < len
        @inbounds x[i] = y[i]
    elseif dev_id == ndev && i > 1
        @inbounds x[i - 1] = y[i]
    elseif i > 1 && i < len
        @inbounds x[i - 1] = y[i]
    end
    return nothing
end

function _multi_copy_arraytoghost(x::ArrayPart, y::ArrayPart, ndev::Integer)
    #x is the ghost array and y is the array
    i = get_global_id()
    dev_id = device_id(x)
    len = length(x)
    if dev_id == 1 && i < len
        @inbounds x[i] = y[i]
    elseif dev_id == ndev && i < len
        @inbounds x[i + 1] = y[i]
    elseif i > 1 && i < len
        @inbounds x[i] = y[i]
    end
    return nothing
end

function _multi_swap_ghost_lr(arr::ArrayPart, ghost)
    i = get_global_id()
    ng = ghost_dims(arr)
    if i <= ng
        arr[i] = ghost[i]
    end
    return nothing
end

function _multi_swap_ghost_rl(arr::ArrayPart, ghost)
    i = get_global_id()
    ng = ghost_dims(arr)
    len = length(arr)
    if i <= ng
        arr[len - ng + i] = ghost[i]
    end
    return nothing
end

function _multi_swap_2d_ghost_lr(arr::ArrayPart, ghost)
    i = get_global_id()
    if i < size(arr, 1) + 1
        ng = ghost_dims(arr)
        for n in 1:ng
            arr[i, n] = ghost[i, n]
        end
    end
    return nothing
end

function _multi_swap_2d_ghost_rl(arr::ArrayPart, ghost)
    i = get_global_id()
    dim = size(arr)
    if i < dim[1] + 1
        ng = ghost_dims(arr)
        for n in 1:ng
            arr[i, dim[2] - ng + n] = ghost[i, n]
        end
    end
    return nothing
end

end # module Multi
