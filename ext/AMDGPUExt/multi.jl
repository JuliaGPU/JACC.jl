module Multi

import Base: Callable
using JACC, AMDGPU
using AMDGPUExt: AMDGPUBackend

@inline ndevices() = length(AMDGPU.devices())

function JACC.Multi.ndev(::AMDGPUBackend)
    return ndevices()
end

struct ArrayPart{T, N}
    a::ROCDeviceArray{T, N, AMDGPU.Device.AS.Global}
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

@inline JACC.Multi.device_id(::AMDGPUBackend, p::ArrayPart) = p.dev_id

struct MultiArray{T, N, NG}
    a1::Vector{ArrayPart{T, N}}
    a2::Vector{ROCArray{T, N}}
    orig_size::Any
end

JACC.to_host(x::MultiArray) = convert(Base.Array, x)

@inline ghost_dims(x::MultiArray{T, N, NG}) where {T, N, NG} = NG
@inline JACC.Multi.part_length(::AMDGPUBackend, x::MultiArray) = size(x.a2[1])[end]

@inline process_param(x, dev_id) = x
@inline process_param(x::MultiArray, dev_id) = x.a1[dev_id]

JACC.Multi.multi_array_type(::AMDGPUBackend) = MultiArray

# FIXME:
#   - what about ghost elements
function Base.convert(::Type{Base.Array}, x::MultiArray{T, 1}) where {T}
    AMDGPU.device_id!(1)
    ndev = ndevices()
    ret = Base.Array{T, 1}(undef, x.orig_size)
    partlen = cld(x.orig_size, ndev)
    lastlen = x.orig_size - ((ndev - 1) * partlen)
    for i in 1:ndev
        AMDGPU.device_id!(i)
        if i == ndev
            copyto!(ret, (((i - 1) * partlen) + 1), x.a2[i], 1, lastlen)
        else
            copyto!(ret, (((i - 1) * partlen) + 1), x.a2[i], 1, partlen)
        end
    end
    AMDGPU.device_id!(1)
    return ret
end

function Base.convert(::Type{Base.Array}, x::MultiArray{T, 2}) where {T}
    AMDGPU.device_id!(1)
    ndev = ndevices()
    ret = Base.Array{T, 2}(undef, x.orig_size)
    partlen = cld(x.orig_size[2], ndev)
    lastlen = x.orig_size[2] - ((ndev - 1) * partlen)
    for i in 1:ndev
        AMDGPU.device_id!(i)
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
    AMDGPU.device_id!(1)
    return ret
end

function make_multi_array(x::Base.Vector{T}) where {T}
    ndev = ndevices()
    AMDGPU.device_id!(1)
    total_length = length(x)
    partlen = cld(total_length, ndev)
    parts = Vector{ROCVector{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 1}}(undef, ndev)

    for i in 1:ndev
        AMDGPU.device_id!(i)
        parts[i] = ROCArray(x[(((i - 1) * partlen) + 1):(i * partlen)])
        devparts[i] = ArrayPart(rocconvert(parts[i]), i, ndev, 0)
    end

    AMDGPU.device_id!(1)
    return MultiArray{T, 1, 0}(devparts, parts, total_length)
end

function make_multi_array(x::Base.Vector{T}, ghost_dims) where {T}
    ndev = ndevices()
    AMDGPU.device_id!(1)
    total_length = length(x)
    partlen = cld(total_length, ndev)
    parts = Vector{ROCVector{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 1}}(undef, ndev)
    ng = ghost_dims

    for i in 1:ndev
        AMDGPU.device_id!(i)
        if i == 1
            parts[i] = ROCArray(x[((i - 1) * partlen + 1):(i * partlen + ng)])
        elseif i == ndev
            parts[i] = ROCArray(x[((i - 1) * partlen + 1 - ng):(i * partlen)])
        else
            parts[i] = ROCArray(x[((i - 1) * partlen + 1 - ng):(i * partlen + ng)])
        end
        devparts[i] = ArrayPart(rocconvert(parts[i]), i, ndev, ng)
    end

    AMDGPU.device_id!(1)
    return MultiArray{T, 1, ng}(devparts, parts, total_length)
end

function make_multi_array(x::Base.Matrix{T}) where {T}
    ndev = ndevices()
    AMDGPU.device_id!(1)
    total_length = size(x, 2)
    partlen = cld(total_length, ndev)
    parts = Vector{ROCMatrix{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 2}}(undef, ndev)

    for i in 1:ndev
        AMDGPU.device_id!(i)
        parts[i] = ROCArray(x[:, (((i - 1) * partlen) + 1):(i * partlen)])
        devparts[i] = ArrayPart(rocconvert(parts[i]), i, ndev, 0)
    end

    AMDGPU.device_id!(1)
    return MultiArray{T, 2, 0}(devparts, parts, size(x))
end

function make_multi_array(x::Base.Matrix{T}, ghost_dims) where {T}
    ndev = ndevices()
    AMDGPU.device_id!(1)
    total_length = size(x, 2)
    partlen = cld(total_length, ndev)
    parts = Vector{ROCMatrix{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 2}}(undef, ndev)
    ng = ghost_dims

    for i in 1:ndev
        AMDGPU.device_id!(i)
        if i == 1
            parts[i] = ROCArray(x[
                :, ((i - 1) * partlen + 1):(i * partlen + ng)])
        elseif i == ndev
            parts[i] = ROCArray(x[
                :, ((i - 1) * partlen + 1 - ng):(i * partlen)])
        else
            parts[i] = ROCArray(x[
                :, ((i - 1) * partlen + 1 - ng):(i * partlen + ng)])
        end
        devparts[i] = ArrayPart(rocconvert(parts[i]), i, ndev, ng)
    end

    AMDGPU.device_id!(1)
    return MultiArray{T, 2, ng}(devparts, array_ret, size(x))
end

function JACC.Multi.array(::AMDGPUBackend, x::Base.Array; ghost_dims)
    if ghost_dims == 0 || ndevices() == 1
        return make_multi_array(x)
    else
        return make_multi_array(x, ghost_dims)
    end
end

function JACC.Multi.ghost_shift(::AMDGPUBackend, i::Integer, arr::ArrayPart)
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
        ::AMDGPUBackend, (i, j)::NTuple{2, Integer}, arr::ArrayPart)
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

function JACC.Multi.sync_ghost_elems!(
        ::AMDGPUBackend, arr::MultiArray{T, 1}) where {T}
    AMDGPU.device_id!(1)
    ndev = ndevices()
    ng = ghost_dims(arr)
    if ng == 0
        return
    end

    #Left to right swapping
    for i in 1:(ndev - 1)
        AMDGPU.device_id!(i)
        tmp = Base.Array(arr.a2[i])
        size = length(tmp)
        AMDGPU.device_id!(i + 1)
        ghost_lr = ROCArray(tmp[(size + 1 - 2 * ng):(size - ng)])
        @roc groupsize=32 gridsize=1 _multi_swap_ghost_lr(
            arr.a1[i + 1], ghost_lr)
    end

    #Right to left swapping
    for i in 2:ndev
        AMDGPU.device_id!(i)
        tmp = Base.Array(arr.a2[i])
        size = length(tmp)
        AMDGPU.device_id!(i - 1)
        ghost_rl = ROCArray(tmp[(1 + ng):(2 * ng)])
        @roc groupsize=32 gridsize=1 _multi_swap_ghost_rl(
            arr.a1[i - 1], ghost_rl)
    end

    for i in 1:ndev
        AMDGPU.device_id!(i)
        AMDGPU.synchronize()
    end

    AMDGPU.device_id!(1)
end

function JACC.Multi.sync_ghost_elems!(
        ::AMDGPUBackend, arr::MultiArray{T, 2}) where {T}
    AMDGPU.device_id!(1)
    ndev = ndevices()
    ng = ghost_dims(arr)
    if ng == 0
        return
    end

    #Left to right swapping
    for i in 1:(ndev - 1)
        AMDGPU.device_id!(i)
        dim = size(arr.a2[i])
        tmp = Base.Array(arr.a2[i][:, (dim[2] + 1 - 2 * ng):(dim[2] - ng)])
        AMDGPU.device_id!(i + 1)
        ghost_lr = ROCArray(tmp)
        numThreads = 512
        threads = min(dim[1], numThreads)
        blocks = cld(dim[1], threads)
        @roc groupsize=threads gridsize=blocks _multi_swap_2d_ghost_lr(
            arr.a1[i + 1], ghost_lr)
    end

    #Right to left swapping
    for i in 2:ndev
        AMDGPU.device_id!(i)
        tmp = Base.Array(arr.a2[i][:, (1 + ng):(2 * ng)])
        AMDGPU.device_id!(i - 1)
        dim = size(arr.a2[i - 1])
        ghost_rl = ROCArray(tmp)
        numThreads = 512
        threads = min(dim[1], numThreads)
        blocks = cld(dim[1], threads)
        @roc groupsize=threads gridsize=blocks _multi_swap_2d_ghost_rl(
            arr.a1[i - 1], ghost_rl)
    end

    for i in 1:ndev
        AMDGPU.device_id!(i)
        AMDGPU.synchronize()
    end

    AMDGPU.device_id!(1)
end

function JACC.Multi.copy!(::AMDGPUBackend, x::MultiArray, y::MultiArray)
    AMDGPU.device_id!(1)
    ndev = ndevices()

    if ndims(x.a2[1]) == 1
        numThreads = 512
        if ghost_dims(x) == 0 && ghost_dims(y) != 0
            # "gcopytoarray"
            for i in 1:ndev
                AMDGPU.device_id!(i)
                size = length(y.a2[i])
                threads = min(size, numThreads)
                blocks = cld(size, threads)
                @roc groupsize=threads gridsize=blocks _multi_copy_ghosttoarray(
                    x.a1[i], y.a1[i], ndev)
            end
        elseif ghost_dims(x) != 0 && ghost_dims(y) == 0
            # "copytogarray"
            for i in 1:ndev
                AMDGPU.device_id!(i)
                size = length(x.a2[i])
                threads = min(size, numThreads)
                blocks = cld(size, threads)
                @roc groupsize=threads gridsize=blocks _multi_copy_arraytoghost(
                    x.a1[i], y.a1[i], ndev)
            end
        else
            for i in 1:ndev
                AMDGPU.device_id!(i)
                size = length(x.a2[i])
                threads = min(size, numThreads)
                blocks = cld(size, threads)
                @roc groupsize=threads gridsize=blocks _multi_copy(
                    x.a1[i], y.a1[i])
            end
        end

        for i in 1:ndev
            AMDGPU.device_id!(i)
            AMDGPU.synchronize()
        end

    elseif ndims(x.a2[1]) == 2

        # TODO: handle arrays with ghost elements

        for i in 1:ndev
            AMDGPU.device_id!(i)
            ssize = size(x.a2[i])
            numThreads = 16
            Mthreads = min(ssize[1], numThreads)
            Mblocks = cld(ssize[1], Mthreads)
            Nthreads = min(ssize[2], numThreads)
            Nblocks = cld(ssize[2], Nthreads)
            threads = (Mthreads, Nthreads)
            blocks = (Mblocks, Nblocks)
            @roc groupsize=threads gridsize=blocks _multi_copy_2d(
                x.a1[i], y.a1[i])
        end

        for i in 1:ndev
            AMDGPU.device_id!(i)
            AMDGPU.synchronize()
        end
    end

    AMDGPU.device_id!(1)
end

function JACC.Multi.parallel_for(::AMDGPUBackend, N::Integer, f::Callable, x...)
    AMDGPU.device_id!(1)
    ndev = ndevices()
    N_multi = cld(N, ndev)
    numThreads = 256
    threads = min(N_multi, numThreads)
    blocks = cld(N_multi, threads)

    for i in 1:ndev
        AMDGPU.device_id!(i)
        dev_id = i
        @roc groupsize=threads gridsize=blocks _multi_parallel_for_amdgpu(
            N_multi, f, process_param.((x), dev_id)...)
    end

    for i in 1:ndev
        AMDGPU.device_id!(i)
        AMDGPU.synchronize()
    end

    AMDGPU.device_id!(1)
end

function JACC.Multi.parallel_for(
        ::AMDGPUBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = ndevices()
    N_multi = ceil(Int, N / ndev)
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N_multi, numThreads)
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N_multi, Nthreads)
    blocks = (Mblocks, Nblocks)

    for i in 1:ndev
        AMDGPU.device_id!(i)
        dev_id = i
        @roc groupsize=threads gridsize=blocks _multi_parallel_for_amdgpu_MN(
            M, N_multi, f, process_param.((x), dev_id)...)
    end

    for i in 1:ndev
        AMDGPU.device_id!(i)
        AMDGPU.synchronize()
    end

    AMDGPU.device_id!(1)
end

function JACC.Multi.parallel_reduce(
        ::AMDGPUBackend, N::Integer, f::Callable, x...)
    AMDGPU.device_id!(1)
    ndev = length(AMDGPU.devices())
    ret = Vector{Any}(undef, ndev)
    rret = Vector{Any}(undef, ndev)
    N_multi = cld(N, ndev)
    numThreads = 512
    threads = min(N_multi, numThreads)
    blocks = cld(N_multi, threads)
    final_rret = AMDGPU.zeros(Float64, 1)

    for i in 1:ndev
        AMDGPU.device_id!(i)
        ret[i] = AMDGPU.zeros(Float64, blocks)
        rret[i] = AMDGPU.zeros(Float64, 1)
    end

    for i in 1:ndev
        AMDGPU.device_id!(i)
        dev_id = i
        @roc groupsize=threads gridsize=blocks _multi_parallel_reduce_amdgpu(
            N_multi, ret[i], f, process_param.((x), dev_id)...)
        @roc groupsize=threads gridsize=1 _multi_reduce_kernel_amdgpu(
            blocks, ret[i], rret[i])
    end

    for i in 1:ndev
        AMDGPU.device_id!(i)
        AMDGPU.synchronize()
    end

    final_rret = 0.0
    for i in 1:ndev
        AMDGPU.device_id!(i)
        final_rret += Base.Array(rret[i])[]
    end

    # for i in 1:ndev
    #     tmp_final_rret += tmp_rret[i][1]
    # end
    # final_rret = tmp_final_rret

    AMDGPU.device_id!(1)

    return final_rret
end

function JACC.Multi.parallel_reduce(::AMDGPUBackend,
        (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = ndevices()
    ret = Vector{Any}(undef, ndev)
    rret = Vector{Any}(undef, ndev)
    N_multi = cld(N, ndev)
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N_multi, numThreads)
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N_multi, Nthreads)
    blocks = (Mblocks, Nblocks)
    final_rret = AMDGPU.zeros(Float64, 1)

    for i in 1:ndev
        AMDGPU.device_id!(i)
        ret[i] = AMDGPU.zeros(Float64, (Mblocks, Nblocks))
        rret[i] = AMDGPU.zeros(Float64, 1)
    end

    for i in 1:ndev
        AMDGPU.device_id!(i)
        dev_id = i

        @roc groupsize=threads gridsize=blocks _multi_parallel_reduce_amdgpu_MN(
            (M, N_multi), ret[i], f, process_param.((x), dev_id)...)

        @roc groupsize=threads gridsize=(1, 1) _multi_reduce_kernel_amdgpu_MN(
            blocks, ret[i], rret[i])
    end

    for i in 1:ndev
        AMDGPU.device_id!(i)
        AMDGPU.synchronize()
    end

    final_rret = 0.0
    for i in 1:ndev
        AMDGPU.device_id!(i)
        final_rret += Base.Array(rret[i])[]
    end

    AMDGPU.device_id!(1)

    return final_rret
end

# COV_EXCL_START
function _multi_copy(x, y)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i <= length(x)
        @inbounds x[i] = y[i]
    end
    return nothing
end

function _multi_copy_2d(x, y)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    dims = size(x)
    if i <= dims[1] && j <= dims[2]
        @inbounds x[i, j] = y[i, j]
    end
    return nothing
end

function _multi_copy_ghosttoarray(x::ArrayPart, y::ArrayPart, ndev::Integer)
    #x is the array and y is the ghost array
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
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
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
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
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ng = ghost_dims(arr)
    if i <= ng
        arr[i] = ghost[i]
    end
    return nothing
end

function _multi_swap_ghost_rl(arr::ArrayPart, ghost)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ng = ghost_dims(arr)
    len = length(arr)
    if i <= ng
        arr[len - ng + i] = ghost[i]
    end
    return nothing
end

function _multi_swap_2d_ghost_lr(arr::ArrayPart, ghost)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i < size(arr, 1) + 1
        ng = ghost_dims(arr)
        for n in 1:ng
            arr[i, n] = ghost[i, n]
        end
    end
    return nothing
end

function _multi_swap_2d_ghost_rl(arr::ArrayPart, ghost)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    dim = size(arr)
    if i < dim[1] + 1
        ng = ghost_dims(arr)
        for n in 1:ng
            arr[i, dim[2] - ng + n] = ghost[i, n]
        end
    end
    return nothing
end

function _multi_parallel_for_amdgpu(N, f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i <= N
        f(i, x...)
    end
    return nothing
end

function _multi_parallel_for_amdgpu_MN(M, N, f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    if (i <= M) && (j <= N)
        f(i, j, x...)
    end
    return nothing
end

function _multi_parallel_reduce_amdgpu(N, ret, f, x...)
    shared_mem = @ROCStaticLocalArray(Float64, 512)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ti = workitemIdx().x
    tmp::Float64 = 0.0
    shared_mem[ti] = 0.0

    if i <= N
        tmp = @inbounds f(i, x...)
        shared_mem[workitemIdx().x] = tmp
    end
    AMDGPU.sync_workgroup()
    if (ti <= 256)
        shared_mem[ti] += shared_mem[ti + 256]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 128)
        shared_mem[ti] += shared_mem[ti + 128]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 64)
        shared_mem[ti] += shared_mem[ti + 64]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 32)
        shared_mem[ti] += shared_mem[ti + 32]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 16)
        shared_mem[ti] += shared_mem[ti + 16]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 8)
        shared_mem[ti] += shared_mem[ti + 8]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 4)
        shared_mem[ti] += shared_mem[ti + 4]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 2)
        shared_mem[ti] += shared_mem[ti + 2]
    end
    AMDGPU.sync_workgroup()
    if (ti == 1)
        shared_mem[ti] += shared_mem[ti + 1]
        ret[workgroupIdx().x] = shared_mem[ti]
    end
    return nothing
end

function _multi_reduce_kernel_amdgpu(N, red, ret)
    shared_mem = @ROCStaticLocalArray(Float64, 512)
    i = workitemIdx().x
    ii = i
    tmp::Float64 = 0.0
    if N > 512
        while ii <= N
            tmp += @inbounds red[ii]
            ii += 512
        end
    elseif (i <= N)
        tmp = @inbounds red[i]
    end
    shared_mem[workitemIdx().x] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 256)
        shared_mem[i] += shared_mem[i + 256]
    end
    AMDGPU.sync_workgroup()
    if (i <= 128)
        shared_mem[i] += shared_mem[i + 128]
    end
    AMDGPU.sync_workgroup()
    if (i <= 64)
        shared_mem[i] += shared_mem[i + 64]
    end
    AMDGPU.sync_workgroup()
    if (i <= 32)
        shared_mem[i] += shared_mem[i + 32]
    end
    AMDGPU.sync_workgroup()
    if (i <= 16)
        shared_mem[i] += shared_mem[i + 16]
    end
    AMDGPU.sync_workgroup()
    if (i <= 8)
        shared_mem[i] += shared_mem[i + 8]
    end
    AMDGPU.sync_workgroup()
    if (i <= 4)
        shared_mem[i] += shared_mem[i + 4]
    end
    AMDGPU.sync_workgroup()
    if (i <= 2)
        shared_mem[i] += shared_mem[i + 2]
    end
    AMDGPU.sync_workgroup()
    if (i == 1)
        shared_mem[i] += shared_mem[i + 1]
        ret[1] = shared_mem[1]
    end
    AMDGPU.sync_workgroup()
    return nothing
end

function _multi_parallel_reduce_amdgpu_MN((M, N), ret, f, x...)
    shared_mem = @ROCStaticLocalArray(Float64, 16*16)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ti = workitemIdx().x
    tj = workitemIdx().y
    bi = workgroupIdx().x
    bj = workgroupIdx().y

    sid = ((ti - 1) * 16) + tj
    tmp::Float64 = 0.0
    shared_mem[sid] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds f(i, j, x...)
        shared_mem[sid] = tmp
    end
    AMDGPU.sync_workgroup()
    if (ti <= 8 && tj <= 8)
        shared_mem[sid] += shared_mem[((ti + 7) * 16) + (tj + 8)]
        shared_mem[sid] += shared_mem[((ti - 1) * 16) + (tj + 8)]
        shared_mem[sid] += shared_mem[((ti + 7) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 4 && tj <= 4)
        shared_mem[sid] += shared_mem[((ti + 3) * 16) + (tj + 4)]
        shared_mem[sid] += shared_mem[((ti - 1) * 16) + (tj + 4)]
        shared_mem[sid] += shared_mem[((ti + 3) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 2 && tj <= 2)
        shared_mem[sid] += shared_mem[((ti + 1) * 16) + (tj + 2)]
        shared_mem[sid] += shared_mem[((ti - 1) * 16) + (tj + 2)]
        shared_mem[sid] += shared_mem[((ti + 1) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti == 1 && tj == 1)
        shared_mem[sid] += shared_mem[ti * 16 + (tj + 1)]
        shared_mem[sid] += shared_mem[((ti - 1) * 16) + (tj + 1)]
        shared_mem[sid] += shared_mem[ti * 16 + tj]
        ret[bi, bj] = shared_mem[sid]
    end
    return nothing
end

function _multi_reduce_kernel_amdgpu_MN((M, N), red, ret)
    shared_mem = @ROCStaticLocalArray(Float64, 16*16)
    i = workitemIdx().x
    j = workitemIdx().y
    ii = i
    jj = j

    sid = ((i - 1) * 16) + j
    tmp::Float64 = 0.0
    shared_mem[sid] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = workitemIdx().y
            while jj <= N
                tmp = tmp + @inbounds red[ii, jj]
                jj += 16
            end
            ii += 16
        end
    elseif M > 16
        while ii <= N
            tmp = tmp + @inbounds red[ii, jj]
            ii += 16
        end
    elseif N > 16
        while jj <= N
            tmp = tmp + @inbounds red[ii, jj]
            jj += 16
        end
    elseif M <= 16 && N <= 16
        if i <= M && j <= N
            tmp = tmp + @inbounds red[i, j]
        end
    end
    shared_mem[sid] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 8 && j <= 8)
        if (i + 8 <= M && j + 8 <= N)
            shared_mem[sid] += shared_mem[((i + 7) * 16) + (j + 8)]
        end
        if (i <= M && j + 8 <= N)
            shared_mem[sid] += shared_mem[((i - 1) * 16) + (j + 8)]
        end
        if (i + 8 <= M && j <= N)
            shared_mem[sid] += shared_mem[((i + 7) * 16) + j]
        end
    end
    AMDGPU.sync_workgroup()
    if (i <= 4 && j <= 4)
        if (i + 4 <= M && j + 4 <= N)
            shared_mem[sid] += shared_mem[((i + 3) * 16) + (j + 4)]
        end
        if (i <= M && j + 4 <= N)
            shared_mem[sid] += shared_mem[((i - 1) * 16) + (j + 4)]
        end
        if (i + 4 <= M && j <= N)
            shared_mem[sid] += shared_mem[((i + 3) * 16) + j]
        end
    end
    AMDGPU.sync_workgroup()
    if (i <= 2 && j <= 2)
        if (i + 2 <= M && j + 2 <= N)
            shared_mem[sid] += shared_mem[((i + 1) * 16) + (j + 2)]
        end
        if (i <= M && j + 2 <= N)
            shared_mem[sid] += shared_mem[((i - 1) * 16) + (j + 2)]
        end
        if (i + 2 <= M && j <= N)
            shared_mem[sid] += shared_mem[((i + 1) * 16) + j]
        end
    end
    AMDGPU.sync_workgroup()
    if (i == 1 && j == 1)
        if (i + 1 <= M && j + 1 <= N)
            shared_mem[sid] += shared_mem[i * 16 + (j + 1)]
        end
        if (i <= M && j + 1 <= N)
            shared_mem[sid] += shared_mem[((i - 1) * 16) + (j + 1)]
        end
        if (i + 1 <= M && j <= N)
            shared_mem[sid] += shared_mem[i * 16 + j]
        end
        ret[1] = shared_mem[sid]
    end
    return nothing
end
# COV_EXCL_STOP

end # module Multi
