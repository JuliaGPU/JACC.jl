module Multi

import Base: Callable
using JACC, CUDA

@inline ndevices() = length(devices())

function JACC.Multi.ndev(::CUDABackend)
    return ndevices()
end

struct ArrayPart{T, N}
    a::CuDeviceArray{T, N, CUDA.AS.Global}
    dev_id::Int
    ndev::Int32
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

@inline JACC.Multi.device_id(::CUDABackend, p::ArrayPart) = p.dev_id

struct MultiArray{T, N, NG}
    a1::Vector{ArrayPart{T, N}}
    a2::Vector{CuArray{T, N}}
    orig_size::Any
end

JACC.to_host(x::MultiArray) = convert(Base.Array, x)

@inline ghost_dims(x::MultiArray{T, N, NG}) where {T, N, NG} = NG
@inline JACC.Multi.part_length(::CUDABackend, x::MultiArray) = size(x.a2[1])[end]

@inline process_param(x, dev_id) = x
@inline process_param(x::MultiArray, dev_id) = x.a1[dev_id]

JACC.Multi.multi_array_type(::CUDABackend) = MultiArray

# FIXME:
#   - what about ghost elements
function Base.convert(::Type{Base.Array}, x::MultiArray{T, 1}) where {T}
    device!(0)
    ndev = ndevices()
    ret = Base.Array{T, 1}(undef, x.orig_size)
    partlen = cld(x.orig_size, ndev)
    lastlen = x.orig_size - ((ndev - 1) * partlen)
    for i in 1:ndev
        device!(i - 1)
        if i == ndev
            copyto!(ret, (((i - 1) * partlen) + 1), x.a2[i], 1, lastlen)
        else
            copyto!(ret, (((i - 1) * partlen) + 1), x.a2[i], 1, partlen)
        end
    end
    device!(0)
    return ret
end

function Base.convert(::Type{Base.Array}, x::MultiArray{T, 2}) where {T}
    device!(0)
    ndev = ndevices()
    ret = Base.Array{T, 2}(undef, x.orig_size)
    partlen = cld(x.orig_size[2], ndev)
    lastlen = x.orig_size[2] - ((ndev - 1) * partlen)
    for i in 1:ndev
        device!(i - 1)
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
    device!(0)
    return ret
end

function make_multi_array(x::Base.Vector{T}) where {T}
    ndev = ndevices()
    device!(0)
    total_length = length(x)
    partlen = cld(total_length, ndev)
    parts = Vector{CuVector{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 1}}(undef, ndev)

    for i in 1:ndev
        device!(i - 1)
        parts[i] = CuArray(x[(((i - 1) * partlen) + 1):(i * partlen)])
        devparts[i] = ArrayPart(cudaconvert(parts[i]), i, ndev, 0)
    end

    device!(0)
    return MultiArray{T, 1, 0}(devparts, parts, total_length)
end

function make_multi_array(x::Base.Vector{T}, ghost_dims) where {T}
    ndev = ndevices()
    device!(0)
    total_length = length(x)
    partlen = cld(total_length, ndev)
    parts = Vector{CuVector{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 1}}(undef, ndev)
    ng = ghost_dims

    for i in 1:ndev
        device!(i - 1)
        if i == 1
            parts[i] = CuArray(x[((i - 1) * partlen + 1):(i * partlen + ng)])
        elseif i == ndev
            parts[i] = CuArray(x[((i - 1) * partlen + 1 - ng):(i * partlen)])
        else
            parts[i] = CuArray(x[((i - 1) * partlen + 1 - ng):(i * partlen + ng)])
        end
        devparts[i] = ArrayPart(cudaconvert(parts[i]), i, ndev, ng)
    end

    device!(0)
    return MultiArray{T, 1, ng}(devparts, parts, total_length)
end

function make_multi_array(x::Base.Matrix{T}) where {T}
    ndev = ndevices()
    device!(0)
    total_length = size(x, 2)
    partlen = cld(total_length, ndev)
    parts = Vector{CuMatrix{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 2}}(undef, ndev)

    for i in 1:ndev
        device!(i - 1)
        parts[i] = CuArray(x[:, (((i - 1) * partlen) + 1):(i * partlen)])
        devparts[i] = ArrayPart(cudaconvert(parts[i]), i, ndev, 0)
    end

    device!(0)
    return MultiArray{T, 2, 0}(devparts, parts, size(x))
end

function make_multi_array(x::Base.Matrix{T}, ghost_dims) where {T}
    ndev = ndevices()
    device!(0)
    total_length = size(x, 2)
    partlen = cld(total_length, ndev)
    parts = Vector{CuMatrix{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 2}}(undef, ndev)
    ng = ghost_dims

    for i in 1:ndev
        device!(i - 1)
        if i == 1
            parts[i] = CuArray(x[:, ((i - 1) * partlen + 1):(i * partlen + ng)])
        elseif i == ndev
            parts[i] = CuArray(x[:, ((i - 1) * partlen + 1 - ng):(i * partlen)])
        else
            parts[i] = CuArray(x[
                :, ((i - 1) * partlen + 1 - ng):(i * partlen + ng)])
        end
        devparts[i] = ArrayPart(cudaconvert(parts[i]), i, ndev, ng)
    end

    device!(0)
    return MultiArray{T, 2, ng}(devparts, array_ret, size(x))
end

function JACC.Multi.array(::CUDABackend, x::Base.Array; ghost_dims)
    if ghost_dims == 0 || ndevices() == 1
        return make_multi_array(x)
    else
        return make_multi_array(x, ghost_dims)
    end
end

function JACC.Multi.ghost_shift(::CUDABackend, i::Integer, arr::ArrayPart)
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
        ::CUDABackend, (i, j)::NTuple{2, Integer}, arr::ArrayPart)
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
        ::CUDABackend, arr::MultiArray{T, 1}) where {T}
    device!(0)
    ndev = ndevices()
    ng = ghost_dims(arr)
    if ng == 0
        return
    end

    #Left to right swapping
    for i in 1:(ndev - 1)
        device!(i - 1)
        tmp = Base.Array(arr.a2[i])
        size = length(tmp)
        device!(i)
        ghost_lr = CuArray(tmp[(size + 1 - 2 * ng):(size - ng)])
        @cuda threads=32 blocks=1 _multi_swap_ghost_lr(arr.a1[i + 1], ghost_lr)
    end

    #Right to left swapping
    for i in 2:ndev
        device!(i - 1)
        tmp = Base.Array(arr.a2[i])
        size = length(tmp)
        device!(i - 2)
        ghost_rl = CuArray(tmp[(1 + ng):(2 * ng)])
        @cuda threads=32 blocks=1 _multi_swap_ghost_rl(
            arr.a1[i - 1], ghost_rl)
    end

    for i in 1:ndev
        device!(i - 1)
        CUDA.synchronize()
    end

    device!(0)
end

function JACC.Multi.sync_ghost_elems!(
        ::CUDABackend, arr::MultiArray{T, 2}) where {T}
    device!(0)
    ndev = ndevices()
    ng = ghost_dims(arr)
    if ng == 0
        return
    end

    #Left to right swapping
    for i in 1:(ndev - 1)
        device!(i - 1)
        dim = size(arr.a2[i])
        tmp = Base.Array(arr.a2[i][:, (dim[2] + 1 - 2 * ng):(dim[2] - ng)])
        device!(i)
        ghost_lr = CuArray(tmp)
        numThreads = 512
        threads = min(dim[1], numThreads)
        blocks = cld(dim[1], threads)
        @cuda threads=threads blocks=blocks _multi_swap_2d_ghost_lr(
            arr.a1[i + 1], ghost_lr)
    end

    #Right to left swapping
    for i in 2:ndev
        device!(i - 1)
        tmp = Base.Array(arr.a2[i][:, (1 + ng):(2 * ng)])
        device!(i - 2)
        dim = size(arr.a2[i - 1])
        ghost_rl = CuArray(tmp)
        numThreads = 512
        threads = min(dim[1], numThreads)
        blocks = cld(dim[1], threads)
        @cuda threads=threads blocks=blocks _multi_swap_2d_ghost_rl(
            arr.a1[i - 1], ghost_rl)
    end

    for i in 1:ndev
        device!(i - 1)
        CUDA.synchronize()
    end

    device!(0)
end

function JACC.Multi.copy!(::CUDABackend, x::MultiArray, y::MultiArray)
    device!(0)
    ndev = ndevices()

    if ndims(x.a2[1]) == 1
        numThreads = 512
        if ghost_dims(x) == 0 && ghost_dims(y) != 0
            # "gcopytoarray"
            for i in 1:ndev
                device!(i - 1)
                size = length(y.a2[i])
                threads = min(size, numThreads)
                blocks = cld(size, threads)
                @cuda threads=threads blocks=blocks _multi_copy_ghosttoarray(
                    x.a1[i], y.a1[i], ndev)
            end
        elseif ghost_dims(x) != 0 && ghost_dims(y) == 0
            # "copytogarray"
            for i in 1:ndev
                device!(i - 1)
                size = length(x.a2[i])
                threads = min(size, numThreads)
                blocks = cld(size, threads)
                @cuda threads=threads blocks=blocks _multi_copy_arraytoghost(
                    x.a1[i], y.a1[i], ndev)
            end
        else
            for i in 1:ndev
                device!(i - 1)
                size = length(x.a2[i])
                threads = min(size, numThreads)
                blocks = cld(size, threads)
                @cuda threads=threads blocks=blocks _multi_copy(
                    x.a1[i], y.a1[i])
            end
        end

        for i in 1:ndev
            device!(i - 1)
            CUDA.synchronize()
        end

    elseif ndims(x.a2[1]) == 2

        # TODO: handle arrays with ghost elements

        for i in 1:ndev
            device!(i - 1)
            ssize = size(x.a2[i])
            numThreads = 16
            Mthreads = min(ssize[1], numThreads)
            Mblocks = cld(ssize[1], Mthreads)
            Nthreads = min(ssize[2], numThreads)
            Nblocks = cld(ssize[2], Nthreads)
            threads = (Mthreads, Nthreads)
            blocks = (Mblocks, Nblocks)
            @cuda threads=threads blocks=blocks _multi_copy_2d(x.a1[i], y.a1[i])
        end

        for i in 1:ndev
            device!(i - 1)
            CUDA.synchronize()
        end
    end

    device!(0)
end

function JACC.Multi.parallel_for(::CUDABackend, N::Integer, f::Callable, x...)
    device!(0)
    ndev = length(devices())
    N_multi = cld(N, ndev)
    numThreads = 256
    threads = min(N_multi, numThreads)
    blocks = cld(N_multi, threads)

    for i in 1:ndev
        device!(i - 1)
        dev_id = i
        @cuda threads=threads blocks=blocks _multi_parallel_for_cuda(
            N_multi, f, process_param.((x), dev_id)...)
    end

    for i in 1:ndev
        device!(i - 1)
        CUDA.synchronize()
    end

    device!(0)
end

function JACC.Multi.parallel_for(::CUDABackend,
        (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = length(devices())
    N_multi = cld(N, ndev)
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N_multi, numThreads)
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N_multi, Nthreads)
    blocks = (Mblocks, Nblocks)

    for i in 1:ndev
        device!(i - 1)
        dev_id = i
        @cuda threads=threads blocks=blocks _multi_parallel_for_cuda_MN(
            M, N_multi, f, process_param.((x), dev_id)...)
    end

    for i in 1:ndev
        device!(i - 1)
        CUDA.synchronize()
    end

    device!(0)
end

function JACC.Multi.parallel_reduce(
        ::CUDABackend, N::Integer, f::Callable, x...)
    device!(0)
    ndev = length(devices())
    ret = Vector{Any}(undef, ndev)
    rret = Vector{Any}(undef, ndev)
    N_multi = cld(N, ndev)
    numThreads = 512
    threads = min(N_multi, numThreads)
    blocks = cld(N_multi, threads)
    shmem_size = 512 * sizeof(Float64)

    for i in 1:ndev
        device!(i - 1)
        ret[i] = CUDA.zeros(Float64, blocks)
        rret[i] = CUDA.zeros(Float64, 1)
    end

    for i in 1:ndev
        device!(i - 1)
        dev_id = i
        @cuda threads=threads blocks=blocks shmem=shmem_size _multi_parallel_reduce_cuda(
            N_multi, ret[i], f, process_param.((x), dev_id)...)
        @cuda threads=threads blocks=1 shmem=shmem_size _multi_reduce_kernel_cuda(
            blocks, ret[i], rret[i])
    end

    for i in 1:ndev
        device!(i - 1)
        CUDA.synchronize()
    end

    final_rret = 0.0
    for i in 1:ndev
        final_rret += Base.Array(rret[i])[]
    end

    device!(0)

    return final_rret
end

function JACC.Multi.parallel_reduce(::CUDABackend,
        (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = length(devices())
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
    shmem_size = 16 * 16 * sizeof(Float64)

    for i in 1:ndev
        device!(i - 1)
        ret[i] = CUDA.zeros(Float64, (Mblocks, Nblocks))
        rret[i] = CUDA.zeros(Float64, 1)
    end

    for i in 1:ndev
        device!(i - 1)
        dev_id = i

        @cuda threads=threads blocks=blocks shmem=shmem_size _multi_parallel_reduce_cuda_MN(
            (M, N_multi), ret[i], f, process_param.((x), dev_id)...)

        @cuda threads=threads blocks=(1, 1) shmem=shmem_size _multi_reduce_kernel_cuda_MN(
            blocks, ret[i], rret[i])
    end

    for i in 1:ndev
        device!(i - 1)
        CUDA.synchronize()
    end

    final_rret = 0.0
    for i in 1:ndev
        final_rret += Base.Array(rret[i])[]
    end

    device!(0)

    return final_rret
end

# COV_EXCL_START
function _multi_copy(x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] = y[i]
    end
    return nothing
end

function _multi_copy_2d(x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    dims = size(x)
    if i <= dims[1] && j <= dims[2]
        @inbounds x[i, j] = y[i, j]
    end
    return nothing
end

function _multi_copy_ghosttoarray(x::ArrayPart, y::ArrayPart, ndev::Integer)
    #x is the array and y is the ghost array
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
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
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
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
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ng = ghost_dims(arr)
    if i <= ng
        arr[i] = ghost[i]
    end
    return nothing
end

function _multi_swap_ghost_rl(arr::ArrayPart, ghost)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ng = ghost_dims(arr)
    len = length(arr)
    if i <= ng
        arr[len - ng + i] = ghost[i]
    end
    return nothing
end

function _multi_swap_2d_ghost_lr(arr::ArrayPart, ghost)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i < size(arr, 1) + 1
        ng = ghost_dims(arr)
        for n in 1:ng
            arr[i, n] = ghost[i, n]
        end
    end
    return nothing
end

function _multi_swap_2d_ghost_rl(arr::ArrayPart, ghost)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    dim = size(arr)
    if i < dim[1] + 1
        ng = ghost_dims(arr)
        for n in 1:ng
            arr[i, dim[2] - ng + n] = ghost[i, n]
        end
    end
    return nothing
end

function _multi_parallel_for_cuda(N, f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        f(i, x...)
    end
    return nothing
end

function _multi_parallel_for_cuda_MN(M, N, f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (i <= M) && (j <= N)
        f(i, j, x...)
    end
    return nothing
end

function _multi_parallel_reduce_cuda(N, ret, f, x...)
    shared_mem = CuDynamicSharedArray(Float64, 512)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x
    tmp::Float64 = 0.0
    shared_mem[ti] = 0.0

    if i <= N
        tmp = @inbounds f(i, x...)
        shared_mem[threadIdx().x] = tmp
    end
    sync_threads()
    if (ti <= 256)
        shared_mem[ti] += shared_mem[ti + 256]
    end
    sync_threads()
    if (ti <= 128)
        shared_mem[ti] += shared_mem[ti + 128]
    end
    sync_threads()
    if (ti <= 64)
        shared_mem[ti] += shared_mem[ti + 64]
    end
    sync_threads()
    if (ti <= 32)
        shared_mem[ti] += shared_mem[ti + 32]
    end
    sync_threads()
    if (ti <= 16)
        shared_mem[ti] += shared_mem[ti + 16]
    end
    sync_threads()
    if (ti <= 8)
        shared_mem[ti] += shared_mem[ti + 8]
    end
    sync_threads()
    if (ti <= 4)
        shared_mem[ti] += shared_mem[ti + 4]
    end
    sync_threads()
    if (ti <= 2)
        shared_mem[ti] += shared_mem[ti + 2]
    end
    sync_threads()
    if (ti == 1)
        shared_mem[ti] += shared_mem[ti + 1]
        ret[blockIdx().x] = shared_mem[ti]
    end
    return nothing
end

function _multi_reduce_kernel_cuda(N, red, ret)
    shared_mem = CuDynamicSharedArray(Float64, 512)
    i = threadIdx().x
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
    shared_mem[threadIdx().x] = tmp
    sync_threads()
    if (i <= 256)
        shared_mem[i] += shared_mem[i + 256]
    end
    sync_threads()
    if (i <= 128)
        shared_mem[i] += shared_mem[i + 128]
    end
    sync_threads()
    if (i <= 64)
        shared_mem[i] += shared_mem[i + 64]
    end
    sync_threads()
    if (i <= 32)
        shared_mem[i] += shared_mem[i + 32]
    end
    sync_threads()
    if (i <= 16)
        shared_mem[i] += shared_mem[i + 16]
    end
    sync_threads()
    if (i <= 8)
        shared_mem[i] += shared_mem[i + 8]
    end
    sync_threads()
    if (i <= 4)
        shared_mem[i] += shared_mem[i + 4]
    end
    sync_threads()
    if (i <= 2)
        shared_mem[i] += shared_mem[i + 2]
    end
    sync_threads()
    if (i == 1)
        shared_mem[i] += shared_mem[i + 1]
        ret[1] = shared_mem[1]
    end
    return nothing
end

function _multi_parallel_reduce_cuda_MN((M, N), ret, f, x...)
    shared_mem = CuDynamicSharedArray(Float64, 16 * 16)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ti = threadIdx().x
    tj = threadIdx().y
    bi = blockIdx().x
    bj = blockIdx().y

    sid = ((ti - 1) * 16) + tj
    tmp::Float64 = 0.0
    shared_mem[sid] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds f(i, j, x...)
        shared_mem[sid] = tmp
    end
    sync_threads()
    if (ti <= 8 && tj <= 8)
        shared_mem[sid] += shared_mem[((ti + 7) * 16) + (tj + 8)]
        shared_mem[sid] += shared_mem[((ti - 1) * 16) + (tj + 8)]
        shared_mem[sid] += shared_mem[((ti + 7) * 16) + tj]
    end
    sync_threads()
    if (ti <= 4 && tj <= 4)
        shared_mem[sid] += shared_mem[((ti + 3) * 16) + (tj + 4)]
        shared_mem[sid] += shared_mem[((ti - 1) * 16) + (tj + 4)]
        shared_mem[sid] += shared_mem[((ti + 3) * 16) + tj]
    end
    sync_threads()
    if (ti <= 2 && tj <= 2)
        shared_mem[sid] += shared_mem[((ti + 1) * 16) + (tj + 2)]
        shared_mem[sid] += shared_mem[((ti - 1) * 16) + (tj + 2)]
        shared_mem[sid] += shared_mem[((ti + 1) * 16) + tj]
    end
    sync_threads()
    if (ti == 1 && tj == 1)
        shared_mem[sid] += shared_mem[ti * 16 + (tj + 1)]
        shared_mem[sid] += shared_mem[((ti - 1) * 16) + (tj + 1)]
        shared_mem[sid] += shared_mem[ti * 16 + tj]
        ret[bi, bj] = shared_mem[sid]
    end
    return nothing
end

function _multi_reduce_kernel_cuda_MN((M, N), red, ret)
    shared_mem = CuDynamicSharedArray(Float64, 16 * 16)
    i = threadIdx().x
    j = threadIdx().y
    ii = i
    jj = j

    sid = ((i - 1) * 16) + j
    tmp::Float64 = 0.0
    shared_mem[sid] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = threadIdx().y
            while jj <= N
                tmp = tmp + @inbounds red[ii, jj]
                jj += 16
            end
            ii += 16
        end
    elseif M > 16
        while ii <= N
            tmp = tmp + @inbounds red[ii, j]
            ii += 16
        end
    elseif N > 16
        while jj <= N
            tmp = tmp + @inbounds red[i, jj]
            jj += 16
        end
    elseif M <= 16 && N <= 16
        if i <= M && j <= N
            tmp = tmp + @inbounds red[i, j]
        end
    end
    shared_mem[sid] = tmp
    sync_threads()
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
    sync_threads()
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
    sync_threads()
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
    sync_threads()
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
