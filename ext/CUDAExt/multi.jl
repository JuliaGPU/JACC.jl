module Multi

import Base: Callable
using JACC, CUDA
using CUDAExt: CUDATag

@inline ndevices() = length(devices())

function JACC.Multi.ndev(::CUDATag)
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

@inline JACC.Multi.device_id(::CUDATag, p::ArrayPart) = p.dev_id

struct MultiArray{T, N, NG}
    a1::Vector{ArrayPart{T, N}}
    a2::Vector{CuArray{T, N}}
    orig_size::Any
end

JACC.to_host(x::MultiArray) = convert(Base.Array, x)

@inline ghost_dims(x::MultiArray{T, N, NG}) where {T, N, NG} = NG
@inline JACC.Multi.part_length(::CUDATag, x::MultiArray) = size(x.a2[1])[end]

@inline process_param(x, dev_id) = x
@inline process_param(x::MultiArray, dev_id) = x.a1[dev_id]

JACC.Multi.multi_array_type(::CUDATag) = MultiArray

function Base.convert(::Type{Base.Array}, x::MultiArray{T, 1, NG}) where {T, NG}
    ndev = ndevices()
    ret = Base.Array{T, 1}(undef, x.orig_size)
    partlen = cld(x.orig_size, ndev)
    lastlen = x.orig_size - ((ndev - 1) * partlen)
    for i in 1:ndev
        device!(i - 1)
        if i == 1
            copyto!(ret, (((i - 1) * partlen) + 1), x.a2[i], 1, partlen)
        elseif i == ndev
            copyto!(ret, (((i - 1) * partlen) + 1), x.a2[i], 1 + NG, lastlen)
        else
            copyto!(ret, (((i - 1) * partlen) + 1), x.a2[i], 1 + NG, partlen)
        end
    end
    device!(0)
    return ret
end

function Base.convert(::Type{Base.Array}, x::MultiArray{T, 2, NG}) where {T, NG}
    ndev = ndevices()
    ret = Base.Array{T, 2}(undef, x.orig_size)
    total_length = x.orig_size[2]
    partlen = cld(total_length, ndev)
    lastlen = total_length - ((ndev - 1) * partlen)
    for i in 1:ndev
        device!(i - 1)
        if i == 1
            copyto!(
                ret,
                CartesianIndices((
                    1:size(x.a2[i], 1),
                    (((i - 1) * partlen) + 1):(i * partlen)
                )),
                x.a2[i],
                CartesianIndices((1:size(x.a2[i], 1), 1:partlen))
            )
        elseif i == ndev
            copyto!(
                ret,
                CartesianIndices((
                    1:size(x.a2[i], 1),
                    (((i - 1) * partlen) + 1):total_length
                )),
                x.a2[i],
                CartesianIndices((1:size(x.a2[i], 1), (1 + NG):(lastlen + NG)))
            )
        else
            copyto!(
                ret,
                CartesianIndices((
                    1:size(x.a2[i], 1),
                    (((i - 1) * partlen) + 1):(i * partlen)
                )),
                x.a2[i],
                CartesianIndices((1:size(x.a2[i], 1), (1 + NG):(partlen + NG)))
            )
        end
    end
    device!(0)
    return ret
end

function make_multi_array(x::Base.Vector{T}) where {T}
    ndev = ndevices()
    total_length = length(x)
    partlen = cld(total_length, ndev)
    parts = Vector{CuVector{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 1}}(undef, ndev)

    for i in 1:ndev
        device!(i - 1)
        if i == ndev
            parts[i] = CuArray(x[(((i - 1) * partlen) + 1):total_length])
        else
            parts[i] = CuArray(x[(((i - 1) * partlen) + 1):(i * partlen)])
        end
        devparts[i] = ArrayPart(cudaconvert(parts[i]), i, ndev, 0)
    end

    device!(0)
    return MultiArray{T, 1, 0}(devparts, parts, total_length)
end

function make_multi_array(x::Base.Vector{T}, ghost_dims) where {T}
    ndev = ndevices()
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
            parts[i] = CuArray(x[((i - 1) * partlen + 1 - ng):total_length])
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
    total_length = size(x, 2)
    partlen = cld(total_length, ndev)
    parts = Vector{CuMatrix{T}}(undef, ndev)
    devparts = Vector{ArrayPart{T, 2}}(undef, ndev)

    for i in 1:ndev
        device!(i - 1)
        if i == ndev
            parts[i] = CuArray(x[:, (((i - 1) * partlen) + 1):total_length])
        else
            parts[i] = CuArray(x[:, (((i - 1) * partlen) + 1):(i * partlen)])
        end
        devparts[i] = ArrayPart(cudaconvert(parts[i]), i, ndev, 0)
    end

    device!(0)
    return MultiArray{T, 2, 0}(devparts, parts, size(x))
end

function make_multi_array(x::Base.Matrix{T}, ghost_dims) where {T}
    ndev = ndevices()
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
            parts[i] = CuArray(x[:, ((i - 1) * partlen + 1 - ng):total_length])
        else
            parts[i] = CuArray(x[
                :, ((i - 1) * partlen + 1 - ng):(i * partlen + ng)])
        end
        devparts[i] = ArrayPart(cudaconvert(parts[i]), i, ndev, ng)
    end

    device!(0)
    return MultiArray{T, 2, ng}(devparts, parts, size(x))
end

function JACC.Multi.array(::CUDATag, x::Base.Array; ghost_dims)
    if ghost_dims == 0 || ndevices() == 1
        return make_multi_array(x)
    else
        return make_multi_array(x, ghost_dims)
    end
end

function JACC.Multi.ghost_shift(::CUDATag, i::Integer, arr::ArrayPart)
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
        ::CUDATag, (i, j)::NTuple{2, Integer}, arr::ArrayPart)
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
        ::CUDATag, arr::MultiArray{T, 1}) where {T}
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
        ::CUDATag, arr::MultiArray{T, 2}) where {T}
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

function JACC.Multi.copy!(::CUDATag, x::MultiArray, y::MultiArray)
    device!(0)
    ndev = ndevices()

    if ndims(x.a2[1]) == 1
        numThreads = 512
        if ghost_dims(x) == 0 && ghost_dims(y) != 0
            # "gcopytoarray"
            for i in 1:ndev
                device!(i - 1)
                size = length(x.a2[i])
                threads = min(size, numThreads)
                blocks = cld(size, threads)
                @cuda threads=threads blocks=blocks _multi_copy_ghosttoarray(
                    x.a1[i], y.a1[i], ndev)
            end
        elseif ghost_dims(x) != 0 && ghost_dims(y) == 0
            # "copytogarray"
            for i in 1:ndev
                device!(i - 1)
                size = length(y.a2[i])
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

function JACC.Multi.parallel_for(::CUDATag, N::Integer, f::Callable, x...)
    ndev = length(devices())
    N_part = cld(N, ndev)
    N_last = N - ((ndev - 1) * N_part)
    numThreads = 256
    threads = min(N_part, numThreads)
    blocks = cld(N_part, threads)

    for i in 1:ndev
        device!(i - 1)
        dev_id = i
        if i == ndev
            @cuda threads=threads blocks=blocks _multi_parallel_for_cuda(
                N_last, f, process_param.((x), dev_id)...)
        else
            @cuda threads=threads blocks=blocks _multi_parallel_for_cuda(
                N_part, f, process_param.((x), dev_id)...)
        end
    end

    for i in 1:ndev
        device!(i - 1)
        CUDA.synchronize()
    end

    device!(0)
end

function JACC.Multi.parallel_for(::CUDATag,
        (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = length(devices())
    N_part = cld(N, ndev)
    N_last = N - ((ndev - 1) * N_part)
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N_part, numThreads)
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N_part, Nthreads)
    blocks = (Mblocks, Nblocks)

    for i in 1:ndev
        device!(i - 1)
        dev_id = i
        if i == ndev
            @cuda threads=threads blocks=blocks _multi_parallel_for_cuda_MN(
                M, N_last, f, process_param.((x), dev_id)...)
        else
            @cuda threads=threads blocks=blocks _multi_parallel_for_cuda_MN(
                M, N_part, f, process_param.((x), dev_id)...)
        end
    end

    for i in 1:ndev
        device!(i - 1)
        CUDA.synchronize()
    end

    device!(0)
end

function JACC.Multi.parallel_reduce(
        ::CUDATag, N::Integer, f::Callable, x...)
    ndev = length(devices())
    ret = Vector{Any}(undef, ndev)
    rret = Vector{Any}(undef, ndev)
    N_part = cld(N, ndev)
    N_last = N - ((ndev - 1) * N_part)
    numThreads = 512
    threads = numThreads
    blocks = cld(N_part, threads)
    shmem_size = 512 * sizeof(Float64)

    for i in 1:ndev
        device!(i - 1)
        ret[i] = CUDA.zeros(Float64, blocks)
        rret[i] = CUDA.zeros(Float64, 1)
    end

    for i in 1:ndev
        device!(i - 1)
        dev_id = i
        if i == ndev
            @cuda threads=threads blocks=blocks shmem=shmem_size _multi_parallel_reduce_cuda(
                N_last, ret[i], f, process_param.((x), dev_id)...)
        else
            @cuda threads=threads blocks=blocks shmem=shmem_size _multi_parallel_reduce_cuda(
                N_part, ret[i], f, process_param.((x), dev_id)...)
        end
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

function JACC.Multi.parallel_reduce(::CUDATag,
        (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = length(devices())
    ret = Vector{Any}(undef, ndev)
    rret = Vector{Any}(undef, ndev)
    N_part = cld(N, ndev)
    N_last = N - ((ndev - 1) * N_part)
    numThreads = 16
    Mthreads = numThreads
    Nthreads = numThreads
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N_part, Nthreads)
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

        if i == ndev
            @cuda threads=threads blocks=blocks shmem=shmem_size _multi_parallel_reduce_cuda_MN(
                (M, N_last), ret[i], f, process_param.((x), dev_id)...)
        else
            @cuda threads=threads blocks=blocks shmem=shmem_size _multi_parallel_reduce_cuda_MN(
                (M, N_part), ret[i], f, process_param.((x), dev_id)...)
        end

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
    id = JACC.Multi.ghost_shift(CUDABackend, i, y)
    x[i] = y[id]
    return nothing
end

function _multi_copy_arraytoghost(x::ArrayPart, y::ArrayPart, ndev::Integer)
    #x is the ghost array and y is the array
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    id = JACC.Multi.ghost_shift(CUDABackend, i, x)
    x[id] = y[i]
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
    shmem_length = blockDim().x
    shared_mem = CuDynamicSharedArray(Float64, shmem_length)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x

    @inbounds shared_mem[ti] = 0.0

    if i <= N
        tmp = @inline f(i, x...)
        @inbounds shared_mem[ti] = tmp
    end

    op = +
    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        sync_threads()
        tn = 2^p
        if (ti <= tn)
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + tn])
        end
    end

    if ti == 1
        @inbounds ret[blockIdx().x] = shared_mem[ti]
    end

    return nothing
end

function _multi_reduce_kernel_cuda(N, red, ret)
    shmem_length = blockDim().x
    shared_mem = CuDynamicSharedArray(Float64, shmem_length)
    i = threadIdx().x

    op = +
    tmp::Float64 = 0.0
    for ii in i:shmem_length:N
        tmp = op(tmp, @inbounds red[ii])
    end
    @inbounds shared_mem[threadIdx().x] = tmp

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        sync_threads()
        tn = 2^p
        if i <= tn
            @inbounds shared_mem[i] = op(shared_mem[i], shared_mem[i + tn])
        end
    end

    if i == 1
        @inbounds ret[1] = shared_mem[1]
    end

    return nothing
end

function _multi_parallel_reduce_cuda_MN((M, N), ret, f, x...)
    shared_mem = CuDynamicSharedArray(Float64, (16, 16))
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ti = threadIdx().x
    tj = threadIdx().y
    bi = blockIdx().x
    bj = blockIdx().y

    tmp::Float64 = 0.0
    @inbounds shared_mem[ti, tj] = tmp

    if (i <= M && j <= N)
        tmp = @inline f(i, j, x...)
        @inbounds shared_mem[ti, tj] = tmp
    end

    op = +
    for n in (8, 4, 2, 1)
        sync_threads()
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

function _multi_reduce_kernel_cuda_MN((M, N), red, ret)
    shared_mem = CuDynamicSharedArray(Float64, (16, 16))
    i = threadIdx().x
    j = threadIdx().y

    tmp::Float64 = 0.0
    for ci in CartesianIndices((i:16:M, j:16:N))
        tmp = +(tmp, @inbounds red[ci])
    end
    @inbounds shared_mem[i, j] = tmp

    op = +
    for n in (8, 4, 2, 1)
        sync_threads()
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


# COV_EXCL_STOP

end # module Multi
