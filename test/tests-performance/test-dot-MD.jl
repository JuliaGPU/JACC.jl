#-------------------------Threads  

function dot_threads((M, N), x, y)
    tmp = zeros(Threads.nthreads())
    ret = zeros(1)
    Threads.@threads for j in 1:N
        for i in 1:M
            tmp[Threads.threadid()] = tmp[Threads.threadid()] .+
                                      x[i, j] * y[i, j]
        end
    end
    for i in 1:Threads.nthreads()
        ret = ret .+ tmp[i]
    end
    return ret
end

SIZE = 3000
x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
@time begin
    dot_threads((SIZE, SIZE), x, y)
end

#-------------------------CUDA

function dot_cuda_kernel((M, N), ret, x, y)
    shared_mem = @cuDynamicSharedMem(Float64, 16*16)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ii = i
    jj = j

    shared_mem[((i - 1) * 16) + j] = 0.0
    tmp::Float64 = 0.0

    if M > 16 && N > 16
        while ii <= M
            jj = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            while jj <= N
                tmp = tmp + @inbounds x[ii, jj] * y[ii, jj]
                jj += 16
            end
            ii += 16
        end
    elseif M > 16
        while ii <= N
            tmp = tmp + @inbounds x[ii, jj] * y[ii, jj]
            ii += 16
        end
    elseif N > 16
        while jj <= N
            tmp = tmp + @inbounds x[ii, jj] * y[ii, jj]
            jj += 16
        end
    elseif M <= 16 && N <= 16
        if i <= M && j <= N
            tmp = tmp + @inbounds x[i, j] * y[i, j]
        end
    end
    shared_mem[(i - 1) * 16 + j] = tmp
    sync_threads()
    if (i <= 8 && j <= 8 && i + 8 <= M && j + 8 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 7) * 16) + (j + 8)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 8)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 7) * 16) + j]
    end
    sync_threads()
    if (i <= 4 && j <= 4 && i + 4 <= M && j + 4 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 3) * 16) + (j + 4)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 4)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 3) * 16) + j]
    end
    sync_threads()
    if (i <= 2 && j <= 2 && i + 2 <= M && j + 2 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 1) * 16) + (j + 2)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 2)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 1) * 16) + j]
    end
    sync_threads()
    if (i == 1 && j == 1 && i + 1 <= M && j + 1 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[i * 16 + (j + 1)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 1)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[i * 16 + j]
        ret[1] = shared_mem[((i - 1) * 16) + j]
    end
    return nothing
end

function dot_cuda((M, N), x, y)
    maxPossibleThreads = 16
    Mthreads = min(M, maxPossibleThreads)
    Nthreads = min(N, maxPossibleThreads)
    ret = CUDA.zeros(1)
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=1 shmem=16*16*
                                                                 sizeof(Float64) dot_cuda_kernel(
        (M, N), ret, x, y)
    return ret[1]
end

SIZE = 3000
x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
dx = CuArray(x)
dy = CuArray(y)
@time begin
    res = dot_cuda((SIZE, SIZE), dx, dy)
end

#-------------------------AMDGPU

function dot_amdgpu_kernel((M, N), ret, x, y)
    shared_mem = @ROCDynamicLocalArray(Float64, (16, 16))
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ii = i
    jj = j
    tmp::Float64 = 0.0

    if M > 16 && N > 16
        while ii <= M
            jj = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
            while jj <= N
                tmp += x[ii, jj] * y[ii, jj]
                jj += 16
            end
            ii += 16
        end
    elseif M > 16
        while ii <= N
            tmp += x[ii, jj] * y[ii, jj]
            ii += 16
        end
    elseif N > 16
        while jj <= N
            tmp += x[ii, jj] * y[ii, jj]
            jj += 16
        end
    else
        tmp = x[i, j] * y[i, j]
    end
    shared_mem[i, j] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 8 && j <= 8)
        shared_mem[i, j] += shared_mem[i + 8, j + 8]
        shared_mem[i, j] += shared_mem[i, j + 8]
        shared_mem[i, j] += shared_mem[i + 8, j]
    end
    AMDGPU.sync_workgroup()
    if (i <= 4 && j <= 4)
        shared_mem[i, j] += shared_mem[i + 4, j + 4]
        shared_mem[i, j] += shared_mem[i, j + 4]
        shared_mem[i, j] += shared_mem[i + 4, j]
    end
    AMDGPU.sync_workgroup()
    if (i <= 2 && j <= 2)
        shared_mem[i, j] += shared_mem[i + 2, j + 2]
        shared_mem[i, j] += shared_mem[i, j + 2]
        shared_mem[i, j] += shared_mem[i + 2, j]
    end
    AMDGPU.sync_workgroup()
    if (i == 1 && j == 1)
        shared_mem[i, j] += shared_mem[i + 1, j + 1]
        shared_mem[i, j] += shared_mem[i, j + 1]
        shared_mem[i, j] += shared_mem[i + 1, j]
        ret[1] += shared_mem[i, j]
    end

    return nothing
end

function dot_amdgpu_kernel((M, N), ret, x, y)
    shared_mem = @ROCDynamicLocalArray(Float64, 16*16)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ii = i
    jj = j
    shared_mem[((i - 1) * 16) + j] = 0.0
    tmp::Float64 = 0.0

    if M > 16 && N > 16
        while ii <= M
            jj = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
            while jj <= N
                tmp = tmp + @inbounds x[ii, jj] * y[ii, jj]
                jj += 16
            end
            ii += 16
        end
    elseif M > 16
        while ii <= N
            tmp = tmp + @inbounds x[ii, jj] * y[ii, jj]
            ii += 16
        end
    elseif N > 16
        while jj <= N
            tmp = tmp + @inbounds x[ii, jj] * y[ii, jj]
            jj += 16
        end
    elseif M <= 16 && N <= 16
        if i <= M && j <= N
            tmp = tmp + @inbounds x[i, j] * y[i, j]
        end
    end
    shared_mem[(i - 1) * 16 + j] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 8 && j <= 8 && i + 8 <= M && j + 8 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 7) * 16) + (j + 8)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 8)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 7) * 16) + j]
    end
    AMDGPU.sync_workgroup()
    if (i <= 4 && j <= 4 && i + 4 <= M && j + 4 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 3) * 16) + (j + 4)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 4)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 3) * 16) + j]
    end
    AMDGPU.sync_workgroup()
    if (i <= 2 && j <= 2 && i + 2 <= M && j + 2 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 1) * 16) + (j + 2)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 2)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 1) * 16) + j]
    end
    AMDGPU.sync_workgroup()
    if (i == 1 && j == 1 && i + 1 <= M && j + 1 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[i * 16 + (j + 1)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 1)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[i * 16 + j]
        ret[1] = shared_mem[((i - 1) * 16) + j]
    end
    return nothing
end

function dot_amdgpu((M, N), x, y)
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N, numThreads)
    ret = AMDGPU.zeros(1)
    @roc groupsize=(Mthreads, Nthreads) gridsize=(Mthreads, Nthreads) localmem=16*
                                                                               16*
                                                                               sizeof(Float64) dot_amdgpu_kernel(
        (M, N), ret, x, y)
    return ret[1]
end

SIZE = 300
x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
dx = ROCArray(x)
dy = ROCArray(y)
@time begin
    res = dot_amdgpu((SIZE, SIZE), dx, dy)
end

#-------------------------oneAPI

function dot_oneapi_kernel((M, N), ret, x, y)
    shared_mem = oneLocalArray(Float32, 16 * 16)
    i = get_global_id(0)
    j = get_global_id(1)
    ii = i
    jj = j

    tmp::Float32 = 0.0

    if M > 16 && N > 16
        while ii <= M
            jj = get_global_id(1)
            while jj <= N
                tmp += @inbounds x[ii, jj] * y[ii, jj]
                jj += 16
            end
            ii += 16
        end
    elseif M > 16
        while ii <= N
            tmp += @inbounds x[ii, jj] * y[ii, jj]
            ii += 16
        end
    elseif N > 16
        while jj <= N
            tmp += @inbounds x[ii, jj] * y[ii, jj]
            jj += 16
        end
    else
        M <= 16 && N <= 16
        if i <= M && j <= N
            tmp = x[i, j] * y[i, j]
        end
    end
    shared_mem[(i - 1) * 16 + j] = tmp
    barrier()
    if (i <= 8 && j <= 8 && i + 8 <= M && j + 8 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 7) * 16) + (j + 8)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 8)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 7) * 16) + j]
    end
    barrier()
    if (i <= 4 && j <= 4 && i + 4 <= M && j + 4 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 3) * 16) + (j + 4)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 4)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 3) * 16) + j]
    end
    barrier()
    if (i <= 2 && j <= 2 && i + 2 <= M && j + 2 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 1) * 16) + (j + 2)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 2)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i + 1) * 16) + j]
    end
    barrier()
    if (i == 1 && j == 1 && i + 1 <= M && j + 1 <= N)
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[i * 16 + (j + 1)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[((i - 1) * 16) + (j + 1)]
        shared_mem[((i - 1) * 16) + j] = shared_mem[((i - 1) * 16) + j] +
                                         shared_mem[i * 16 + j]
        ret[1] = shared_mem[((i - 1) * 16) + j]
    end
    return nothing
end

function dot_oneapi((M, N), x, y)
    numItems = 16
    Mitems = min(M, numItems)
    Nitems = min(N, numItems)
    ret = oneAPI.zeros(Float32, 1)
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=1 dot_oneapi_kernel(
        (M, N), ret, x, y)
    return ret[1]
end

SIZE = 300
x = ones(Float32, SIZE, SIZE)
y = ones(Float32, SIZE, SIZE)
dx = oneArray(x)
dy = oneArray(y)
@time begin
    res = dot_oneapi((SIZE, SIZE), dx, dy)
end

#-------------------------JACC

function dot(i, j, x, y)
    return @inbounds x[i, j] * y[i, j]
end

SIZE = 30
x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
jx = JACC.Array(x)
jy = JACC.Array(y)
@time begin
    res = JACC.parallel_reduce((SIZE, SIZE), dot, jx, jy)
end

function dot(i, j, x, y)
    return @inbounds x[i, j] * y[i, j]
end

SIZE = 300
x = ones(Float32, SIZE, SIZE)
y = ones(Float32, SIZE, SIZE)
jx = JACC.Array(x)
jy = JACC.Array(y)
@time begin
    res = JACC.parallel_reduce((SIZE, SIZE), dot, jx, jy)
end
