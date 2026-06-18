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
    ti = threadIdx().x
    tj = threadIdx().y
    bi = blockIdx().x
    bj = blockIdx().y

    tmp::Float64 = 0.0
    shared_mem[((ti - 1) * 16) + tj] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds x[i, j] * y[i, j]
        shared_mem[(ti - 1) * 16 + tj] = tmp
    end
    sync_threads()
    if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + tj]
    end
    sync_threads()
    if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + tj]
    end
    sync_threads()
    if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + tj]
    end
    sync_threads()
    if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + tj]
        ret[bi, bj] = shared_mem[((ti - 1) * 16) + tj]
    end
    return nothing
end

function reduce_kernel((M, N), red, ret)
    shared_mem = @cuDynamicSharedMem(Float64, 16*16)

    i = threadIdx().x
    j = threadIdx().y
    ii = i
    jj = j

    tmp::Float64 = 0.0
    shared_mem[(i - 1) * 16 + j] = tmp

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
    shared_mem[(i - 1) * 16 + j] = tmp
    red[i, j] = shared_mem[(i - 1) * 16 + j]
    sync_threads()
    if (i <= 8 && j <= 8)
        if (i + 8 <= M && j + 8 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 7) * 16) + (j + 8)]
        end
        if (i <= M && j + 8 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 8)]
        end
        if (i + 8 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 7) * 16) + j]
        end
    end
    sync_threads()
    if (i <= 4 && j <= 4)
        if (i + 4 <= M && j + 4 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 3) * 16) + (j + 4)]
        end
        if (i <= M && j + 4 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 4)]
        end
        if (i + 4 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 3) * 16) + j]
        end
    end
    sync_threads()
    if (i <= 2 && j <= 2)
        if (i + 2 <= M && j + 2 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 1) * 16) + (j + 2)]
        end
        if (i <= M && j + 2 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 2)]
        end
        if (i + 2 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 1) * 16) + j]
        end
    end
    sync_threads()
    if (i == 1 && j == 1)
        if (i + 1 <= M && j + 1 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[i * 16 + (j + 1)]
        end
        if (i <= M && j + 1 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 1)]
        end
        if (i + 1 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[i * 16 + j]
        end
        ret[1] = shared_mem[((i - 1) * 16) + j]
    end
    return nothing
end

function dot_cuda((M, N), x, y)
    maxPossibleThreads = 16
    Mthreads = min(M, maxPossibleThreads)
    Nthreads = min(N, maxPossibleThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)
    ret = CUDA.zeros(Float64, (Mblocks, Nblocks))
    rret = CUDA.zeros(Float64, 1)
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) shmem=16*
                                                                                  16*
                                                                                  sizeof(Float64) dot_cuda_kernel(
        (M, N), ret, x, y)
    CUDA.@sync @cuda threads=(Mblocks, Nblocks) blocks=(1, 1) shmem=16*16*
                                                                    sizeof(Float64) reduce_kernel(
        (Mblocks, Nblocks), ret, rret)
    return rret
end

SIZE = 200
x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
dx = CuArray(x)
dy = CuArray(y)
@time begin
    res = dot_cuda((SIZE, SIZE), dx, dy)
end

#-------------------------AMDGPU

function dot_amdgpu_kernel((M, N), ret, x, y)
    shared_mem = @ROCDynamicLocalArray(Float64, 16*16)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ti = workitemIdx().x
    tj = workitemIdx().y
    bi = workgroupIdx().x
    bj = workgroupIdx().y

    tmp::Float64 = 0.0
    shared_mem[((ti - 1) * 16) + tj] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds x[i, j] * y[i, j]
        shared_mem[(ti - 1) * 16 + tj] = tmp
    end
    AMDGPU.sync_workgroup()
    if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + tj]
        ret[bi, bj] = shared_mem[((ti - 1) * 16) + tj]
    end
    return nothing
end

function reduce_kernel((M, N), red, ret)
    shared_mem = @ROCDynamicLocalArray(Float64, 16*16)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ii = i
    jj = j

    tmp::Float64 = 0.0
    shared_mem[(i - 1) * 16 + j] = tmp

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
    shared_mem[(i - 1) * 16 + j] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 8 && j <= 8)
        if (i + 8 <= M && j + 8 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 7) * 16) + (j + 8)]
        end
        if (i <= M && j + 8 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 8)]
        end
        if (i + 8 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 7) * 16) + j]
        end
    end
    AMDGPU.sync_workgroup()
    if (i <= 4 && j <= 4)
        if (i + 4 <= M && j + 4 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 3) * 16) + (j + 4)]
        end
        if (i <= M && j + 4 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 4)]
        end
        if (i + 4 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 3) * 16) + j]
        end
    end
    AMDGPU.sync_workgroup()
    if (i <= 2 && j <= 2)
        if (i + 2 <= M && j + 2 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 1) * 16) + (j + 2)]
        end
        if (i <= M && j + 2 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 2)]
        end
        if (i + 2 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 1) * 16) + j]
        end
    end
    AMDGPU.sync_workgroup()
    if (i == 1 && j == 1)
        if (i + 1 <= M && j + 1 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[i * 16 + (j + 1)]
        end
        if (i <= M && j + 1 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 1)]
        end
        if (i + 1 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[i * 16 + j]
        end
        ret[1] = shared_mem[((i - 1) * 16) + j]
    end
    return nothing
end

function dot_amdgpu((M, N), x, y)
    maxPossibleThreads = 16
    Mthreads = min(M, maxPossibleThreads)
    Nthreads = min(N, maxPossibleThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)
    ret = AMDGPU.zeros(Float64, (Mblocks, Nblocks))
    rret = AMDGPU.zeros(Float64, 1)
    @roc groupsize=(Mthreads, Nthreads) gridsize=(
        Mblocks*Mthreads, Nblocks*Nthreads) localmem=16*16*
                                                     sizeof(Float64) dot_amdgpu_kernel(
        (M, N), ret, x, y)
    @roc groupsize=(Mblocks, Nblocks) gridsize=(Mblocks, Nblocks) localmem=16*
                                                                           16*
                                                                           sizeof(Float64) reduce_kernel(
        (Mblocks, Nblocks), ret, rret)
    return rret
end

SIZE = 200
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
    ti = get_local_id(0)
    tj = get_local_id(1)
    bi = get_group_id(0)
    bj = get_group_id(1)

    tmp::Float32 = 0.0
    shared_mem[((ti - 1) * 16) + tj] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds x[i, j] * y[i, j]
        shared_mem[(ti - 1) * 16 + tj] = tmp
    end
    barrier()
    if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + tj]
    end
    barrier()
    if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + tj]
    end
    barrier()
    if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + tj]
    end
    barrier()
    if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + tj]
        ret[bi, bj] = shared_mem[((ti - 1) * 16) + tj]
    end
    return nothing
end

function reduce_kernel((M, N), red, ret)
    shared_mem = oneLocalArray(Float32, 16 * 16)
    i = get_local_id(0)
    j = get_local_id(1)
    ii = i
    jj = j

    tmp::Float32 = 0.0
    shared_mem[(i - 1) * 16 + j] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = get_local_id(1)
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
    shared_mem[(i - 1) * 16 + j] = tmp
    barrier()
    if (i <= 8 && j <= 8)
        if (i + 8 <= M && j + 8 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 7) * 16) + (j + 8)]
        end
        if (i <= M && j + 8 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 8)]
        end
        if (i + 8 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 7) * 16) + j]
        end
    end
    barrier()
    if (i <= 4 && j <= 4)
        if (i + 4 <= M && j + 4 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 3) * 16) + (j + 4)]
        end
        if (i <= M && j + 4 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 4)]
        end
        if (i + 4 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 3) * 16) + j]
        end
    end
    barrier()
    if (i <= 2 && j <= 2)
        if (i + 2 <= M && j + 2 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 1) * 16) + (j + 2)]
        end
        if (i <= M && j + 2 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 2)]
        end
        if (i + 2 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 1) * 16) + j]
        end
    end
    barrier()
    if (i == 1 && j == 1)
        if (i + 1 <= M && j + 1 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[i * 16 + (j + 1)]
        end
        if (i <= M && j + 1 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 1)]
        end
        if (i + 1 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[i * 16 + j]
        end
        ret[1] = shared_mem[((i - 1) * 16) + j]
    end
    return nothing
end

function dot_oneapi((M, N), x, y)
    maxPossibleItems = 16
    Mitems = min(M, maxPossibleItems)
    Nitems = min(N, maxPossibleItems)
    Mgroups = ceil(Int, M / Mitems)
    Ngroups = ceil(Int, N / Nitems)
    ret = oneAPI.zeros(Float32, (Mgroups, Ngroups))
    rret = oneAPI.zeros(Float32, 1)
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(Mgroups, Ngroups) dot_oneapi_kernel(
        (M, N), ret, x, y)
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(1, 1) reduce_kernel(
        (Mgroups, Ngroups), ret, rret)
    return rret
end

SIZE = 200
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

SIZE = 200
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
