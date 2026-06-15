#-------------------------Threads

function dot_threads(SIZE, x, y)
    tmp = zeros(Threads.nthreads())
    ret = zeros(1)
    Threads.@threads for i in 1:SIZE
        tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ x[i] * y[i]
    end
    for i in 1:Threads.nthreads()
        ret = ret .+ tmp[i]
    end
    return ret
end

SIZE = 10000000
x = ones(SIZE)
y = ones(SIZE)
@time begin
    dot_threads(SIZE, x, y)
end

#-------------------------CUDA

function dot_cuda_kernel(SIZE, ret, x, y)
    shared_mem = @cuDynamicSharedMem(Float64, 512)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x
    tmp::Float64 = 0.0
    shared_mem[threadIdx().x] = 0.0

    if i <= SIZE
        tmp = @inbounds x[i] * y[i]
        shared_mem[threadIdx().x] = tmp
        sync_threads()
    end
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

function reduce_kernel(SIZE, red, ret)
    shared_mem = @cuDynamicSharedMem(Float64, 512)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ii = i
    tmp::Float64 = 0.0
    if SIZE > 512
        while ii <= SIZE
            tmp += @inbounds red[ii]
            ii += 512
        end
    else
        tmp = @inbounds red[i]
    end
    shared_mem[i] = tmp
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

function dot_cuda(SIZE, x, y)
    maxPossibleThreads = 512
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE / threads)
    ret = CUDA.zeros(Float64, blocks)
    rret = CUDA.zeros(Float64, 1)
    CUDA.@sync @cuda threads=threads blocks=blocks shmem=512*sizeof(Float64) dot_cuda_kernel(
        SIZE, ret, x, y)
    CUDA.@sync @cuda threads=threads blocks=1 shmem=512*sizeof(Float64) reduce_kernel(
        blocks, ret, rret)
    return rret
end

SIZE = 10000000
x = ones(SIZE)
y = ones(SIZE)
dx = CuArray(x)
dy = CuArray(y)
@time begin
    res = dot_cuda(SIZE, dx, dy)
end

#-------------------------AMDGPU

function dot_amdgpu_kernel(SIZE, ret, x, y)
    shared_mem = @ROCDynamicLocalArray(Float64, 512)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ti = workitemIdx().x
    tmp::Float64 = 0.0
    shared_mem[ti] = 0.0

    if i <= SIZE
        tmp = @inbounds x[i] * y[i]
        shared_mem[ti] = tmp
        AMDGPU.sync_workgroup()
    end
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
    AMDGPU.sync_workgroup()
    return nothing
end

function reduce_kernel(SIZE, red, ret)
    shared_mem = @ROCDynamicLocalArray(Float64, 512)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ii = i
    tmp::Float64 = 0.0
    if SIZE > 512
        while ii <= SIZE
            tmp += @inbounds red[ii]
            ii += 512
        end
    else
        tmp = @inbounds red[i]
    end
    shared_mem[i] = tmp
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
    return nothing
end

function dot_amdgpu(SIZE, x, y)
    maxPossibleThreads = 512
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE / threads)
    ret = AMDGPU.zeros(Float64, blocks)
    rret = AMDGPU.zeros(Float64, 1)
    @roc groupsize=threads gridsize=threads*blocks localmem=512*
    sizeof(Float64) dot_amdgpu_kernel(
        SIZE, ret, x, y)
    @roc groupsize=threads gridsize=threads localmem=512*sizeof(Float64) reduce_kernel(
        blocks, ret, rret)
    return rret
end

SIZE = 10000000
x = ones(SIZE)
y = ones(SIZE)
dx = ROCArray(x)
dy = ROCArray(y)
@time begin
    res = dot_amdgpu(SIZE, dx, dy)
end

#-------------------------oneAPI

function dot_oneapi_kernel(SIZE, ret, x, y)
    shared_mem = oneLocalArray(Float32, 256)
    i = get_global_id(0)
    ti = get_local_id(0)
    tmp::Float32 = 0.0
    shared_mem[ti] = 0.0
    if i <= SIZE
        tmp = @inbounds x[i] * y[i]
        shared_mem[ti] = tmp
        barrier()
    end
    if (ti <= 128)
        shared_mem[ti] += shared_mem[ti + 128]
    end
    barrier()
    if (ti <= 64)
        shared_mem[ti] += shared_mem[ti + 64]
    end
    barrier()
    if (ti <= 32)
        shared_mem[ti] += shared_mem[ti + 32]
    end
    barrier()
    if (ti <= 16)
        shared_mem[ti] += shared_mem[ti + 16]
    end
    barrier()
    if (ti <= 8)
        shared_mem[ti] += shared_mem[ti + 8]
    end
    barrier()
    if (ti <= 4)
        shared_mem[ti] += shared_mem[ti + 4]
    end
    barrier()
    if (ti <= 2)
        shared_mem[ti] += shared_mem[ti + 2]
    end
    barrier()
    if (ti == 1)
        shared_mem[ti] += shared_mem[ti + 1]
        ret[get_group_id(0)] = shared_mem[ti]
    end
    barrier()
    return nothing
end

function reduce_kernel(SIZE, red, ret)
    shared_mem = oneLocalArray(Float32, 256)
    i = get_global_id(0)
    ii = i
    tmp::Float32 = 0.0
    if SIZE > 256
        while ii <= SIZE
            tmp += @inbounds red[ii]
            ii += 256
        end
    else
        tmp = @inbounds red[i]
    end
    shared_mem[i] = tmp
    barrier()
    if (i <= 128)
        shared_mem[i] += shared_mem[i + 128]
    end
    barrier()
    if (i <= 64)
        shared_mem[i] += shared_mem[i + 64]
    end
    barrier()
    if (i <= 32)
        shared_mem[i] += shared_mem[i + 32]
    end
    barrier()
    if (i <= 16)
        shared_mem[i] += shared_mem[i + 16]
    end
    barrier()
    if (i <= 8)
        shared_mem[i] += shared_mem[i + 8]
    end
    barrier()
    if (i <= 4)
        shared_mem[i] += shared_mem[i + 4]
    end
    barrier()
    if (i <= 2)
        shared_mem[i] += shared_mem[i + 2]
    end
    barrier()
    if (i == 1)
        shared_mem[i] += shared_mem[i + 1]
        ret[1] = shared_mem[1]
    end
    return nothing
end

function dot_oneapi(SIZE, x, y)
    numItems = 256
    items = min(SIZE, numItems)
    groups = ceil(Int, SIZE / items)
    ret = oneAPI.zeros(Float32, groups)
    rret = oneAPI.zeros(Float32, 1)
    oneAPI.@sync @oneapi items=items groups=groups dot_oneapi_kernel(
        SIZE, ret, x, y)
    oneAPI.@sync @oneapi items=items groups=1 reduce_kernel(SIZE, ret, rret)
    return rret
end

SIZE = 512
x = ones(Float32, SIZE)
y = ones(Float32, SIZE)
dx = oneArray(x)
dy = oneArray(y)
@time begin
    res = dot_oneapi(SIZE, dx, dy)
end

#-------------------------JACC

function dot(i, x, y)
    return @inbounds x[i] * y[i]
end

SIZE = 100000000
x = ones(SIZE)
y = ones(SIZE)
jx = JACC.Array(x)
jy = JACC.Array(y)
@time begin
    res = JACC.parallel_reduce(SIZE, dot, jx, jy)
end

function dot(i, x, y)
    return @inbounds x[i] * y[i]
end

SIZE = 512
x = ones(Float32, SIZE)
y = ones(Float32, SIZE)
jx = JACC.Array(x)
jy = JACC.Array(y)
@time begin
    res = JACC.parallel_reduce(SIZE, dot, jx, jy)
end
