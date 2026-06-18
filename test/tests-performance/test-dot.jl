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

SIZE = 1000000
x = ones(SIZE)
y = ones(SIZE)
@time begin
    dot_threads(SIZE, x, y)
end

#-------------------------CUDA

function dot_cuda_kernel(SIZE, ret, x, y)
    shared_mem = @cuDynamicSharedMem(Float64, 512)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ii = i
    tmp::Float64 = 0.0
    if SIZE > 512
        while ii <= SIZE
            tmp += @inbounds x[ii] * y[ii]
            ii += 512
        end
    else
        tmp = @inbounds x[i] * y[i]
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
    ret = CUDA.zeros(1)
    CUDA.@sync @cuda threads=threads blocks=1 shmem=512*sizeof(Float64) dot_cuda_kernel(
        SIZE, ret, x, y)
    return ret[1]
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
    shared_mem = @ROCDynamicLocalArray(Float64, 1024)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ii = i
    tmp::Float64 = 0.0
    if SIZE > 1024
        while ii <= SIZE
            tmp += @inbounds x[ii] * y[ii]
            ii += 1024
        end
    else
        tmp = @inbounds x[i] * y[i]
    end
    shared_mem[i] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 512)
        shared_mem[i] += shared_mem[i + 512]
    end
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
    maxPossibleThreads = 1024
    threads = min(SIZE, maxPossibleThreads)
    ret = AMDGPU.zeros(1)
    @roc groupsize=threads gridsize=threads localmem=1024*sizeof(Float64) dot_amdgpu_kernel(
        SIZE, ret, x, y)
    return ret[1]
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
    i = get_global_id()
    ii = i
    tmp::Float32 = 0.0
    if SIZE > 256
        while ii <= SIZE
            tmp += @inbounds x[ii] * y[ii]
            ii += 256
        end
    else
        tmp = @inbounds x[i] * y[i]
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
        ret[1] = shared_mem[i]
    end
    return nothing
end

function dot_oneapi(SIZE, x, y)
    numItems = 256
    items = min(SIZE, numItems)
    ret = oneAPI.zeros(Float32, 1)
    oneAPI.@sync @oneapi items=items groups=1 dot_oneapi_kernel(SIZE, ret, x, y)
    #oneAPI.@sync @oneapi items = items groups = 1 dot_oneapi_kernel(SIZE, x, y)
    return ret[1]
end

SIZE = 100000000
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

SIZE = 100000000
x = ones(Float32, SIZE)
y = ones(Float32, SIZE)
jx = JACC.Array(x)
jy = JACC.Array(y)
@time begin
    res = JACC.parallel_reduce(SIZE, dot, jx, jy)
end
