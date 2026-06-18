#-------------------------Threads  

function axpy_threads(SIZE, alpha, x, y)
    Threads.@threads for i in 1:SIZE
        @inbounds x[i] = x[i] + alpha * y[i]
    end
end

SIZE = 1000000
x = ones(SIZE)
y = ones(SIZE)
alpha = 2.0
@time begin
    axpy_threads(SIZE, alpha, x, y)
end

#-------------------------CUDA

function axpy_cuda_kernel(alpha, x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds x[i] = x[i] + alpha * y[i]
    return nothing
end

function axpy_cuda(SIZE, alpha, x, y)
    maxPossibleThreads = attribute(
        device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE / threads)
    CUDA.@sync @cuda threads=threads blocks=blocks axpy_cuda_kernel(alpha, x, y)
end

SIZE = 10000000
x = ones(SIZE)
y = ones(SIZE)
alpha = 2.0
dx = CuArray(x)
dy = CuArray(y)
@time begin
    axpy_cuda(SIZE, alpha, dx, dy)
end

#-------------------------AMDGPU

function axpy_amdgpu_kernel(alpha, x, y)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    @inbounds x[i] = x[i] + alpha * y[i]
    return nothing
end

function axpy_amdgpu(SIZE, alpha, x, y)
    maxPossibleThreads = 512
    threads = min(SIZE, maxPossibleThreads)
    blocks = ceil(Int, SIZE / threads)
    @roc groupsize=threads gridsize=threads*blocks axpy_amdgpu_kernel(
        alpha, x, y)
end

SIZE = 1000000
x = ones(SIZE)
y = ones(SIZE)
alpha = 2.0
dx = ROCArray(x)
dy = ROCArray(y)
@time begin
    axpy_amdgpu(SIZE, alpha, dx, dy)
end

#-------------------------oneAPI

function axpy_oneapi_kernel(alpha::Float32, x, y)
    i = get_global_id()
    @inbounds x[i] = x[i] + alpha * y[i]
    return nothing
end

function axpy_oneapi(SIZE, alpha::Float32, x, y)
    maxPossibleItems = 256
    items = min(SIZE, maxPossibleItems)
    groups = ceil(Int, SIZE / items)
    oneAPI.@sync @oneapi items=items groups=groups axpy_oneapi_kernel(
        alpha, x, y)
end

SIZE = 100000
x = ones(Float32, SIZE)
y = ones(Float32, SIZE)
alpha::Float32 = 2.0
dx = oneArray(x)
dy = oneArray(y)
@time begin
    axpy_oneapi(SIZE, alpha, dx, dy)
end

#-------------------------JACC

function axpy(i, alpha, x, y)
    if i <= length(x)
        @inbounds x[i] += alpha * y[i]
    end
end

SIZE = 100000000
x = ones(SIZE)
y = ones(SIZE)
alpha = 2.0
jx = JACC.Array(x)
jy = JACC.Array(y)
@time begin
    JACC.parallel_for(SIZE, axpy, alpha, jx, jy)
end

function axpy(i, alpha::Float32, x, y)
    if i <= length(x)
        @inbounds x[i] += alpha * y[i]
    end
end

SIZE = 100000000
x = ones(Float32, SIZE)
y = ones(Float32, SIZE)
falpha::Float32 = 2.0
jx = JACC.Array(x)
jy = JACC.Array(y)
@time begin
    JACC.parallel_for(SIZE, axpy, falpha, jx, jy)
end
