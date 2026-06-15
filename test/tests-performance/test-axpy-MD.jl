ds

function axpy_threads((SIZE, SIZE), alpha, x, y)
    Threads.@threads for j in 1:SIZE
        for i in 1:SIZE
            @inbounds x[i, j] = x[i, j] + alpha * y[i, j]
        end
    end
end

SIZE = 50
x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
alpha = 2.0
@time begin
    axpy_threads((SIZE, SIZE), alpha, x, y)
end

#-------------------------CUDA

function axpy_cuda_kernel(alpha, x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds x[i, j] = x[i, j] + alpha * y[i, j]
    return nothing
end

function axpy_cuda((M, N), alpha, x, y)
    maxPossibleThreads = 16
    Mthreads = min(M, maxPossibleThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nthreads = min(N, maxPossibleThreads)
    Nblocks = ceil(Int, N / Nthreads)
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) axpy_cuda_kernel(
        alpha, x, y)
end

SIZE = 50
x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
alpha = 2.0
dx = CuArray(x)
dy = CuArray(y)
@time begin
    axpy_cuda((SIZE, SIZE), alpha, dx, dy)
end

#-------------------------AMDGPU

function axpy_amdgpu_kernel(alpha, x, y)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    @inbounds x[i, j] = x[i, j] + alpha * y[i, j]
    return nothing
end

function axpy_amdgpu((M, N), alpha, x, y)
    maxPossibleThreads = 16
    Mthreads = min(M, maxPossibleThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nthreads = min(N, maxPossibleThreads)
    Nblocks = ceil(Int, N / Nthreads)
    @roc groupsize=(Mthreads, Nthreads) gridsize=(
        Mthreads*Mblocks, Nthreads*Nblocks) axpy_amdgpu_kernel(alpha, x, y)
end

SIZE = 50
x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
alpha = 2.0
dx = ROCArray(x)
dy = ROCArray(y)
@time begin
    axpy_amdgpu((SIZE, SIZE), alpha, dx, dy)
end

#-------------------------oneAPI

function axpy_oneapi_kernel(alpha::Float32, x, y)
    i = get_global_id(0)
    j = get_global_id(1)
    @inbounds x[i, j] = x[i, j] + alpha * y[i, j]
    return nothing
end

function axpy_oneapi((M, N), alpha::Float32, x, y)
    maxPossibleItems = 16
    Mitems = min(M, maxPossibleItems)
    Nitems = min(N, maxPossibleItems)
    Mgroups = ceil(Int, M / Mitems)
    Ngroups = ceil(Int, N / Nitems)
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(Mgroups, Ngroups) axpy_oneapi_kernel(
        alpha, x, y)
end

SIZE = 50
x = ones(Float32, SIZE, SIZE)
y = ones(Float32, SIZE, SIZE)
alpha::Float32 = 2.5
dx = oneArray(x)
dy = oneArray(y)
@time begin
    axpy_oneapi((SIZE, SIZE), alpha, dx, dy)
end

#-------------------------JACC

function axpy(i, j, alpha, x, y)
    @inbounds x[i, j] = x[i, j] + alpha * y[i, j]
end

SIZE = 5000
x = ones(SIZE, SIZE)
y = ones(SIZE, SIZE)
alpha = 2.0
jx = JACC.Array(x)
jy = JACC.Array(y)
@time begin
    JACC.parallel_for((SIZE, SIZE), axpy, alpha, jx, jy)
end

function axpy(i, j, alpha::Float32, x, y)
    @inbounds x[i, j] = x[i, j] + alpha * y[i, j]
end

SIZE = 50
x = ones(Float32, SIZE, SIZE)
y = ones(Float32, SIZE, SIZE)
alpha::Float32 = 2.0
jx = JACC.Array(x)
jy = JACC.Array(y)
@time begin
    JACC.parallel_for((SIZE, SIZE), axpy, alpha, jx, jy)
end
