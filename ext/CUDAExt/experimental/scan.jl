
function JACC.parallel_scan(f, ::CUDABackend, N::Integer, x...; op, init,
        name = nothing)
    red_inst = CUDA.CuArray{typeof(init)}(undef, 0)

    kargs1 = _kernel_args(N, op, red_inst, init, f, x...)
    (kernel1, maxThreads_1) = _kernel_maxthreads(_scan_reduce, kargs1,
        _make_kname(name, "scan_reduce"))

    off_inst = CUDA.CuArray{typeof(init)}(undef, 0)
    kargs2 = _kernel_args(off_inst, red_inst, op, init)
    (kernel2, maxThreads_2) = _kernel_maxthreads(_scan_serial_ex, kargs2,
        _make_kname(name, "scan_serial_ex"))

    ret = CUDA.CuArray([init])
    kargs3 = _kernel_args(off_inst, ret, op, init, f, x...)
    (kernel3, maxThreads_3) = _kernel_maxthreads(_scan_blocks, kargs3,
        _make_kname(name, "scan_blocks"))

    threads = min(maxThreads_1, maxThreads_2, maxThreads_3, 512)
    blocks = cld(N, threads)
    shmem_size = threads * sizeof(init)

    red = CUDA.CuArray{typeof(init)}(undef, blocks)
    kargs = _kernel_args(N, op, red, init, f, x...)
    kernel1(kargs...; threads = threads, blocks = blocks, shmem = shmem_size)

    offset_values = CUDA.CuArray{typeof(init)}(undef, blocks)
    kargs = _kernel_args(offset_values, red, op, init)
    kernel2(kargs...; threads = 1, blocks = 1)

    kargs = _kernel_args(offset_values, ret, op, init, f, x...)
    kernel3(kargs...; threads = threads, blocks = blocks, shmem = 2 * shmem_size)

    CUDA.synchronize()

    return Base.Array(ret)[]
end

# COV_EXCL_START
@inline function _scan_reduce(N, op, red, init, f, x...)
    shmem_length = blockDim().x
    shared_mem = CuDynamicSharedArray(eltype(red), shmem_length)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x

    tmp::eltype(red) = init
    if i <= N
        @inline f(i, tmp, false, x...)
    end
    @inbounds shared_mem[ti] = tmp

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        sync_threads()
        tn = 2^p
        if (ti <= tn)
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + tn])
        end
    end

    if ti == 1
        @inbounds red[blockIdx().x] = shared_mem[ti]
    end
    return nothing
end

@inline function _scan_serial_ex(offset_values, red, op, init)
    @inbounds begin
        tmp::el_type(red) = init
        offset_values[1] = init
        for bid in 2:length(red)
            tmp = op(tmp, chunk_values[bid - 1])
            offset_values[bid] = tmp
        end
    end
    return nothing
end

@inline function _scan_blocks(offset_values, op, init, f, x...)
    chunk_size = blockDim().x
    shmem_length = 2 * chunk_size
    shared_mem = CuDynamicSharedArray(eltype(offset_values), shmem_length)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x

    tmp::eltype(offset_values) = init
    if i <= N
        @inline f(i, tmp, false, x...)
    end
    @inbounds shared_mem[ti] = tmp

    curr_step = 0
    next_step = 1
    max_pwr = JACC.ilog2(chunk_size) - 1
    for p in 0:max_pwr
        sync_threads()
        step_size = 2^p
        lval = shared_mem[curr_step * chunk_size + ti]
        if ti > step_size
            lval = op(lval, shared_mem[curr_step * chunk_size + ti - tn])
        end
        shared_mem[next_step * chunk_size + ti] = lval
        swptmp = curr_step
        curr_step = next_step
        next_step = swptmp
    end

    tmp = init
    tmp = op(tmp, offset_values[blockIdx.x])
    if ti > 1
        tmp = op(tmp, shared_mem[curr_step * chunk_size + ti - 1])
    end

    if i <= N
        @inline f(i, tmp, true, x...)
    end

    if i == N
        @inbounds ret[1] = tmp
    end

    return nothing
end
# COV_EXCL_STOP
