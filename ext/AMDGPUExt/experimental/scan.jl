using AMDGPUExt: _make_kname, _kernel_args, _kernel_maxthreads, AMDGPUTag

function JACC.Experimental.parallel_scan(
        f, ::AMDGPUTag, N::Integer, x...; op, init, name = nothing)
    red_inst = AMDGPU.ROCArray{typeof(init)}(undef, 0)

    kargs1 = _kernel_args(N, op, red_inst, init, f, x...)
    (kernel1, maxThreads_1) = _kernel_maxthreads(_scan_reduce, kargs1,
        _make_kname(name, "scan_reduce"))

    off_inst = AMDGPU.ROCArray{typeof(init)}(undef, 0)
    kargs2 = _kernel_args(off_inst, red_inst, op, init)
    (kernel2, maxThreads_2) = _kernel_maxthreads(_scan_serial_ex, kargs2,
        _make_kname(name, "scan_serial_ex"))

    ret = AMDGPU.ROCArray([init])
    kargs3 = _kernel_args(off_inst, ret, N, op, init, f, x...)
    (kernel3, maxThreads_3) = _kernel_maxthreads(_scan_blocks, kargs3,
        _make_kname(name, "scan_blocks"))

    threads = min(maxThreads_1, maxThreads_2, maxThreads_3, 512)
    blocks = cld(N, threads)
    shmem_size = threads * sizeof(init)

    red = AMDGPU.ROCArray{typeof(init)}(undef, blocks)
    kargs = _kernel_args(N, op, red, init, f, x...)
    kernel1(kargs...; groupsize = threads, gridsize = blocks, shmem = shmem_size)

    offset_values = AMDGPU.ROCArray{typeof(init)}(undef, blocks)
    kargs = _kernel_args(offset_values, red, op, init)
    kernel2(kargs...; groupsize = 1, gridsize = 1)

    kargs = _kernel_args(offset_values, ret, N, op, init, f, x...)
    kernel3(kargs...; groupsize = threads, gridsize = blocks, shmem = 2 * shmem_size)

    AMDGPU.synchronize()

    return Base.Array(ret)[]
end

# COV_EXCL_START
@inline function _scan_reduce(N, op, red, init, f, x...)
    chunk_size = workgroupDim().x
    shared_mem = @ROCDynamicLocalArray(eltype(red), chunk_size, false)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ti = workitemIdx().x

    tmp::Ref{eltype(red)} = Ref(init)
    if i <= N
        @inline f(i, tmp, false, x...)
    end
    @inbounds shared_mem[ti] = tmp[]

    t_size = chunk_size
    step_size = chunk_size >> 1
    while step_size > 0
        AMDGPU.sync_workgroup()
        if ti <= step_size
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + step_size])
        end
        AMDGPU.sync_workgroup()
        check_odd = t_size & 1
        if check_odd == 1
            if ti == 1
                @inbounds shared_mem[1] = op(shared_mem[1], shared_mem[t_size])
            end
        end
        t_size = step_size
        step_size >>= 1
    end

    AMDGPU.sync_workgroup()

    if ti == 1
        @inbounds red[workgroupIdx().x] = shared_mem[ti]
    end
    return nothing
end

@inline function _scan_serial_ex(offset_values, red, op, init)
    @inbounds begin
        tmp::eltype(red) = init
        offset_values[1] = init
        for bid in 2:length(red)
            tmp = op(tmp, red[bid - 1])
            offset_values[bid] = tmp
        end
    end
    return nothing
end

@inline function _scan_blocks(offset_values, ret, N, op, init, f, x...)
    chunk_size = workgroupDim().x
    shared_mem = @ROCDynamicLocalArray(eltype(offset_values), 2 * chunk_size, false)
    block_id = workgroupIdx().x
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ti = workitemIdx().x

    tmp::eltype(offset_values) = init
    if i <= N
        ref = Ref(tmp)
        @inline f(i, ref, false, x...)
        tmp = ref[]
    end
    @inbounds shared_mem[ti] = tmp

    curr_step = 0
    next_step = 1
    max_pwr = JACC.ilog2(chunk_size) - 1
    for p in 0:max_pwr
        AMDGPU.sync_workgroup()
        tn = 2^p
        @inbounds lval = shared_mem[curr_step * chunk_size + ti]
        if ti > tn
            @inbounds lval = op(lval, shared_mem[curr_step * chunk_size + ti - tn])
        end
        @inbounds shared_mem[next_step * chunk_size + ti] = lval
        swptmp = curr_step
        curr_step = next_step
        next_step = swptmp
    end
    AMDGPU.sync_workgroup()

    tmp = init
    @inbounds tmp = op(tmp, offset_values[block_id])
    if ti > 1
        @inbounds tmp = op(tmp, shared_mem[curr_step * chunk_size + ti - 1])
    end

    if i <= N
        ref = Ref(tmp)
        @inline f(i, ref, true, x...)
        tmp = ref[]
    end

    if i == N
        @inbounds ret[1] = tmp
    end

    return nothing
end
# COV_EXCL_STOP
