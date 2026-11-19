using MetalExt: _kernel_maxthreads, _make_kname, MetalTag

function JACC.Experimental.parallel_scan(f, ::MetalTag, N::Integer, x...;
        op, init, name = nothing)
    red_inst = Metal.MtlArray{typeof(init)}(undef, 0)

    kargs1 = (Val(512), N, op, red_inst, init, f, x...)
    (kernel1, maxThreads_1) = _kernel_maxthreads(
        _scan_reduce, kargs1, _make_kname(name, "scan_reduce"))

    off_inst = Metal.MtlArray{typeof(init)}(undef, 0)
    kargs2 = (off_inst, red_inst, op, init)
    (kernel2, maxThreads_2) = _kernel_maxthreads(_scan_serial_ex, kargs2,
        _make_kname(name, "scan_serial_ex"))

    ret = Metal.MtlArray([init])
    kargs3 = (Val(512), off_inst, ret, N, op, init, f, x...)
    (kernel3, maxThreads_3) = _kernel_maxthreads(_scan_blocks, kargs3,
        _make_kname(name, "scan_blocks"))

    threads = min(maxThreads_1, maxThreads_2, maxThreads_3, 512)
    blocks = cld(N, threads)
    shmem_size = threads * sizeof(init)

    red = Metal.MtlArray{typeof(init)}(undef, blocks)
    kargs = (Val(threads), N, op, red, init, f, x...)
    kernel1(kargs...; threads = threads, groups = blocks)

    offset_values = Metal.MtlArray{typeof(init)}(undef, blocks)
    kargs = (offset_values, red, op, init)
    kernel2(kargs...; threads = 1, groups = 1)

    kargs = (Val(512), offset_values, ret, N, op, init, f, x...)
    kernel3(kargs...; threads = threads, groups = blocks)

    Metal.synchronize()

    return Base.Array(ret)[]
end

# COV_EXCL_START
@inline function _scan_reduce(
        ::Val{chunk_size}, N, op, red, init, f, x...) where {chunk_size}
    shared_mem = MtlThreadGroupArray(eltype(red), chunk_size)
    i = thread_position_in_grid().x
    ti = thread_position_in_threadgroup().x

    tmp::Ref{eltype(red)} = Ref(init)
    if i <= N
        @inline f(i, tmp, false, x...)
    end
    @inbounds shared_mem[ti] = tmp[]

    t_size = chunk_size
    step_size = chunk_size >> 1
    while step_size > 0
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        if ti <= step_size
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + step_size])
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        check_odd = t_size & 1
        if check_odd == 1
            if ti == 1
                @inbounds shared_mem[1] = op(shared_mem[1], shared_mem[t_size])
            end
        end
        t_size = step_size
        step_size >>= 1
    end

    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    if ti == 1
        @inbounds red[threadgroup_position_in_grid().x] = shared_mem[1]
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

@inline function _scan_blocks(
        ::Val{chunk_size}, offset_values, ret, N, op, init, f, x...) where {chunk_size}
    shared_mem = MtlThreadGroupArray(eltype(offset_values), 2 * chunk_size)
    i = thread_position_in_grid().x
    ti = thread_position_in_threadgroup().x

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
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
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
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    tmp = init
    @inbounds tmp = op(tmp, offset_values[threadgroup_position_in_grid().x])
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
