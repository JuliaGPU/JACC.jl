using oneAPIExt: _make_kernel, _make_kname

function JACC.Experimental.parallel_scan(f, ::oneAPIBackend, N::Integer, x...;
        op, init, name = nothing)
    red_inst = oneAPI.oneArray{typeof(init)}(undef, 0)

    kernel1 = _make_kernel(_make_kname(name, "scan_reduce"), _scan_reduce,
        Val(256), N, op, red_inst, init, f, x...)

    off_inst = oneAPI.oneArray{typeof(init)}(undef, 0)
    kernel2 = _make_kernel(_make_kname(name, "scan_serial_ex"), _scan_serial_ex,
        off_inst, red_inst, op, init)

    ret = oneAPI.oneArray([init])
    kernel3 = _make_kernel(_make_kname(name, "scan_blocks"), _scan_blocks,
        Val(256), off_inst, ret, N, op, init, f, x...)

    items = 256
    groups = cld(N, items)

    red = oneAPI.oneArray{typeof(init)}(undef, groups)
    kernel1(Val(items), N, op, red, init, f, x...; items = items, groups = groups)

    offset_values = oneAPI.oneArray{typeof(init)}(undef, groups)
    kernel2(offset_values, red, op, init; items = 1, groups = 1)

    kernel3(Val(256), offset_values, ret, N, op, init, f, x...; items = items, groups = groups)

    oneAPI.synchronize()

    return Base.Array(ret)[]
end

# COV_EXCL_START
@inline function _scan_reduce(
        ::Val{chunk_size}, N, op, red, init, f, x...) where {chunk_size}
    shared_mem = oneLocalArray(eltype(red), chunk_size)
    i = get_global_id()
    ti = get_local_id()

    tmp::Ref{eltype(red)} = Ref(init)
    if i <= N
        @inline f(i, tmp, false, x...)
    end
    @inbounds shared_mem[ti] = tmp[]

    t_size = chunk_size
    step_size = chunk_size >> 1
    while step_size > 0
        barrier(oneAPI.LOCAL_MEM_FENCE)
        if ti <= step_size
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + step_size])
        end
        barrier(oneAPI.LOCAL_MEM_FENCE)
        check_odd = t_size & 1
        if check_odd == 1
            if ti == 1
                @inbounds shared_mem[1] = op(shared_mem[1], shared_mem[t_size])
            end
        end
        t_size = step_size
        step_size >>= 1
    end

    barrier(oneAPI.LOCAL_MEM_FENCE)

    if ti == 1
        @inbounds red[get_group_id()] = shared_mem[1]
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
    shared_mem = oneLocalArray(eltype(offset_values), 2 * chunk_size)
    i = get_global_id()
    ti = get_local_id()

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
        barrier(oneAPI.LOCAL_MEM_FENCE)
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
    barrier(oneAPI.LOCAL_MEM_FENCE)

    tmp = init
    @inbounds tmp = op(tmp, offset_values[get_group_id()])
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
