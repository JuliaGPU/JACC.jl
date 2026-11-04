function _construct_call(api_func, valid_keys, args...)
    expr = args[end]

    kwargs = map(args[1:(end - 1)]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    have_range = false
    for kwarg in kwargs
        key::Symbol, val = kwarg.args

        if !(key in valid_keys)
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end

        if key === :range
            have_range = true
        end
    end

    if Meta.isexpr(expr, :call)
        func = expr.args[1]
        args = expr.args[2:end]
        return quote
            $api_func(; f = $func, args = ($(args...),), $(kwargs...))
        end
    elseif Meta.isidentifier(expr)
        if !have_range
            kwargs = (kwargs..., :(range = JACC.array_size($expr)))
        end
        return quote
            $api_func(; f = JACC.elem_access, args = ($(expr),), $(kwargs...))
        end
    else
        throw(ArgumentError("last argument should be a kernel function call or an array"))
    end
end

macro parallel_for(args...)
    esc(_construct_call(JACC.parallel_for,
        [:range, :name, :threads, :blocks, :shmem_size, :stream, :sync], args...))
end

macro parallel_reduce(args...)
    esc(_construct_call(JACC.parallel_reduce,
        [:range, :name, :type, :op, :init, :stream, :sync], args...))
end
