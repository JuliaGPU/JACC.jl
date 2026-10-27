
function test_consumer(pkg_code, arg, backend)
    tmp_dir = mktempdir(; prefix = "jacc_")
    cwd = pwd()
    cd(tmp_dir)

    base_cmd = `$(Base.julia_cmd()) --project=$tmp_dir`

    ret = 0

    verbose = parse(Bool, get(ENV, "JACC_CONSUMER_TEST_VERBOSE", "false"))
    try
        # Handle verbose output
        backend_code = "using JACC; JACC.set_backend(ARGS[1])"
        test_args = ``
        if !verbose
            pkg_code = "redirect_stderr(devnull); $pkg_code"
            backend_code = "redirect_stderr(devnull); $backend_code"
            test_args = `-q`
        end

        # Add JACC
        pkg_cmd = `$base_cmd -e $pkg_code $arg`
        if verbose
            @show pwd()
            println("Run: ", pkg_cmd)
        end
        ret += run(pkg_cmd).exitcode

        # Set JACC backend
        backend_cmd = `$base_cmd -e $backend_code $backend`
        if verbose
            @show pwd()
            println("Run: ", backend_cmd)
        end
        res = run(backend_cmd)
        ret += res.exitcode

        # Run some JACC code
        test_file = joinpath(dirname(@__FILE__), "test.jl")
        test_cmd = `$base_cmd $test_file $test_args`
        if verbose
            @show pwd()
            println("Run: ", test_cmd)
        end
        res = run(test_cmd)
        ret += res.exitcode
    catch e
        if verbose
            rethrow()
        end
        ret += 1
    finally
        cd(cwd)
    end

    return ret
end
