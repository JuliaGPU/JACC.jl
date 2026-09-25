using Pkg

script_dir = dirname(@__FILE__)

jacc_dir = ARGS[1]

backend = ARGS[2]
base_cmd = `$(Base.julia_cmd()) --project=$script_dir`
test_file = joinpath(dirname(script_dir), "test.jl")

ret = 0

code = ` 'using Pkg; Pkg.develop(path=ARGS[1]; io=devnull)' `
res = run(`$base_cmd -e $code $jacc_dir`)
ret += res.exitcode

code = ` 'using JACC; redirect_stdout(devnull); JACC.set_backend(ARGS[1])' `
res = run(`$base_cmd -e $code $backend`)
ret += res.exitcode

res = run(`$base_cmd $test_file`)
ret += res.exitcode

exit(ret)
