import JACC
import LibGit2

# if get(ENV, "JACC_INCLUDE_CONSUMER_TESTS", "false") == "true"

    @testset "consumer_local" begin
        script_dir = joinpath("consumer", "local")
        script = joinpath(script_dir, "script.jl")
        params = `$(pkgdir(JACC)) $(JACC.backend)`
        ret = run(`$(Base.julia_cmd()) $script $params`).exitcode
        rm(joinpath(script_dir, "Manifest.toml"))
        rm(joinpath(script_dir, "Project.toml"))
        @test ret == 0
    end

    @testset "consumer_remote" begin
        script_dir = joinpath("consumer", "remote")
        script = joinpath(script_dir, "script.jl")
        params = `$(LibGit2.head(pkgdir(JACC))) $(JACC.backend)`
        ret = run(`$(Base.julia_cmd()) $script $params`).exitcode
        rm(joinpath(script_dir, "Manifest.toml"))
        rm(joinpath(script_dir, "Project.toml"))
        @test ret == 0
    end

# end
