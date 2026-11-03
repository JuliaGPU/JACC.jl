import JACC
import LibGit2

if parse(Bool, get(ENV, "JACC_INCLUDE_CONSUMER_TESTS", "true"))

    include(joinpath("consumer", "test_consumer.jl"))

    @testset "consumer_local" begin
        # pkg_code = ` 'using Pkg; Pkg.develop(path=ARGS[1])' `
        pkg_code = "using Pkg; Pkg.develop(path=ARGS[1])"
        ret = test_consumer(pkg_code, pkgdir(JACC), JACC.backend)
        @test ret == 0
    end

    @testset "consumer_remote" begin
        pkg_code = "using Pkg; Pkg.add(name=\"JACC\", rev=ARGS[1])"
        ret = test_consumer(pkg_code, LibGit2.head(pkgdir(JACC)), JACC.backend)
        @test ret == 0
    end

end
