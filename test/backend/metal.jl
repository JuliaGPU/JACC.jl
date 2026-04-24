import Metal

@testset "TestBackend" begin
    @test JACC.backend == "metal"
    @test JACC.default_backend() == JACC.get_backend(JACC.Backend.metal)
end

@testset "zeros_type" begin
    using Metal
    N = 10
    x = JACC.zeros(Float32, N)
    @test typeof(x) == MtlArray{Float32, 1, Metal.PrivateStorage}
    @test eltype(x) == Float32
end

@testset "ones_type" begin
    using Metal
    N = 10
    x = JACC.ones(Float32, N)
    @test typeof(x) == MtlArray{Float32, 1, Metal.PrivateStorage}
    @test eltype(x) == Float32
end

include("preferences.jl")

@testset "preferences" begin
    test_preferences(:Metal)
end
