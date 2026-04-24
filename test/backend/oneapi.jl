import oneAPI

@testset "TestBackend" begin
    @test JACC.backend == "oneapi"
    @test JACC.default_backend() == JACC.get_backend(JACC.Backend.oneapi)
end

@testset "array_types" begin
    using oneAPI
    import oneAPI.oneL0: DeviceBuffer
    N = 10

    #zeros
    x = JACC.zeros(N)
    @test typeof(x) == oneVector{FloatType, DeviceBuffer}
    @test eltype(x) == FloatType
    y = JACC.zeros(Int32, N)
    @test typeof(y) == oneVector{Int32, DeviceBuffer}
    @test eltype(y) == Int32

    # ones
    x = JACC.ones(N)
    @test typeof(x) == oneVector{FloatType, DeviceBuffer}
    @test eltype(x) == FloatType
    y = JACC.ones(Int32, N)
    @test typeof(y) == oneVector{Int32, DeviceBuffer}
    @test eltype(y) == Int32

    # fill
    x = JACC.fill(10.0, N)
    @test typeof(x) == oneVector{Float64, DeviceBuffer}
    y = JACC.fill(10, (N,))
    @test typeof(y) == oneVector{Int, DeviceBuffer}
    x2 = JACC.fill(10.0, N, N)
    @test typeof(x2) == oneMatrix{Float64, DeviceBuffer}
    y2 = JACC.fill(10, (N, N))
    @test typeof(y2) == oneMatrix{Int, DeviceBuffer}
    x3 = JACC.fill(10.0, N, N, N)
    @test typeof(x3) == oneArray{Float64, 3, DeviceBuffer}
    y3 = JACC.fill(10, (N, N, N))
    @test typeof(y3) == oneArray{Int, 3, DeviceBuffer}

    # array
    x = JACC.array(N)
    @test typeof(x) == oneVector{Float64, DeviceBuffer}
    x = JACC.array(Float32, N)
    @test typeof(x) == oneVector{Float32, DeviceBuffer}
    a = JACC.array(5, 4)
    b = JACC.array((5, 4))
    @test typeof(a) == oneMatrix{Float64, DeviceBuffer}
    @test typeof(b) == oneMatrix{Float64, DeviceBuffer}
    x = JACC.array(; type = Int, dims = 10)
    @test typeof(x) == oneVector{Int, DeviceBuffer}
    x = JACC.array(; type = Complex{Float32}, dims = (5, 5, 5))
    @test typeof(x) == oneArray{Complex{Float32}, 3, DeviceBuffer}
end

@testset "stream" begin
    using oneAPI, oneAPI.oneL0
    sd1 = JACC.default_stream()
    @test typeof(sd1) == ZeCommandQueue
    sd2 = JACC.default_stream()
    @test sd2 === sd1
    s1 = JACC.create_stream()
    @test typeof(s1) == ZeCommandQueue
    @test s1 != sd1
    s2 = JACC.create_stream()
    @test s2 != s1
end

include("preferences.jl")

@testset "preferences" begin
    test_preferences(:oneAPI)
end
