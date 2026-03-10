import CUDA

@testset "TestBackend" begin
    @test JACC.backend == "cuda"
    @test JACC.default_backend() == JACC.get_backend(JACC.Backend.cuda)
end

@testset "array_types" begin
    using CUDA
    import CUDA: CuArray, DeviceMemory
    N = 10

    # zeros
    x = JACC.zeros(Float64, N)
    @test typeof(x) == CuArray{Float64, 1, DeviceMemory}
    @test eltype(x) == Float64
    y = JACC.zeros(Int32, N)
    @test typeof(y) == CuArray{Int32, 1, DeviceMemory}
    @test eltype(y) == Int32

    # ones
    x = JACC.ones(Float64, N)
    @test typeof(x) == CuArray{Float64, 1, DeviceMemory}
    @test eltype(x) == Float64
    y = JACC.ones(Int32, N)
    @test typeof(y) == CuArray{Int32, 1, DeviceMemory}
    @test eltype(y) == Int32

    # fill
    x = JACC.fill(10.0, N)
    @test typeof(x) == CuArray{Float64, 1, DeviceMemory}
    y = JACC.fill(10, (N,))
    @test typeof(y) == CuArray{Int, 1, DeviceMemory}
    x2 = JACC.fill(10.0, N, N)
    @test typeof(x2) == CuArray{Float64, 2, DeviceMemory}
    y2 = JACC.fill(10, (N, N))
    @test typeof(y2) == CuArray{Int, 2, DeviceMemory}
    x3 = JACC.fill(10.0, N, N, N)
    @test typeof(x3) == CuArray{Float64, 3, DeviceMemory}
    y3 = JACC.fill(10, (N, N, N))
    @test typeof(y3) == CuArray{Int, 3, DeviceMemory}

    # array
    x = JACC.array(N)
    @test typeof(x) == CuVector{Float64, DeviceMemory}
    x = JACC.array(Float32, N)
    @test typeof(x) == CuVector{Float32, DeviceMemory}
    a = JACC.array(5, 4)
    b = JACC.array((5, 4))
    @test typeof(a) == CuMatrix{Float64, DeviceMemory}
    @test typeof(b) == CuMatrix{Float64, DeviceMemory}
    x = JACC.array(; type = Int, dims = 10)
    @test typeof(x) == CuVector{Int, DeviceMemory}
    x = JACC.array(; type = Complex{Float32}, dims = (5, 5, 5))
    @test typeof(x) == CuArray{Complex{Float32}, 3, DeviceMemory}
end

@testset "stream" begin
    using CUDA
    sd1 = JACC.default_stream()
    @test typeof(sd1) == CuStream
    sd2 = JACC.default_stream()
    @test sd2 === sd1
    s1 = JACC.create_stream()
    @test typeof(s1) == CuStream
    @test s1 != sd1
    s2 = JACC.create_stream()
    @test s2 != s1
end
