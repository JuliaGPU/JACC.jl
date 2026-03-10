import AMDGPU

@testset "TestBackend" begin
    @test JACC.backend == "amdgpu"
    @test JACC.default_backend() == JACC.get_backend(JACC.Backend.amdgpu)
end

@testset "array_types" begin
    using AMDGPU
    import AMDGPU.Runtime.Mem.HIPBuffer
    N = 10

    # zeros
    x = JACC.zeros(Float64, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, HIPBuffer}
    @test eltype(x) == Float64
    x = JACC.zeros(Int32, N)
    @test typeof(x) == AMDGPU.ROCArray{Int32, 1, HIPBuffer}
    @test eltype(x) == Int32

    # ones
    x = JACC.ones(Float64, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, HIPBuffer}
    @test eltype(x) == Float64
    x = JACC.ones(Int32, N)
    @test typeof(x) == AMDGPU.ROCArray{Int32, 1, HIPBuffer}
    @test eltype(x) == Int32

    # fill
    x = JACC.fill(10.0, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, HIPBuffer}
    y = JACC.fill(10, (N,))
    @test typeof(y) == AMDGPU.ROCArray{Int, 1, HIPBuffer}
    x2 = JACC.fill(10.0, N, N)
    @test typeof(x2) ==
          AMDGPU.ROCArray{Float64, 2, HIPBuffer}
    y2 = JACC.fill(10, (N, N))
    @test typeof(y2) == AMDGPU.ROCArray{Int, 2, HIPBuffer}
    x3 = JACC.fill(10.0, N, N, N)
    @test typeof(x3) ==
          AMDGPU.ROCArray{Float64, 3, HIPBuffer}
    y3 = JACC.fill(10, (N, N, N))
    @test typeof(y3) == AMDGPU.ROCArray{Int, 3, HIPBuffer}

    # array
    x = JACC.array(N)
    @test typeof(x) == ROCVector{Float64, HIPBuffer}
    x = JACC.array(Float32, N)
    @test typeof(x) == ROCVector{Float32, HIPBuffer}
    a = JACC.array(5, 4)
    b = JACC.array((5, 4))
    @test typeof(a) == ROCMatrix{Float64, HIPBuffer}
    @test typeof(b) == ROCMatrix{Float64, HIPBuffer}
    x = JACC.array(; type = Int, dims = 10)
    @test typeof(x) == ROCVector{Int, HIPBuffer}
    x = JACC.array(; type = Complex{Float32}, dims = (5, 5, 5))
    @test typeof(x) == ROCArray{Complex{Float32}, 3, HIPBuffer}
end

@testset "stream" begin
    using AMDGPU
    sd1 = JACC.default_stream()
    @test typeof(sd1) == HIPStream
    sd2 = JACC.default_stream()
    @test sd2 === sd1
    s1 = JACC.create_stream()
    @test typeof(s1) == HIPStream
    @test s1 != sd1
    s2 = JACC.create_stream()
    @test s2 != s1
end
