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

@testset "array_storage_preference" begin
    using AMDGPU
    import AMDGPU.Runtime.Mem: HIPBuffer, HostBuffer
    using Suppressor
    N = 10
    h = ones(Float32, N)
    x = JACC.array(h)
    @test typeof(x) == ROCVector{Float32, HIPBuffer}
    @test JACC.array(x) === x
    @test JACC.array(JACC.default_backend_type(), x) === x
    @test JACC.array(JACC.default_backend_type(), h) isa
          ROCVector{Float32, HIPBuffer}
    try
        @suppress JACC.set_backend("AMDGPU"; storage = :host)
        preferences = load_preference(JACC, "extension_preferences")
        @test preferences["amdgpu"]["storage"] == "host"
        @test JACC.Preferences.Backend._EXT_PREFS[]["amdgpu"] ==
              Dict(:storage => :host)
        xs = JACC.array(h)
        @test typeof(xs) == ROCVector{Float32, HostBuffer}
        @test Array(xs) == h
        @test JACC.array(xs) === xs
        @test JACC.array(JACC.default_backend_type(), xs) === xs
        @test JACC.array(JACC.default_backend_type(), h) isa
              ROCVector{Float32, HostBuffer}
        h2 = ones(Float32, N, N)
        xs2 = JACC.array(h2)
        @test typeof(xs2) == ROCMatrix{Float32, HostBuffer}
        # host-pinned arrays must be usable from kernels (zero-copy path)
        JACC.parallel_for(N, (i, a) -> (a[i] += 1f0), xs)
        @test Array(xs) == h .+ 1f0
    finally
        @suppress JACC.set_backend("AMDGPU"; storage = :device)
    end
    @test JACC.array(h) isa ROCVector{Float32, HIPBuffer}
    @suppress JACC.set_backend("AMDGPU"; storage = :bogus)
    @test_throws ArgumentError JACC.array(h)
    @suppress JACC.set_backend("AMDGPU"; storage = 1)
    @test_throws ArgumentError JACC.array(h)
    @suppress JACC.set_backend("AMDGPU"; storage = :device)
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

@testset "launch bound" begin
    # The default bound, 1024 threads, holds a wave64 kernel to 128 VGPRs; a
    # bound of 256 or fewer lifts it. The block chosen stays within the bound.
    a = JACC.zeros(Float64, 1000)
    spec = JACC.launch_spec(; maxthreads = 64)
    JACC.parallel_for(spec, 1000, a) do i, a
        @inbounds a[i] = i
    end
    @test Array(a) == 1.0:1000.0
    @test spec.threads <= 64
    spec = JACC.launch_spec(; maxthreads = 64)
    JACC.parallel_for(spec, (100, 100), (i, j, a) -> nothing, a)
    @test prod(spec.threads) <= 64
    spec = JACC.launch_spec(; maxthreads = 64, threads = 128)
    @test_throws ArgumentError JACC.parallel_for(
        spec, 1000, (i, a) -> nothing, a)
end

include("preferences.jl")

@testset "preferences" begin
    test_preferences(:AMDGPU)
end
