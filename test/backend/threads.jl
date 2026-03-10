
@testset "TestBackend" begin
    @test JACC.backend == "threads"
    @test JACC.default_backend() == JACC.get_backend(JACC.Backend.threads)
end

@testset "array_types" begin
    N = 10

    # zeros
    x = JACC.zeros(Float32, N)
    @test typeof(x) == Vector{Float32}
    @test eltype(x) == Float32
    y = JACC.zeros(Int32, N)
    @test typeof(y) == Vector{Int32}
    @test eltype(y) == Int32

    # ones
    x = JACC.ones(Float64, N)
    @test typeof(x) == Vector{Float64}
    @test eltype(x) == Float64
    y = JACC.ones(Int32, N)
    @test typeof(y) == Vector{Int32}
    @test eltype(y) == Int32

    # fill
    x = JACC.fill(10.0, N)
    @test typeof(x) == Vector{Float64}
    y = JACC.fill(10, (N,))
    @test typeof(y) == Vector{Int}
    x2 = JACC.fill(10.0, N, N)
    @test typeof(x2) == Matrix{Float64}
    y2 = JACC.fill(10, (N, N))
    @test typeof(y2) == Matrix{Int}
    x3 = JACC.fill(10.0, N, N, N)
    @test typeof(x3) == Array{Float64, 3}
    y3 = JACC.fill(10, (N, N, N))
    @test typeof(y3) == Array{Int, 3}

    # array
    x = JACC.array(N)
    @test typeof(x) == Vector{Float64}
    x = JACC.array(Float32, N)
    @test typeof(x) == Vector{Float32}
    a = JACC.array(5, 4)
    b = JACC.array((5, 4))
    @test typeof(a) == Matrix{Float64}
    @test typeof(b) == Matrix{Float64}
    x = JACC.array(; type = Int, dims = 10)
    @test typeof(x) == Vector{Int}
    x = JACC.array(; type = Complex{Float32}, dims = (5, 5, 5))
    @test typeof(x) == Array{Complex{Float32}, 3}
end
