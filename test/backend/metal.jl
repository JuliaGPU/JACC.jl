import Metal

@testset "TestBackend" begin
    @test JACC.backend == "metal"
    @test JACC.default_backend() == JACC.get_backend(JACC.Backend.metal)
end

@testset "zeros_type" begin
    using Metal
    N = 10
    x = JACC.zeros(Float32, N)
    @test typeof(x) == MtlArray{Float32, 1, Metal.DefaultStorageMode}
    @test eltype(x) == Float32
end

@testset "ones_type" begin
    using Metal
    N = 10
    x = JACC.ones(Float32, N)
    @test typeof(x) == MtlArray{Float32, 1, Metal.DefaultStorageMode}
    @test eltype(x) == Float32
end

@testset "array_storage_preference" begin
    using Metal
    using Suppressor
    N = 10
    h = ones(Float32, N)
    x = JACC.array(h)
    @test typeof(x) == MtlArray{Float32, 1, Metal.DefaultStorageMode}
    @test JACC.array(x) === x
    @test JACC.array(JACC.default_backend_type(), x) === x
    @test JACC.array(JACC.default_backend_type(), h) isa
          MtlArray{Float32, 1, Metal.DefaultStorageMode}
    try
        @suppress JACC.set_backend("Metal"; storage = :shared)
        preferences = load_preference(JACC, "extension_preferences")
        @test preferences["metal"]["storage"] == "shared"
        @test JACC.Preferences.Backend._EXT_PREFS[]["metal"] ==
              Dict(:storage => :shared)
        xs = JACC.array(h)
        @test typeof(xs) == MtlArray{Float32, 1, Metal.SharedStorage}
        @test Metal.is_shared(xs)
        @test Array(xs) == h
        @test JACC.array(xs) === xs
        @test JACC.array(JACC.default_backend_type(), xs) === xs
        @test JACC.array(JACC.default_backend_type(), h) isa
              MtlArray{Float32, 1, Metal.SharedStorage}
        h2 = ones(Float32, N, N)
        xs2 = JACC.array(h2)
        @test typeof(xs2) == MtlArray{Float32, 2, Metal.SharedStorage}
    finally
        @suppress JACC.set_backend("Metal"; storage = :private)
    end
    @test JACC.array(h) isa MtlArray{Float32, 1, Metal.PrivateStorage}
    @suppress JACC.set_backend("Metal"; storage = :bogus)
    @test_throws ArgumentError JACC.array(h)
    @suppress JACC.set_backend("Metal"; storage = 1)
    @test_throws ArgumentError JACC.array(h)
    @suppress JACC.set_backend("Metal"; storage = :private)
end

struct Vec16
    v::NTuple{16, Float32}
end
Base.:+(a::Vec16, b::Vec16) = Vec16(a.v .+ b.v)

# 64 temporaries per item keep the reduce kernels under 512 threads (384 on M1)
struct WideItem end
@inline function (::WideItem)(i, x)
    s = @inbounds x[i]
    t = ntuple(k -> sin(s * k) * exp(-s / k), Val(64))
    return Vec16(ntuple(
        k -> sum(ntuple(j -> t[j] * t[mod1(j + k, 64)], Val(64))), Val(16)))
end

@testset "reduce_below_512_threads" begin
    N = 100_000
    xh = Float32.(1:N) ./ N
    x = JACC.array(xh)
    init = Vec16(ntuple(_ -> 0.0f0, 16))
    expected = sum(i -> WideItem()(i, xh), 1:N)
    res = JACC.parallel_reduce(N, WideItem(), x; op = +, init = init)
    @test all(isapprox.(res.v, expected.v; rtol = 1e-4))
    reducer = JACC.reducer(; range = N, op = +, init = init)
    reducer(WideItem(), x)
    @test all(isapprox.(JACC.get_result(reducer).v, expected.v; rtol = 1e-4))
end

include("preferences.jl")

@testset "preferences" begin
    test_preferences(:Metal)
end
