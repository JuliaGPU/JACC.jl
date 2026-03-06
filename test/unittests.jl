import LinearAlgebra
using ..JACCTestCommon: axpy, dot, seq_axpy, seq_dot

function rand_jacc(dims; type::Type{T} = FloatType) where {T}
    return JACC.to_device(round.(rand(T, dims) * 100))
end

@testset "array" begin
    # zeros
    N = 10
    x = JACC.zeros(N)
    @test eltype(x) == FloatType
    @test zeros(N)≈JACC.to_host(x) rtol=1e-5

    function add_one(i, x)
        @inbounds x[i] += 1
    end

    JACC.parallel_for(N, add_one, x)
    @test ones(N)≈JACC.to_host(x) rtol=1e-5

    # ones
    N = 10
    x = JACC.ones(N)
    @test eltype(x) == FloatType
    @test ones(N)≈JACC.to_host(x) rtol=1e-5

    function minus_one(i, x)
        @inbounds x[i] -= 1
    end

    JACC.parallel_for(N, minus_one, x)
    @test zeros(N)≈JACC.to_host(x) rtol=1e-5

    # fill
    N = 10
    x = JACC.fill(10.0, N)
    @test fill(10.0, N)≈JACC.to_host(x) rtol=1e-5
    fill!(x, 22.2)
    @test fill(22.2, N)≈JACC.to_host(x) rtol=1e-5

    # array
    x = JACC.array(10)
    @test ndims(x) == 1
    @test length(x) == 10
    @test eltype(x) == JACC.default_float()
    x = JACC.array(Float32, 10)
    @test size(x) == (10,)
    @test eltype(x) == Float32
    a = JACC.array(5, 4)
    b = JACC.array((5, 4))
    @test ndims(a) == 2
    @test size(a) == size(b)
    @test eltype(a) == JACC.default_float()
    x = JACC.array(; type = Int, dims = 10)
    @test eltype(x) == Int
    @test ndims(x) == 1
    x = JACC.array(; type = Complex{Float32}, dims = (5, 5, 5))
    @test ndims(x) == 3
    @test eltype(x) == Complex{Float32}
    @test size(x) == (5, 5, 5)
end

@testset "VectorAddLambda" begin
    function f(i, a)
        @inbounds a[i] += 5.0
    end

    alpha = 2.5

    N = 10
    dims = (N)
    a = round.(rand(Float32, dims) * 100)
    a_expected = a .+ 5.0

    a_device = JACC.to_device(a)
    JACC.parallel_for(N, f, a_device)

    @test JACC.to_host(a_device)≈a_expected rtol=1e-5
end

@testset "AXPY" begin
    alpha = 2.5

    N = 10
    # Generate random vectors x and y of length N for the interval [0, 100]
    x = round.(rand(Float32, N) * 100)
    y = round.(rand(Float32, N) * 100)
    alpha = 2.5

    x_device = JACC.to_device(x)
    y_device = JACC.to_device(y)
    JACC.parallel_for(N, axpy, alpha, x_device, y_device)

    x_expected = x
    seq_axpy(N, alpha, x_expected, y)

    @test JACC.to_host(x_device)≈x_expected rtol=1e-1
end

# using Cthulhu
@testset "AtomicCounter" begin
    function axpy_counter!(i, alpha, x, y, counter)
        @inbounds x[i] += alpha * y[i]
        JACC.@atomic counter[1] += 1
    end

    N = Int32(10)
    # Generate random vectors x and y of length N for the interval [0, 100]
    alpha = 2.5

    x = JACC.to_device(round.(rand(Float32, N) * 100))
    y = JACC.to_device(round.(rand(Float32, N) * 100))
    counter = JACC.to_device(Int32[0])
    JACC.parallel_for(N, axpy_counter!, alpha, x, y, counter)

    @test JACC.to_host(counter)[1] == N

    # TODO: clean this up
    # counter = JACC.zeros((1,1,1))
    # try
    #     JACC.parallel_for(N,
    #         (i, counter) -> begin
    #             JACC.@atomic counter[1,1,1] += 1.0
    #         end,
    #         counter)
    # catch err
    #     code_warntype(err; interactive = true)
    # end
    # @test Base.Array(counter)[1,1,1] == N
end

@testset "reduce" begin
    a = JACC.to_device([1 for i in 1:10])
    @test JACC.parallel_reduce(a) == 10
    @test JACC.parallel_reduce(min, a) == 1
    reducer = JACC.reducer(;
        type = eltype(a), range = JACC.array_size(a), op = +)
    reducer(a)
    @test JACC.get_result(reducer) == 10
    a2 = JACC.ones(Int, (2, 2))
    @test JACC.parallel_reduce(min, a2) == 1
    reducer = JACC.reducer(eltype(a2), JACC.array_size(a2), min)
    reducer(a2)
    @test JACC.get_result(reducer) == 1
    reducer(a2) do i, j, a
        a[i, j]
    end
    @test JACC.get_result(reducer) == 1

    SIZE = 1000
    ah = randn(FloatType, SIZE)
    ad = JACC.to_device(ah)
    mxd = JACC.parallel_reduce(SIZE, (i, a) -> a[i], ad; op = max, init = -Inf)
    @test mxd == maximum(ah)
    mxd = JACC.parallel_reduce(max, ad)
    @test mxd == maximum(ah)
    mnd = JACC.parallel_reduce(SIZE, (i, a) -> a[i], ad; op = min, init = Inf)
    @test mnd == minimum(ah)
    mnd = JACC.parallel_reduce(min, ad)
    @test mnd == minimum(ah)

    ah2 = randn(FloatType, (SIZE, SIZE))
    ad2 = JACC.to_device(ah2)
    mxd = JACC.parallel_reduce(
        (SIZE, SIZE), (i, j, a) -> a[i, j], ad2; op = max, init = -Inf)
    @test mxd == maximum(ah2)
    mxd = JACC.parallel_reduce(max, ad2)
    @test mxd == maximum(ah2)
    mnd = JACC.parallel_reduce(
        (SIZE, SIZE), (i, j, a) -> a[i, j], ad2; op = min, init = Inf)
    @test mnd == minimum(ah2)
    mnd = JACC.parallel_reduce(min, ad2)
    @test mnd == minimum(ah2)

    if FloatType != Float32
        SIZE = 10
        x = round.(rand(Float64, SIZE, SIZE) * 100)
        y = round.(rand(Float64, SIZE, SIZE) * 100)
        alpha = 2.5
        dx = JACC.to_device(x)
        dy = JACC.to_device(y)
        res = JACC.parallel_reduce((SIZE, SIZE), dot, dx, dy)
        @test res≈seq_dot(SIZE, SIZE, x, y) rtol=1e-1
    end
end

@testset "reduce-ND" begin
    Nend = JACC.backend == "metal" ? 6 : 7
    for N in 3:Nend
        dims = ntuple(_ -> 3, N)
        ah = randn(FloatType, dims)
        ad = JACC.to_device(ah)
        reducer = JACC.reducer(FloatType, dims)
        reducer(ad)
        @test JACC.get_result(reducer) ≈ sum(ah)

        p = JACC.parallel_reduce(dims, ad) do args...
            id = (args[1:(end - 1)])
            a = args[end]
            elem = a[id...]
            return elem * elem
        end
        @test p ≈ LinearAlgebra.dot(ah, ah)

        mxd = JACC.parallel_reduce(dims,
            (args...) -> begin
                id = (args[1:(end - 1)])
                a = args[end]
                return a[id...]
            end,
            ad; op = max, init = -Inf)
        @test mxd == maximum(ah)

        mnd = JACC.parallel_reduce(min, ad)
        @test mnd == minimum(ah)
    end
end

@testset "LaunchSpec" begin
    # 1D
    N = 100
    dims = (N)
    a = round.(rand(Float32, dims) * 100)
    a_expected = a .+ 5.0
    a_device = JACC.to_device(a)
    JACC.parallel_for(JACC.launch_spec(; threads = 1000), N, a_device) do i, a
        @inbounds a[i] += 5.0
    end
    @test JACC.to_host(a_device)≈a_expected rtol=1e-5
    a_expected = a_expected .+ 5.0
    JACC.parallel_for(; range = N, args = (a_device,),
        f = (i, a) -> begin
            @inbounds a[i] += 5.0
        end, threads = 1000,
        sync = false)
    JACC.synchronize()
    @test JACC.to_host(a_device)≈a_expected rtol=1e-5

    # 2D
    A = JACC.ones(Float32, N, N)
    B = JACC.ones(Float32, N, N)
    C = JACC.zeros(Float32, N, N)
    JACC.parallel_for(JACC.launch_spec(; threads = (16, 16)),
        (N, N), (i, j, A, B, C) -> begin
            @inbounds C[i, j] = A[i, j] + B[i, j]
        end,
        A, B, C)
    C_expected = Float32(2.0) .* ones(Float32, N, N)
    @test JACC.to_host(C)≈C_expected rtol=1e-5

    # 3D
    A = JACC.ones(Float32, N, N, N)
    B = JACC.ones(Float32, N, N, N)
    C = JACC.zeros(Float32, N, N, N)
    JACC.parallel_for(JACC.launch_spec(; threads = (4, 4, 4)),
        (N, N, N), (i, j, k, A, B,
            C) -> begin
            @inbounds C[i, j, k] = A[i, j, k] + B[i, j, k]
        end,
        A, B, C)
    C_expected = Float32(2.0) .* ones(Float32, N, N, N)
    @test JACC.to_host(C)≈C_expected rtol=1e-5

    # reduce
    a = JACC.ones(N)
    res = JACC.parallel_reduce(JACC.launch_spec(), a)
    @test JACC.to_host(res)[] == N
    res = JACC.parallel_reduce(JACC.launch_spec(), N, (i, a) -> a[i], a)
    @test JACC.to_host(res)[] == N
    res = JACC.parallel_reduce(;
        range = N, f = (i, a) -> begin
            a[i]
        end, args = (a,), sync = false)
    JACC.synchronize()
    @test JACC.to_host(res)[] == N
    res = JACC.parallel_reduce(JACC.launch_spec(), max, a)
    JACC.synchronize()
    @test JACC.to_host(res)[] == 1
    res = JACC.parallel_reduce(JACC.launch_spec(), min, a)
    JACC.synchronize()
    @test JACC.to_host(res)[] == 1
    res = JACC.parallel_reduce(
        JACC.launch_spec(), N, (i, a) -> a[i], a; op = max, init = -Inf)
    @test JACC.to_host(res)[] == 1
    a2 = JACC.ones(N, N)
    res = JACC.parallel_reduce(JACC.launch_spec(), a2)
    @test JACC.to_host(res)[] == N * N
    res = JACC.parallel_reduce(
        JACC.launch_spec(), (N, N), (i, j, a) -> a[i, j], a2)
    @test JACC.to_host(res)[] == N * N
    res = JACC.parallel_reduce(JACC.launch_spec(), min, a2)
    @test JACC.to_host(res)[] == 1
    res = JACC.parallel_reduce(
        JACC.launch_spec(), (N, N), (i, j, a) -> a[i, j],
        a2; op = max, init = -Inf)
    @test JACC.to_host(res)[] == 1
end

if JACC.backend != "oneapi"
    @testset "shared" begin
        N = 100
        alpha = 2.5
        x = JACC.ones(N)
        x_shared = JACC.ones(N)
        y = rand_jacc(N)

        function scal(i, x, y, alpha)
            @inbounds x[i] = y[i] * alpha
        end

        function scal_shared(i, x, y, alpha)
            y_shared = JACC.shared(y)
            @inbounds x[i] = y_shared[i] * alpha
        end

        # 1D kernel, 1D array
        JACC.parallel_for(N, scal, x, y, alpha)
        JACC.parallel_for(N, scal_shared, x_shared, y, alpha)
        @test isapprox(JACC.to_host(x), JACC.to_host(x_shared); rtol = 1e-8)
        spec = JACC.launch_spec(threads = 32)
        JACC.parallel_for(spec, N, scal_shared, x_shared, y, alpha)
        @test isapprox(JACC.to_host(x), JACC.to_host(x_shared); rtol = 1e-8)

        n = 10
        S = (n, n)
        A = JACC.zeros(S)
        A_shared = JACC.zeros(S)
        B = rand_jacc(S)

        function scal(i, j, A, B, alpha)
            @inbounds A[i, j] = B[i, j] * alpha
        end

        function scal_shared(i, j, A, B, alpha)
            B_shared = JACC.shared(B)
            @inbounds A[i, j] = B_shared[i, j] * alpha
        end

        # 2D kernel, 2D array
        JACC.parallel_for(S, scal, A, B, alpha)
        JACC.parallel_for(S, scal_shared, A_shared, B, alpha)
        @test isapprox(JACC.to_host(A), JACC.to_host(A_shared); rtol = 1e-8)
        spec = JACC.launch_spec(threads = (4, 4))
        JACC.parallel_for(spec, S, scal_shared, A_shared, B, alpha)
        @test isapprox(JACC.to_host(A), JACC.to_host(A_shared); rtol = 1e-8)
        spec = JACC.launch_spec(threads = (4, 2))
        JACC.parallel_for(spec, S, scal_shared, A_shared, B, alpha)
        @test isapprox(JACC.to_host(A), JACC.to_host(A_shared); rtol = 1e-8)
        spec = JACC.launch_spec(threads = (2, 4))
        JACC.parallel_for(spec, S, scal_shared, A_shared, B, alpha)
        @test isapprox(JACC.to_host(A), JACC.to_host(A_shared); rtol = 1e-8)

        # 2D kernel, 1D array
        A = JACC.zeros(S)
        A_shared = JACC.zeros(S)
        b = rand_jacc(n)
        JACC.parallel_for(S, A, b, alpha) do i, j, A, b, alpha
            @inbounds A[i, j] = b[i]
        end
        JACC.parallel_for(S, A_shared, b, alpha) do i, j, A, b, alpha
            b_shared = JACC.shared(b)
            @inbounds A[i, j] = b_shared[i]
        end
        @test isapprox(JACC.to_host(A), JACC.to_host(A_shared); rtol = 1e-8)
        spec = JACC.launch_spec(threads = (4, 4))
        JACC.parallel_for(spec, S, A_shared, b, alpha) do i, j, A, b, alpha
            b_shared = JACC.shared(b)
            @inbounds A[i, j] = b_shared[i]
        end
        @test isapprox(JACC.to_host(A), JACC.to_host(A_shared); rtol = 1e-8)
        spec = JACC.launch_spec(threads = (4, 2))
        JACC.parallel_for(spec, S, A_shared, b, alpha) do i, j, A, b, alpha
            b_shared = JACC.shared(b)
            @inbounds A[i, j] = b_shared[i]
        end
        @test isapprox(JACC.to_host(A), JACC.to_host(A_shared); rtol = 1e-8)
        spec = JACC.launch_spec(threads = (2, 4))
        JACC.parallel_for(spec, S, A_shared, b, alpha) do i, j, A, b, alpha
            b_shared = JACC.shared(b)
            @inbounds A[i, j] = b_shared[i]
        end
        @test isapprox(JACC.to_host(A), JACC.to_host(A_shared); rtol = 1e-8)

        function test_sync()
            ix = JACC.zeros(Int, N)
            spec = JACC.launch_spec(; threads = N, sync = true)
            JACC.parallel_for(spec, N, ix) do i, x
                shared_mem = JACC.shared(x)
                shared_mem[i] = i
                JACC.sync_workgroup()
                if i > 50
                    shared_mem[i] = shared_mem[i - 50]
                end
                x[i] = shared_mem[i]
            end
            ix_h = JACC.to_host(ix)
            for i in [1, 10, 25, 50]
                @test ix_h[i] == i
                @test ix_h[i + 50] == i
            end
        end
        test_sync()
        test_sync()
    end
else
    @testset "shared" begin
        N = 100
        alpha = 2.5
        x = JACC.ones(N)
        x_shared = JACC.ones(N)
        y = rand_jacc(N)

        function scal(i, x, y, alpha)
            @inbounds x[i] = y[i] * alpha
        end

        function scal_shared(i, x, y, alpha)
            y_shared = JACC.shared(y)
            @inbounds x[i] = y_shared[i] * alpha
        end

        # 1D kernel, 1D array
        JACC.parallel_for(N, scal, x, y, alpha)
        JACC.parallel_for(N, scal_shared, x_shared, y, alpha)
        @test isapprox(JACC.to_host(x), JACC.to_host(x_shared); rtol = 1e-8)
    end
end

@testset "JACC.BLAS" begin
    x = ones(1_000)
    y = ones(1_000)
    jx = JACC.ones(1_000)
    jy = JACC.ones(1_000)
    alpha = 2.0

    seq_axpy(1_000, alpha, x, y)
    ref_result = seq_dot(1_000, x, y)

    JACC.BLAS.axpy(1_000, alpha, jx, jy)
    jresult = JACC.BLAS.dot(1_000, jx, jy)

    @test jresult≈ref_result rtol=1e-8

    x = ones(1_000)
    y = ones(1_000)
    y1 = y * 2
    jx = JACC.ones(1_000)
    jy = JACC.ones(1_000)
    jy1 = jy * 2
    alpha = 2.0

    function seq_scal(N, alpha, x)
        for i in 1:N
            @inbounds x[i] = alpha * x[i]
        end
    end

    function seq_asum(N, x)
        r = 0.0
        for i in 1:N
            @inbounds r += abs(x[i])
        end
        return r
    end

    function seq_nrm2(N, x)
        sum_sq = 0.0
        for i in 1:N
            @inbounds sum_sq += x[i] * x[i]
        end
        r = sqrt(sum_sq)
        return r
    end

    function seq_swap(N, x, y1)
        for i in 1:N
            @inbounds t = x[i]
            @inbounds x[i] = y1[i]
            @inbounds y1[i] = t
        end
    end

    ref_result = seq_axpy(1_000, alpha, x, y)
    ref_result = seq_dot(1_000, x, y)
    JACC.BLAS.axpy(1_000, alpha, jx, jy)
    jresult = JACC.BLAS.dot(1_000, jx, jy)
    @test jresult≈ref_result rtol=1e-8

    seq_scal(1_000, alpha, x)
    JACC.BLAS.scal(1_000, alpha, jx)
    @test x≈JACC.to_host(jx) rtol=1e-8

    seq_axpy(1_000, alpha, x, y)
    JACC.BLAS.axpy(1_000, alpha, jx, jy)
    @test x≈JACC.to_host(jx) atol=1e-8

    r1 = seq_dot(1_000, x, y)
    r2 = JACC.BLAS.dot(1_000, jx, jy)
    @test r1≈r2 atol=1e-8

    r1 = seq_asum(1_000, x)
    r2 = JACC.BLAS.asum(1_000, jx)
    @test r1≈r2 atol=1e-8
    r1 = seq_nrm2(1_000, x)
    r2 = JACC.BLAS.nrm2(1_000, jx)
    @test r1≈r2 atol=1e-8

    seq_swap(1_000, x, y1)
    JACC.BLAS.swap(1_000, jx, jy1)
    @test x == JACC.to_host(jx)
    @test y1 == JACC.to_host(jy1)
end

@testset "Add-2D" begin
    function add!(i, j, A, B, C)
        @inbounds C[i, j] = A[i, j] + B[i, j]
    end

    M = 10
    N = 10
    A = JACC.ones(Float32, M, N)
    B = JACC.ones(Float32, M, N)
    C = JACC.zeros(Float32, M, N)

    JACC.parallel_for((M, N), add!, A, B, C)

    C_expected = Float32(2.0) .* ones(Float32, M, N)
    @test JACC.to_host(C)≈C_expected rtol=1e-5
end

@testset "Add-3D" begin
    function add!(i, j, k, A, B, C)
        @inbounds C[i, j, k] = A[i, j, k] + B[i, j, k]
    end

    L = 10
    M = 10
    N = 10
    A = JACC.ones(Float32, L, M, N)
    B = JACC.ones(Float32, L, M, N)
    C = JACC.zeros(Float32, L, M, N)

    JACC.parallel_for((L, M, N), add!, A, B, C)

    C_expected = Float32(2.0) .* ones(Float32, L, M, N)
    @test JACC.to_host(C)≈C_expected rtol=1e-5
end

@inline function init_add(N)
    dims = ntuple(_ -> 3, N)
    A = JACC.ones(Float32, dims)
    B = JACC.ones(Float32, dims)
    C = JACC.zeros(Float32, dims)
    return dims, A, B, C
end

@testset "Add-ND" begin
    let N = 4
        dims, A, B, C = init_add(N)
        JACC.parallel_for(dims, A, B, C) do i1, i2, i3, i4, A, B, C
            id = CartesianIndex(i1, i2, i3, i4)
            C[id] = A[id] + B[id]
        end
        C_expected = Float32(2.0) .* ones(Float32, dims)
        @test JACC.to_host(C)≈C_expected rtol=1e-5
    end

    let N = 5
        dims, A, B, C = init_add(N)
        JACC.parallel_for(dims, A, B, C) do i1, i2, i3, i4, i5, A, B, C
            id = CartesianIndex(i1, i2, i3, i4, i5)
            C[id] = A[id] + B[id]
        end
        C_expected = Float32(2.0) .* ones(Float32, dims)
        @test JACC.to_host(C)≈C_expected rtol=1e-5
    end

    let N = 6
        dims, A, B, C = init_add(N)
        JACC.parallel_for(dims, A, B, C) do i1, i2, i3, i4, i5, i6, A, B, C
            id = CartesianIndex(i1, i2, i3, i4, i5, i6)
            C[id] = A[id] + B[id]
        end
        C_expected = Float32(2.0) .* ones(Float32, dims)
        @test JACC.to_host(C)≈C_expected rtol=1e-5
    end

    if JACC.backend != "metal"
        let N = 7
            dims, A, B, C = init_add(N)
            JACC.parallel_for(
                dims, A, B, C) do i1, i2, i3, i4, i5, i6, i7, A, B, C
                id = CartesianIndex(i1, i2, i3, i4, i5, i6, i7)
                C[id] = A[id] + B[id]
            end
            C_expected = Float32(2.0) .* ones(Float32, dims)
            @test JACC.to_host(C)≈C_expected rtol=1e-5
        end
    end
end

@testset "do" begin
    L = 10
    M = 10
    N = 10

    # 1D
    a = round.(rand(Float32, N) * 100)
    a_expected = a .+ 5.0

    a_device = JACC.to_device(a)
    JACC.parallel_for(N, a_device) do i, a
        @inbounds a[i] += 5.0
    end
    @test JACC.to_host(a_device)≈a_expected rtol=1e-5

    a_device = JACC.to_device(a)
    res = JACC.parallel_reduce(N, a_device) do i, a
        a[i] * a[i]
    end
    @test res≈seq_dot(N, a, a) rtol=1e-1

    res = JACC.parallel_reduce(N, a_device; op = min, init = Inf) do i, a
        a[i]
    end
    @test res ≈ minimum(a)

    # 2D
    A2 = JACC.ones(Float32, M, N)
    B2 = JACC.ones(Float32, M, N)
    C2 = JACC.zeros(Float32, M, N)
    JACC.parallel_for((M, N), A2, B2, C2) do i, j, A, B, C
        @inbounds C[i, j] = A[i, j] + B[i, j]
    end
    C2_expected = Float32(2.0) .* ones(Float32, M, N)
    @test JACC.to_host(C2)≈C2_expected rtol=1e-5

    res = JACC.parallel_reduce((M, N), A2, B2) do i, j, a, b
        a[i, j] * b[i, j]
    end
    @test res≈seq_dot(M, N, JACC.to_host(A2), JACC.to_host(B2)) rtol=1e-1

    res = JACC.parallel_reduce((M, N), A2; op = min, init = Inf) do i, j, a
        a[i]
    end
    @test res ≈ 1

    # 3D
    A3 = JACC.ones(Float32, L, M, N)
    B3 = JACC.ones(Float32, L, M, N)
    C3 = JACC.zeros(Float32, L, M, N)

    JACC.parallel_for((L, M, N), A3, B3, C3) do i, j, k, A, B, C
        @inbounds C[i, j, k] = A[i, j, k] + B[i, j, k]
    end

    C3_expected = Float32(2.0) .* ones(Float32, L, M, N)
    @test JACC.to_host(C3)≈C3_expected rtol=1e-5

    # 1D
    N = 100
    a = round.(rand(Float32, N) * 100)
    a_expected = a .+ 5.0
    a_device = JACC.to_device(a)
    JACC.parallel_for(JACC.launch_spec(; threads = 1000), N, a_device) do i, a
        @inbounds a[i] += 5.0
    end
    @test JACC.to_host(a_device)≈a_expected rtol=1e-5

    # 2D
    A = JACC.ones(Float32, N, N)
    B = JACC.ones(Float32, N, N)
    C = JACC.zeros(Float32, N, N)
    JACC.parallel_for(JACC.launch_spec(; threads = (16, 16)),
        (N, N), A, B, C) do i, j, A, B, C
        @inbounds C[i, j] = A[i, j] + B[i, j]
    end
    C_expected = Float32(2.0) .* ones(Float32, N, N)
    @test JACC.to_host(C)≈C_expected rtol=1e-5

    # 3D
    A = JACC.ones(Float32, N, N, N)
    B = JACC.ones(Float32, N, N, N)
    C = JACC.zeros(Float32, N, N, N)
    JACC.parallel_for(JACC.launch_spec(; threads = (4, 4, 4)),
        (N, N, N), A, B, C) do i, j, k, A, B, C
        @inbounds C[i, j, k] = A[i, j, k] + B[i, j, k]
    end
    C_expected = Float32(2.0) .* ones(Float32, N, N, N)
    @test JACC.to_host(C)≈C_expected rtol=1e-5
end

@testset "macro" begin
    add5! = (i, a) -> a[i] += 5.0;
    L = 10
    M = 10
    N = 10

    # 1D
    a_device = rand_jacc(N)
    a_host = JACC.to_host(a_device)
    a_expected = a_host .+ 5.0
    JACC.@parallel_for range=N add5!(a_device)
    @test JACC.to_host(a_device) ≈ a_expected rtol=1e-5

    a_device = JACC.to_device(a_host)
    ret = JACC.@parallel_reduce range=N dot(a_device, a_device)
    res = JACC.to_host(ret)[]
    @test res ≈ seq_dot(N, a_host, a_host) rtol=1e-1

    ret = JACC.@parallel_reduce range=N op=min init=Inf JACC.elem_access(a_device)
    res = JACC.to_host(ret)[]
    @test res ≈ minimum(a_host)

    # 2D
    A2 = JACC.ones(M, N)
    B2 = JACC.ones(M, N)
    C2 = JACC.zeros(M, N)
    matrix_sum = (i, j, A, B, C) -> C[i, j] = A[i, j] + B[i, j]
    JACC.@parallel_for range=(M, N) matrix_sum(A2, B2, C2)
    C2_expected = Float32(2.0) .* ones(Float32, M, N)
    @test JACC.to_host(C2) ≈ C2_expected rtol=1e-5

    ret = JACC.@parallel_reduce range=(M, N) dot(A2, B2)
    res = JACC.to_host(ret)[]
    @test res ≈ seq_dot(M, N, JACC.to_host(A2), JACC.to_host(B2)) rtol=1e-5

    ret = JACC.@parallel_reduce range=(M, N) op=min init=Inf JACC.elem_access(A2)
    res = JACC.to_host(ret)[]
    @test res ≈ 1
end

@testset "UnitRange" begin
    a = JACC.zeros(10)

    JACC.parallel_for(2:9, a) do i, a
        a[i] = 1.0
    end
    a_host = JACC.to_host(a)
    @test a_host[2:9] ≈ ones(8)
    @test a_host[1] == 0.0
    @test a_host[10] == 0.0

    A = JACC.zeros(10, 10)

    JACC.parallel_for((2:9, 2:9), A) do i, j, A
        A[i, j] = 1.0
    end
    A_host = JACC.to_host(A)
    @test A_host[2:9, 2:9] ≈ ones(8, 8)
    @test A_host[1, :] ≈ zeros(10)
    @test A_host[10, :] ≈ zeros(10)
    @test A_host[:, 1] ≈ zeros(10)
    @test A_host[:, 10] ≈ zeros(10)

    res = JACC.parallel_reduce(JACC.elem_access, 3:8, a)
    @test res ≈ 6.0

    res = JACC.parallel_reduce(JACC.elem_access, (3:8, 3:8), A)
    @test res ≈ 36.0
end

if JACC.backend != "metal"
    @testset "StepRange" begin
        a = JACC.zeros(10)

        JACC.parallel_for(2:9, a) do i, a
            a[i] = 1.0
        end
        JACC.parallel_for(
            range = 2:2:8, f = (i, a) -> a[i] = 2.0, args = (a,), threads = 8)
        a_host = JACC.to_host(a)
        @test a_host[1] == 0.0
        @test a_host[10] == 0.0
        @test a_host[2:2:8] ≈ 2.0 .* ones(4)
        @test a_host[3:2:7] ≈ ones(3)

        A = JACC.zeros(10, 10)

        JACC.parallel_for((2:9, 2:9), A) do i, j, A
            A[i, j] = 1.0
        end
        JACC.parallel_for(
            range = (2:2:8, 2:2:8), f = (i, j, a) -> a[i, j] = 2.0,
            args = (A,), threads = (8, 8))
        A_host = JACC.to_host(A)
        @test A_host[2:2:8, 2:2:8] ≈ 2.0 .* ones(4, 4)
        @test A_host[3:2:7, 3:2:7] ≈ ones(3, 3)
        @test A_host[1, :] ≈ zeros(10)
        @test A_host[10, :] ≈ zeros(10)
        @test A_host[:, 1] ≈ zeros(10)
        @test A_host[:, 10] ≈ zeros(10)

        ret = JACC.@parallel_reduce range=2:3:8 op=min JACC.elem_access(a)
        res = JACC.to_host(ret)[]
        @test res ≈ 1.0

        ret = JACC.parallel_reduce(;
            range = 2:2:8, f = JACC.elem_access, args = (a,), sync = false)
        JACC.synchronize()
        res = JACC.to_host(ret)[]
        @test res ≈ 8.0

        ret = JACC.@parallel_reduce range=(2:3:8, 2:3:8) op=min JACC.elem_access(A)
        res = JACC.to_host(ret)[]
        @test res ≈ 1.0

        ret = JACC.parallel_reduce(;
            range = (2:2:8, 2:2:8), f = JACC.elem_access, args = (A,), sync = false)
        JACC.synchronize()
        res = JACC.to_host(ret)[]
        @test res ≈ 32.0
    end
end

@testset "CG" begin
    function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
        end
    end

    SIZE = 10
    a0 = JACC.ones(SIZE)
    a1 = JACC.ones(SIZE)
    a2 = JACC.ones(SIZE)
    r = JACC.ones(SIZE)
    p = JACC.ones(SIZE)
    s = JACC.zeros(SIZE)
    x = JACC.zeros(SIZE)
    r_old = JACC.zeros(SIZE)
    r_aux = JACC.zeros(SIZE)
    a1 = a1 * 4
    r = r * 0.5
    p = p * 0.5
    cond = 1.0

    while cond[1, 1] >= 1e-14
        r_old = copy(r)

        JACC.parallel_for(SIZE, matvecmul, a0, a1, a2, p, s, SIZE)

        alpha0 = JACC.parallel_reduce(SIZE, dot, r, r)
        alpha1 = JACC.parallel_reduce(SIZE, dot, p, s)

        alpha = alpha0 / alpha1
        negative_alpha = alpha * -1.0

        JACC.parallel_for(SIZE, axpy, negative_alpha, r, s)
        JACC.parallel_for(SIZE, axpy, alpha, x, p)

        beta0 = JACC.parallel_reduce(SIZE, dot, r, r)
        beta1 = JACC.parallel_reduce(SIZE, dot, r_old, r_old)
        beta = beta0 / beta1

        r_aux = copy(r)

        JACC.parallel_for(SIZE, axpy, beta, r_aux, p)
        ccond = JACC.parallel_reduce(SIZE, dot, r, r)
        cond = ccond

        p = copy(r_aux)
    end
    @test cond[1, 1] <= 1e-14
end

@testset "LBM" begin
    function lbm_kernel(x, y, f, f1, f2, t, w, cx, cy, SIZE)
        u = 0.0
        v = 0.0
        p = 0.0
        x_stream = 0
        y_stream = 0

        if x > 1 && x < SIZE && y > 1 && y < SIZE
            for k in 1:9
                @inbounds x_stream = x - cx[k]
                @inbounds y_stream = y - cy[k]
                ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                iind = (k - 1) * SIZE * SIZE + x_stream * SIZE + y_stream
                @inbounds f[floor(Int, ind)] = f1[floor(Int, iind)]
            end
            for k in 1:9
                ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                @inbounds p = p[1, 1] + f[ind]
                @inbounds u = u[1, 1] + f[ind] * cx[k]
                @inbounds v = v[1, 1] + f[ind] * cy[k]
            end
            u = u / p
            v = v / p
            for k in 1:9
                @inbounds cu = cx[k] * u + cy[k] * v
                @inbounds feq = w[k] * p *
                                (1.0 + 3.0 * cu + cu * cu -
                                 1.5 * ((u * u) + (v * v)))
                ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                @inbounds f2[floor(Int, ind)] = f[floor(Int, ind)] *
                                                (1.0 - 1.0 / t) + feq * 1 / t
            end
        end
    end

    function lbm_threads(f, f1, f2, t, w, cx, cy, SIZE)
        Threads.@sync Threads.@threads for x in 1:SIZE
            for y in 1:SIZE
                u = 0.0
                v = 0.0
                p = 0.0
                x_stream = 0
                y_stream = 0

                if x > 1 && x < SIZE && y > 1 && y < SIZE
                    for k in 1:9
                        @inbounds x_stream = x - cx[k]
                        @inbounds y_stream = y - cy[k]
                        ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                        iind = (k - 1) * SIZE * SIZE + x_stream * SIZE +
                               y_stream
                        @inbounds f[trunc(Int, ind)] = f1[trunc(Int, iind)]
                    end
                    for k in 1:9
                        ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                        @inbounds p = p[1, 1] + f[ind]
                        @inbounds u = u[1, 1] + f[ind] * cx[k]
                        @inbounds v = v[1, 1] + f[ind] * cy[k]
                    end
                    u = u / p
                    v = v / p
                    for k in 1:9
                        @inbounds cu = cx[k] * u + cy[k] * v
                        @inbounds feq = w[k] * p *
                                        (1.0 + 3.0 * cu + cu * cu -
                                         1.5 * ((u * u) + (v * v)))
                        ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                        @inbounds f2[trunc(Int, ind)] = f[trunc(Int, ind)] *
                                                        (1.0 - 1.0 / t) +
                                                        feq * 1 / t
                    end
                end
            end
        end
    end

    SIZE = 10
    f = ones(SIZE * SIZE * 9) .* 2.0
    f1 = ones(SIZE * SIZE * 9) .* 3.0
    f2 = ones(SIZE * SIZE * 9) .* 4.0
    cx = zeros(Int, 9)
    cy = zeros(Int, 9)
    cx[1] = 0
    cy[1] = 0
    cx[2] = 1
    cy[2] = 0
    cx[3] = -1
    cy[3] = 0
    cx[4] = 0
    cy[4] = 1
    cx[5] = 0
    cy[5] = -1
    cx[6] = 1
    cy[6] = 1
    cx[7] = -1
    cy[7] = 1
    cx[8] = -1
    cy[8] = -1
    cx[9] = 1
    cy[9] = -1
    w = ones(9)
    t = 1.0

    df = JACC.to_device(f)
    df1 = JACC.to_device(f1)
    df2 = JACC.to_device(f2)
    dcx = JACC.to_device(cx)
    dcy = JACC.to_device(cy)
    dw = JACC.to_device(w)

    JACC.parallel_for(
        (SIZE, SIZE), lbm_kernel, df, df1, df2, t, dw, dcx, dcy, SIZE)

    lbm_threads(f, f1, f2, t, w, cx, cy, SIZE)

    @test f2≈JACC.to_host(df2) rtol=1e-1
end

if JACC.backend != "metal"
    @testset "Multi" begin
        # Unidimensional arrays
        SIZE = 10
        x = round.(rand(Float64, SIZE) * 100)
        y = round.(rand(Float64, SIZE) * 100)
        alpha = 2.5
        dx = JACC.Multi.array(x)
        dy = JACC.Multi.array(y)
        JACC.Multi.parallel_for(SIZE, alpha, dx, dy) do i, alpha, x, y
            x[i] += alpha * y[i]
        end
        x_expected = x
        seq_axpy(SIZE, alpha, x_expected, y)
        @test JACC.to_host(dx)≈x_expected rtol=1e-1
        res = JACC.Multi.parallel_reduce(SIZE, dot, dx, dy)
        @test res≈seq_dot(SIZE, x_expected, y) rtol=1e-1

        # Multidimensional arrays
        SIZE = 10
        x = round.(rand(Float64, SIZE, SIZE) * 100)
        y = round.(rand(Float64, SIZE, SIZE) * 100)
        alpha = 2.5
        dx = JACC.Multi.array(x)
        dy = JACC.Multi.array(y)
        JACC.Multi.parallel_for(
            (SIZE, SIZE), alpha, dx, dy) do i, j, alpha, x,
        y
            x[i, j] += alpha * y[i, j]
        end
        x_expected = x
        seq_axpy(SIZE, SIZE, alpha, x_expected, y)
        @test JACC.to_host(dx)≈x_expected rtol=1e-1
        res = JACC.Multi.parallel_reduce((SIZE, SIZE), dot, dx, dy)
        @test res≈seq_dot(SIZE, SIZE, x_expected, y) rtol=1e-1

        # HPCG example
        function matvecmul(i, a1, a2, a3, x, y, SIZE, ndev)
            ind = JACC.Multi.ghost_shift(i, a1)
            dev_id = JACC.Multi.device_id(a1)
            if dev_id == 1 && i == 1
                y[ind] = a2[ind] * x[ind] + a1[ind] * x[ind + 1]
            elseif dev_id == ndev && i == SIZE
                y[ind] = a3[ind] * x[ind - 1] + a2[ind] * x[ind]
            else
                y[ind] = a3[ind] * x[ind - 1] + a2[ind] * x[ind] +
                         a1[ind] * x[ind + 1]
            end
        end

        SIZE = 10
        # Initialization of inputs
        a1 = ones(SIZE)
        a2 = ones(SIZE)
        a3 = ones(SIZE)
        r = ones(SIZE)
        p = ones(SIZE)
        s = zeros(SIZE)
        x = zeros(SIZE)
        r_old = zeros(SIZE)
        r_aux = zeros(SIZE)
        a2 = a2 * 4
        r = r * 0.5
        p = p * 0.5
        cond = 1.0
        ndev = JACC.Multi.ndev()
        gja1 = JACC.Multi.array(a1; ghost_dims = 1)
        gja2 = JACC.Multi.array(a2; ghost_dims = 1)
        gja3 = JACC.Multi.array(a3; ghost_dims = 1)
        jr = JACC.Multi.array(r)
        jp = JACC.Multi.array(p)
        gjp = JACC.Multi.array(p; ghost_dims = 1)
        js = JACC.Multi.array(s)
        gjs = JACC.Multi.array(s; ghost_dims = 1)
        jx = JACC.Multi.array(x)
        jr_old = JACC.Multi.array(r_old)
        jr_aux = JACC.Multi.array(r_aux)
        ssize = JACC.Multi.part_length(jp)
        # HPCG Algorithm
        while cond >= 1e-14
            JACC.Multi.copy!(jr_old, jr)
            JACC.Multi.parallel_for(
                SIZE, matvecmul, gja1, gja2, gja3, gjp, gjs, ssize, ndev)
            JACC.Multi.sync_ghost_elems!(gjs)
            JACC.Multi.copy!(js, gjs) #js = gjs
            alpha0 = JACC.Multi.parallel_reduce(SIZE, dot, jr, jr)
            alpha1 = JACC.Multi.parallel_reduce(SIZE, dot, jp, js)
            alpha = alpha0 / alpha1
            m_alpha = alpha * (-1.0)
            JACC.Multi.parallel_for(SIZE, axpy, m_alpha, jr, js)
            JACC.Multi.parallel_for(SIZE, axpy, alpha, jx, jp)
            beta0 = JACC.Multi.parallel_reduce(SIZE, dot, jr, jr)
            beta1 = JACC.Multi.parallel_reduce(SIZE, dot, jr_old, jr_old)
            beta = beta0 / beta1
            JACC.Multi.copy!(jr_aux, jr)
            JACC.Multi.parallel_for(SIZE, axpy, beta, jr_aux, jp)
            ccond = JACC.Multi.parallel_reduce(SIZE, dot, jr, jr)
            cond = ccond
            JACC.Multi.copy!(jp, jr_aux)
            JACC.Multi.copy!(gjp, jp) #gjp = jp
            JACC.Multi.sync_ghost_elems!(gjp)
        end
        @test cond <= 1e-14
    end
end

if JACC.backend != "amdgpu" && JACC.backend != "metal"
    @testset "CG Async" begin
        function matvecmul(i, a1, a2, a3, x, y, SIZE)
            if i == 1
                y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
            elseif i == SIZE
                y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
            elseif i > 1 && i < SIZE
                y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
            end
        end

        SIZE = 10
        a0 = JACC.Async.ones(1, SIZE)
        a1 = JACC.Async.ones(1, SIZE)
        a2 = JACC.Async.ones(1, SIZE)
        r = JACC.Async.ones(2, SIZE)
        p = JACC.Async.ones(1, SIZE)
        s1 = JACC.Async.zeros(1, SIZE)
        s2 = JACC.Async.zeros(2, SIZE)
        x = JACC.Async.zeros(1, SIZE)
        r_old = JACC.Async.zeros(1, SIZE)
        r_aux = JACC.Async.zeros(1, SIZE)
        a1 = a1 * 4
        r = r * 0.5
        p = p * 0.5
        cond = 1.0

        while cond[1, 1] >= 1e-14
            copyto!(r, r_old)

            JACC.Async.parallel_for(1, SIZE, matvecmul, a0, a1, a2, p, s1, SIZE)

            alpha1 = JACC.Async.parallel_reduce(1, SIZE, dot, p, s1)
            alpha0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
            JACC.Async.synchronize()

            alpha = JACC.to_host(alpha0)[] / JACC.to_host(alpha1)[]
            negative_alpha = alpha * -1.0

            copyto!(s2, s1)
            JACC.Async.parallel_for(1, SIZE, axpy, alpha, x, p)
            JACC.Async.parallel_for(2, SIZE, axpy, negative_alpha, r, s2)
            JACC.Async.synchronize()

            beta1 = JACC.Async.parallel_reduce(1, SIZE, dot, r_old, r_old)
            beta0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
            JACC.Async.synchronize()
            beta = JACC.to_host(beta0)[] / JACC.to_host(beta1)[]

            copyto!(r, r_aux)

            JACC.Async.parallel_for(1, SIZE, axpy, beta, r_aux, p)
            ccond = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
            JACC.Async.synchronize()
            cond = JACC.to_host(ccond)[]

            copyto!(p, r_aux)
        end
        @test cond[1, 1] <= 1e-14
    end
end


@testset "rand-Float32" begin
    N = 1_000
    x = JACC.zeros(Float32, N)

    function rand_kernel(i, x)
        @inbounds x[i] = rand(Float32) * 2 - 1
    end

    JACC.@parallel_for range=N rand_kernel(x)
    x_host = JACC.to_host(x)
    @test all(-1 .<= x_host .<= 1)

    JACC.@parallel_for range=N ((i, x) -> @inbounds x[i] = rand(Float32))(x)
    x_host = JACC.to_host(x)
    @test all(0 .<= x_host .<= 1)
end


if JACC.backend != "metal"
    @testset "rand-Float64" begin
        N = 1_000
        x = JACC.zeros(Float64, N)

        function rand_kernel(i, x)
            @inbounds x[i] = rand(Float64)
        end

        JACC.@parallel_for range=N rand_kernel(x)
        x_host = JACC.to_host(x)
        @test all(0 .<= x_host .<= 1)

        JACC.@parallel_for range=N ((i, x) -> @inbounds x[i] = rand(Float64))(x)
        x_host = JACC.to_host(x)
        @test all(0 .<= x_host .<= 1)
    end
end

