module BLAS

using JACC

# BLAS function definition
function _axpy(i, alpha, x, y)
    @inbounds x[i] += alpha * y[i]
end

function _dot(i, x, y)
    return @inbounds x[i] * y[i]
end

function _scal(i, alpha, x)
    @inbounds x[i] = alpha * x[i]
end

# Considering real vector as of now
function _asum(i, x)
    return @inbounds abs(x[i])
end

function _swap(i, x, y)
    @inbounds a = x[i]
    @inbounds x[i] = y[i]
    @inbounds y[i] = a
end

function _nrm2(i, x)
    return @inbounds x[i] * x[i]
end

function _copy(i, x, y)
    @inbounds y[i] = x[i]
end

# Considering real vectors for now
function _rot(i, x, y, c, s)
    @inbounds xi = x[i]
    @inbounds x[i] = c * xi + s * y[i]
    @inbounds y[i] = -s * xi + c * y[i]
end

# function _rotmg(i, d1, d2, x1, y1)

# end
# Computing parameters for a Givens Rotation. All parameters here are real scalars
function rotg(a, b)
    if abs(b) == 0
        c = 1.0
        s = 0.0
        r = abs(a)
        z = s
    elseif abs(a) == 0
        c = 0.0
        s = -sign(b)
        r = abs(b)
        z = 1.0
    else
        r = sqrt(a * a + b * b)
        c = a / r
        s = -b / r
        z = s
        if c != 0
            z = 1 / c
        else
            z = 1
        end
    end
    a = r
    b = z
    return a, b, c, s
end

# Parallel implementation of the BLAS function
function axpy(n::I, alpha, x, y) where {I <: Integer}
    JACC.parallel_for(n, _axpy, alpha, x, y)
end

function dot(n::I, x, y) where {I <: Integer}
    JACC.parallel_reduce(n, _dot, x, y)
end

function scal(n::I, alpha, x) where {I <: Integer}
    JACC.parallel_for(n, _scal, alpha, x)
end

function asum(n::I, x) where {I <: Integer}
    JACC.parallel_reduce(n, _asum, x)
end

function swap(n::I, x, y) where {I <: Integer}
    JACC.parallel_for(n, _swap, x, y)
end

function nrm2(n::I, x) where {I <: Integer}
    tmp = JACC.parallel_reduce(n, _nrm2, x)
    return sqrt.(tmp)
    # println(typeof(tmp))
    #JACC.parallel_for(1,_sqrt,tmp)
    # tmp_h = Base.Array(tmp)
    # println(typeof(tmp_h))
    # ttmp=sqrt(tmp_h[1])
    # println(typeof(ttmp))
    # tmp_h[1]=ttmp
    # return tmp_h
end

function copy(n::I, x, y) where {I <: Integer}
    JACC.parallel_for(n, _copy, x, y)
end

function rot(n::I, x, y, c, s) where {I <: Integer}
    JACC.parallel_for(n, _rot, x, y, c, s)
end

end # module BLAS
