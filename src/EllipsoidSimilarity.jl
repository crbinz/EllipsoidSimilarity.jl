module EllipsoidSimilarity

using PDMats, NLopt
import Distances: evaluate, SqMahalanobis

export Ellipsoid,
       Compound,
       TransformationEnergy,
       GeneralizedFocalDist,
       similarity

"""An Ellipsoid is defined as the level set 
(x - m)' A (x - m) = 1 

where m is a px1 vector and A is a pxp positive definite matrix.
"""
immutable Ellipsoid{T<:AbstractFloat}
    m::Vector{T}
    A::AbstractPDMat{T}
    
    function Ellipsoid(m2::Vector{T}, A2::Matrix{T})
        local A_pd
        try
            A_pd = PDMat(A2)
        catch e
            if isa(e, Base.LinAlg.PosDefException)
                error("Matrix must be postivie definite")
            else
                throw(e)
            end
        end

        if length(m2) != A_pd.dim
            error("Center vector dimension and matrix dimension do not match")
        end
            
        new(m2, A_pd)
    end
end

Ellipsoid{T<:AbstractFloat}(m::Vector{T},A::Matrix{T}) = Ellipsoid{T}(m,A)

abstract EllipsoidSimilarityMeasure <: Real

"""Compound similarity

Product of three exponential factors accounting for location,
orientation, and shape of the ellipsoids. It satisfies the properties
(given in the paper) of a strong similarity measure.

Ref: Moshtaghi, M., et al., "Clustering ellipses for anomaly
detection". Pattern Recognition 44 (2011) pp. 55-69.
"""
type Compound <: EllipsoidSimilarityMeasure end

"""Transformation energy similarity

Product of three exponential factors accounting for location,
orientation, and shape of the ellipsoids. It satisfies the properties
(given in the paper) of a strong similarity measure.

Ref: Moshtaghi, M., et al., "Clustering ellipses for anomaly
detection". Pattern Recognition 44 (2011) pp. 55-69.
"""
type TransformationEnergy <: EllipsoidSimilarityMeasure end

"""Generalized Focal Distance

Average of the planar focal distances in all dimensions.

Ref: Moshtaghi, M., et al., "Clustering ellipses for anomaly
detection". Pattern Recognition 44 (2011) pp. 55-69.
"""
type GeneralizedFocalDist <: EllipsoidSimilarityMeasure end

function similarity( meas::GeneralizedFocalDist, E1::Ellipsoid, E2::Ellipsoid )
    epts = focalsegments(E1)
    fpts = focalsegments(E2)

    dis = 0.0
    for i in 1:(E1.A.dim-1)
        dis += focaldist(epts[i], fpts[i])
    end
    return 1.0 - dis/(E1.A.dim-1)
end

function focaldist( epts::Tuple{Vector{Float64},Vector{Float64}},
                    fpts::Tuple{Vector{Float64},Vector{Float64}} )
    # d ordering:
    #   d(e1, f1)
    #   d(e1, f2)
    #   d(e2, f1)
    #   d(e2, f2)
    d = zeros(4)
    for i in 1:2
        d[2i-1] = norm(epts[i] - fpts[1])
        d[2i]   = norm(epts[i] - fpts[2])
    end

    return 0.25 * ( minimum(d[1:2]) +
                    minimum(d[3:4]) +
                    minimum([d[1],d[3]]) +
                    minimum([d[2],d[4]]) )
end

"Returns coordinates of all focal segment endpoints of E"
function focalsegments( E::Ellipsoid )
    segments = Array{Tuple{Vector{eltype(E.m)},Vector{eltype(E.m)}}}(E.A.dim - 1)

    (evals, evecs) = eig(E.A.mat)
    sortidx = sortperm(evals)

    for i in 1:(E.A.dim-1)
        α2 = evals[sortidx[i+1]]
        u2 = evecs[:,sortidx[i+1]]
        α1 = evals[sortidx[i]]

        s = 0.5 * sqrt((α2 - α1)/(α2 * α1))
        segments[i] = (E.m + s*u2, E.m - s*u2)
    end
    return segments
end

function similarity( meas::TransformationEnergy, E1::Ellipsoid, E2::Ellipsoid; approx = false )
    S1, R1 = scale_and_rot_matrix(E1.A)
    S2, R2 = scale_and_rot_matrix(E2.A)
    M_12 = S2*R2*inv(R1)*inv(S1)
    M_21 = S1*R1*inv(R2)*inv(S2)
    d_12 = S2*R2*(E2.m - E1.m)
    d_21 = S1*R1*(E1.m - E2.m)

    if !approx
        # solve the minimization problem posed in the paper
        dist_1 = -te_minimize(M_12, d_12)
        dist_2 = -te_minimize(M_21, d_21)

        return 1.0/maximum([dist_1,dist_2])
    else
        # approximation given by the authors at the end of Section 4.
        # find the maximum singular values for each of M_12 and M_21
        σ_12 = maximum(svd(M_12)[2])
        σ_21 = maximum(svd(M_21)[2])

        return 1.0 / maximum([σ_12 + norm(d_12),
                        σ_21 + norm(d_21)])
    end
end

function te_minimize{T<:AbstractFloat}( M::Matrix{T}, d::Vector{T} )
    N = length(d)
    @assert N == size(M)[1]
    f(x, grad) = -norm(M*x + d)
    opt = Opt(:LN_COBYLA, N)
    xtol_rel!(opt,1e-4)

    min_objective!(opt, f)

    h(x, grad) = norm(x) - 1.0 # should be unit vector
    equality_constraint!(opt, h, 1e-6)

    xi = zeros(N)
    xi[1] = 1.0
    (minf,minx,ret) = optimize(opt, xi)

    return minf
end

"""Compute the 'scale matrix' given an ellipse represented by `A`. This
is a diagonal matrix whose elements are the recipricol of the square
root of the eigenvalues of `A`.
"""
function scale_and_rot_matrix{T<:AbstractFloat}( A::AbstractPDMat{T} )
    (_, S, V) = svd(A.mat)
    return (sqrt(inv(diagm(S))), V)
end


function similarity( meas::Compound, E1::Ellipsoid, E2::Ellipsoid, p = 2 )
    check_dims(E1,E2)
    
    # the authors state that using the Euclidean norm in
    # `location_similarity` leads to the measure being highly
    # sensitive to differences in the mean. Instead, they recommend
    # the use of a generalized statistical measure, the Mahalanobis
    # distance, which is implemented here. A general
    # `location_similarity` function for use with any norm is provided
    # below.
    g1 = location_similarity_mahal(E1.m, E1.A.mat, E2.m, E2.A.mat )

    g2 = orientation_similarity(E1.A.mat, E2.A.mat, p)
    
    g3 = shape_similarity(E1.A.mat, E2.A.mat, p)

    return g1*g2*g3
end
    
function compound_similarity{T<:AbstractFloat}( m1::Vector{T}, A1::Matrix{T},
                                                m2::Vector{T}, A2::Matrix{T} )
    compound_similarity(Ellipsoid(m1,A1),Ellipsoid(m2,A2))
end

function check_dims(E1::Ellipsoid, E2::Ellipsoid)
    @assert E1.A.dim == E2.A.dim
end

function location_similarity{T<:AbstractFloat}( m1::Vector{T}, m2::Vector{T}, p )
    return exp(-norm(m1 - m2, p))
end

function location_similarity_mahal{T<:AbstractFloat}( m1::Vector{T}, A1::Matrix{T},
                                                      m2::Vector{T}, A2::Matrix{T} )
    return exp(-evaluate(SqMahalanobis(inv(A1+A2)),m1,m2))
end

function orientation_similarity{T<:AbstractFloat}(A1::Matrix{T}, A2::Matrix{T}, p )
    R1 = eig(A1)[2] 
    R2 = eig(A2)[2]
    
    θ = acos(diag(R1'*R2))
    return exp(-norm(sin(θ), p))
end
               
function shape_similarity{T<:AbstractFloat}(A1::Matrix{T}, A2::Matrix{T}, p )
    α = sort!(eig(A1)[1])
    β = sort!(eig(A2)[1])

    α_star = 1./sqrt(α)
    β_star = 1./sqrt(β)

    return exp(-norm(α_star - β_star, p))
end

end


    
