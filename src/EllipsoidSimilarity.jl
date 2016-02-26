module EllipsoidSimilarity

using PDMats

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


"""Compound similarity

Product of three exponential factors accounting for location,
orientation, and shape of the ellipsoids. It satisfies the properties
of a strong similarity measure.

Ref: Moshtaghi, M., et al., "Clustering ellipses for anomaly
detection". Pattern Recognition 44 (2011) pp. 55-69.
"""
function compound_similarity( E1::Ellipsoid, E2::Ellipsoid, p = 2 )
    check_dims(E1,E2)
    
    g1 = location_similarity(E1.m, E2.m, p)

    g2 = orientation_similarity(E1.A, E2.A, p)
    
    g3 = shape_similarity(E1.A, E2.A, p)

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

function orientation_similarity{T<:AbstractFloat}(A1::AbstractPDMat{T}, A2::AbstractPDMat{T}, p )
    R1 = eig(A1.mat)[2] 
    R2 = eig(A2.mat)[2]
    
    θ = acos(diag(R1'*R2))
    return exp(-norm(sin(θ), p))
end
               
function shape_similarity{T<:AbstractFloat}(A1::AbstractPDMat{T}, A2::AbstractPDMat{T}, p )
    α = sort!(eig(A1.mat)[1])
    β = sort!(eig(A2.mat)[1])

    α_star = 1./sqrt(α)
    β_star = 1./sqrt(β)

    return exp(-norm(α_star - β_star, p))
end

end
