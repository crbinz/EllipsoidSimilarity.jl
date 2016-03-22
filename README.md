### Installation
In the Julia REPL, type
`Pkg.clone("https://github.com/crbinz/EllipsoidSimilarity.jl")`

### Quick start
First, let's define an ellipsoid as the pair `(m, A)`, where `m` is an `Nx1` vector, `A` is a symmetric, positive definite matrix, and  
`(x - m)' A (x - m) = 1`  
describes an N-dimensional ellipsoid. In other words, `m` is the center of the ellipsoid, and `A` describes its shape, size, and orientation.

The following measures of similarity are available in this package:

1. Compound similarity: `compound_similarity(m1, A1, m2, A2)`
1. "Transformation energy" similarity: `te_similarity(m1, A1, m2, A2)`

## Reference
1. Moshtaghi, M., Havens, T.C., Bezdek, J.C., Park, L., Leckie, C., Rajasegarar, S., Keller, J.M. and Palaniswami, M., "Clustering ellipses for anomaly detection". *Pattern Recognition* 44 (2011) pp. 55-69. [doi:10.1016/j.patcog.2010.07.024](http://dx.doi.org/10.1016/j.patcog.2010.07.024)
