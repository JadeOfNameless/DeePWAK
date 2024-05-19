# workaround for NearestNeighbors.jl not excluding identities 
using Distances, LinearAlgebra

# inverted euclidean distance with diagonal set to 0
function invdist(E)
    1 ./ (pairwise(Euclidean(),E,E) + (I*Inf));
end

#ordering of neighbors
function neighbors(E)
    m,n = size(E)
    D = invdist(E)
    G = sortperm(D,dims=1,rev=true) .% n
    G[G .== 0] .= n
    return mapslices(invperm,G,dims=1)
end

function knn(E,k)
    G = neighbors(E)
    G = G .<= k
    return Matrix{Float32}(G)
end
