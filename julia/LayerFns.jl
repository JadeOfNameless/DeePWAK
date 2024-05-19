using Flux, LinearAlgebra, CUDA, Distributions

#[Float] -> Float
#Shannon entroy
function H(W::AbstractArray)
    W = abs.(W) .+ eps(eltype(W))
    W = W ./ sum(W)
    return -sum(W .* log2.(W))
end

# [[Float]] -> [Float]
#row/coumn Shannon entropy (default := column)
function H(W::AbstractMatrix;dims=1)
    W = abs.(W) .+ eps(eltype(W))
    W = W ./ sum(W,dims=dims)
    return -sum(W .* log2.(W),dims=dims)
end

# [[Float]] -> [[Float]]
# constructs weighted affinity kernel from adjacency matrix
# sets diagonal to 0
# normalizes rows/columns to sum to 1 (default := columns)
function wak(G::AbstractArray; dims=1)
    G = zerodiag(G)
    G = G ./ (sum(G,dims=dims) .+ eps(eltype(G)))
    return G
end

function pwak(K::AbstractMatrix; dims=1)
    P = K' * K
    return wak(P)
end


# [CuArray] -> [CuArray]
# version of Euclidean distance compatible with Flux's automatic differentiation
# calculates pairwise distance matrix by column (default) or row
function euclidean(x::CuArray{Float32};dims=1)
    x2 = sum(x .^ 2, dims=dims)
    D = x2' .+ x2 .- 2 * x' * x
    # Numerical stability: possible small negative numbers due to precision errors
    D = sqrt.(max.(D, 0) .+ eps(Float32))  # Ensure no negative values due to numerical errors
    return D
end
using CUDA

# [CuArray] -> [CuArray]
# function to calculate cosine similarity matrix
function cossim(x::CuArray{Float32}; dims=1)
    # Normalize each column (or row, depending on 'dims') to unit length
    norms = sqrt.(sum(x .^ 2, dims=dims))
    x_normalized = x ./ norms

    # Compute the cosine similarity matrix
    # For cosine similarity, the matrix multiplication of normalized vectors gives the cosine of angles between vectors
    C = x_normalized' * x_normalized

    # Ensure the diagonal elements are 1 (numerical stability)
    i = CartesianIndex.(1:size(C, 1), 1:size(C, 1))
    C[i] .= 1.0

    return C
end


# [CuArray] -> [CuArray]
# returns reciprocal Euclidean distance matrix
function normeucl(x::AbstractArray)
    return 1 ./ (euclidean(x) .+ eps(Float32))
end

function loss(X,θ,α=0.5,β=0.5,γ=1,δ=1)
    md,dd,dm = θ.layers
    W_dd = dd.weight
    #W_dd = mapreduce(x->x.weight,vcat,dd.layers)

    H_md = (mean ∘ H)(md.weight)
    H_dd = (mean ∘ H)(W_dd)
    #H_E = (H_md + H_dd) / 2
    H_dm = (mean ∘ H)(dm.weight)
    H_E = (H_md + H_dd + H_dm) / 3

    E = θ[1:2](X)
    D = 1 ./ (euclidean(E) .+ eps(Float32))
    D = wak(D)

    H_D = (mean ∘ H)(D)
    #ℍ = softmod(D,P,γ) 
    L = Flux.mse(X,θ[3]((D * E')'))
    return α * H_E + β * H_D + log(L) #- δ * ℍ
    #L = Flux.mse(X,θ(X))
    #return H_E * L
end

# [[Float]] -> [[Float]]
# ZFC whitening
function zfc(X::AbstractMatrix;dims=2)
    μ = mean(X,dims=dims);
    X_0 = X .- μ;
    Σ = cov(X_0,dims=dims);
    Λ,U = eigen(Σ);
    W = U * Diagonal(sqrt.(1 ./(Λ .- minimum(Λ) .+ eps(Float32)))) * U';
    X̃ = W * X;
    return X̃
end

# ∀ n:Int -> [Float n n] -> [Float n n] -> Float -> Float
# modularity for probabilistic cluster assignment
# accepts weighted adjacency matrix G, weighted partition matrix P, resolution γ
function softmod(G::AbstractMatrix,P::AbstractMatrix,γ::Union{Integer,Float32,Float64})
    P_v = P * G
    P_e = P * G'
    e = sum(P_e,dims=2)
    μ = mean(e)
    K = sum(P_v,dims=2)
    calH = 1/(2 * μ) * sum(e .- γ .* K .^ 2 ./(2 * μ))
    return calH
end

function mlp(l,f::Function)
    θ = foldl(l[3:length(l)],
              init=Chain(Dense(l[1] => l[2],f))) do layers,d
        d_0 = size(layers[length(layers)].weight)[1]
        return Chain(layers...,Dense(d_0 => d,f))
    end
end
    

# ∀ m,l:Int f:(Float -> Float) -> Chain [(Dense m m f) l]...
function mlp(m::Integer,l::Integer,σ=relu)
    return Chain(map(_->Dense(m => m, σ),1:l)...)
end

function mlp(m::Integer,d::Integer,l::Integer,s::Integer,f=σ)
    n = maximum([m,d])
    return Chain(Dense(m => s * n,f),
                 map(_->Dense(s * n => s * n, f),1:(l-2))...,
                 Dense(4*n => d,f))
end

# ∀ m,l:Int f:(Float -> Float) -> Chain (Dense m f) l
function mlp4x(m::Integer,d::Integer,l::Integer,σ=σ)
    n = maximum([m,d])
    return Chain(Dense(m => 4 * n),
                 map(_->Dense(4 * n => 4*n, σ),1:(l-2))...,
                 Dense(4*n => d))
end
