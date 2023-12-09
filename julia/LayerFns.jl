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

function zerodiag(G::AbstractArray)
    m, n = size(G)
    G = G .* (1 .- I(n))
    return G
end

# [CuArray] -> [CuArray]
# workaround for moving identity matrix to GPU 
function zerodiag(G::CuArray)
    m, n = size(G)
    G = G .* (1 .- I(n) |> gpu)
    return G
end

function neighborcutoff(G::AbstractArray; ϵ=0.0001)
    M = G .> ϵ
    return G .* M
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

# ∀ m,l:Int f:(Float -> Float) -> Chain [(Dense m m f) l]...
function mlp(m::Integer,l::Integer,σ=relu)
    return Chain(map(_->Dense(m => m, σ),1:l)...)
end

# ∀ m,l:Int f:(Float -> Float) -> Chain (Dense m f) l
function mlp4x(m::Integer,d::Integer,l::Integer,σ=σ)
    n = maximum([m,d])
    return Chain(Dense(m => 4 * n),
                 map(_->Dense(4 * n => 4*n, σ),1:(l-2))...,
                 Dense(4*n => d))
end

# ∀ A:Type m,n:Int -> [A m n] -> k:Int -> ([A m k],[A m n-k])
function sampledat(X::AbstractArray,k)
    _,n = size(X)
    sel = sample(1:n,k,replace=false)
    test = X[:,sel]
    train = X[:,Not(sel)]
    return test,train
end

# ∀ m,n:Int -> [Float m n] -> [Float m n]
#scales each column (default) or row to [-1,1]
function scaledat(X::AbstractArray,dims=1)
    Y = X ./ maximum(abs.(X),dims=dims)
    Y[isnan.(Y)] .= 0
    return Y
end

# ∀ n,m,d,c:Int -> Autoencoder m d -> ClustNetwork d c -> [Float m n] -> Float 
function loss(f,π::ClustNetwork,X)
    E = f[1](X)
    D = normeucl(E)

    C = π(X)
    P = C' * C
    G = wak(D) .* P

    Ehat = (G * E')'
    Flux.mse(f[2](Ehat),X)
end

# ∀ n,m,d,c:Int -> ClustNetwork d c -> Autoencoder m d -> [Float m n] -> Float
function modularity(f::ClustNetwork,autoencoder,X)
    E = autoencoder[1](X)
    D = normeucl(E)
    G = wak(D)

    C = f(X)
    P = C' * C
    return -softmod(G,P,γ)
end
