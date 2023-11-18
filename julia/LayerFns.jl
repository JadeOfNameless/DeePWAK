using Flux, LinearAlgebra, CUDA, Distributions

function H(W::AbstractArray)
    W = abs.(W) .+ eps(eltype(W))
    W = W ./ sum(W)
    return -sum(W .* log2.(W))
end

function H(W::AbstractMatrix;dims=1)
    W = abs.(W) .+ eps(eltype(W))
    W = W ./ sum(W,dims=dims)
    return -sum(W .* log2.(W),dims=dims)
end

function wak(G::AbstractArray;dims=1)
    m, n = size(G)
    G = G .* (1 .- I(n))
    G = G ./ (sum(G,dims=dims) .+ eps(eltype(G)))
    return G
end
function wak(G::CuArray;dims=1)
    m, n = size(G)
    G = G .* (1 .- (I(n)|>gpu))
    G = G ./ (sum(G,dims=dims) .+ eps(eltype(G)))
    return G
end

function euclidean(x::CuArray{Float32};dims=1)
    x2 = sum(x .^ 2, dims=dims)
    D = x2' .+ x2 .- 2 * x' * x
    # Numerical stability: possible small negative numbers due to precision errors
    D = sqrt.(max.(D, 0) .+ eps(Float32))  # Ensure no negative values due to numerical errors
    return D
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

function zfc(X::AbstractMatrix;dims=2)
    μ = mean(X,dims=dims);
    X_0 = X .- μ;
    Σ = cov(X_0,dims=dims);
    Λ,U = eigen(Σ);
    W = U * Diagonal(sqrt.(1 ./(Λ .- minimum(Λ) .+ eps(Float32)))) * U';
    X̃ = W * X;
    return X̃
end

function softmod(G::AbstractMatrix,P::AbstractMatrix,γ::Union{Integer,Float32,Float64})
    P_v = P * G
    P_e = P * G'
    e = sum(P_e,dims=2)
    μ = mean(e)
    K = sum(P_v,dims=2)
    calH = 1/(2 * μ) * sum(e .- γ .* K .^ 2 ./(2 * μ))
    return calH
end
