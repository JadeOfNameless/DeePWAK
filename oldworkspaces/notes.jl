using Flux, CUDA, LinearAlgebra, ProgressMeter, TensorOperations,
    TensorCast

mutable struct ParamSpace
end

mutable struct Hyperparams
end

mutable struct Model
    #∀ 𝐗,𝐘:ParamSpace -> 𝐗 -> 𝐘
end

mutable struct ModelFamily
    #∀ 𝐗,𝐘:ParamSpace -> Hyperparams -> Model(𝐗,𝐘)
end

mutable struct Loss
    #∀ Φ:ModelFamily θ:Hyperparams 𝐗:ParamSpace -> Φ(θ) -> 𝐗 -> ℝ

mutable struct FluxModel
    epochs::Int
    opt::Flux.Optimizer
    loader::Flux.DataLoader
    X::AbstractArray{Union{Float32,Float64}}
    δ::Chain
    L
end
    

mutable struct UpdateRule
    #∀ (Φ:ModelFamily θ,θ':Hyperparams) -> Φ(θ) -> Φ(θ')
end

mutable struct ModelSpace
    ϕ::ModelFamily
    υ::UpdateRule
end

mutable struct DeePWAK
    #EncoderSpace 
    Θ::ModelFamily
    #DistanceSpace 
    Δ::ModelFamily
    #PartitionSpace 
    Π::ModelFamily
    #DecoderSpace 
    Θ_inv::ModelFamily
    #Loss
    L::Model #X -> X̂ -> ℝ
end

mutable struct DeepWAKer
    Λ::DeePWAK
    θ::Model #X -> E
    δ::Model #E -> D
    π::Model #D -> P
    θ_inv::Model #E -> X̂
    L::Model
end

function wak(G)
    #Matrix -> Matrix
    #returns G with row sums normalized to 1
    G = G .* (1 .- I)
    W = sum(G,dims=2)
    K = G ./ W
    K[isnan.(K)] .= 0
    return K
end

function dewak(X,G)
    M = wak(G)
    return Flux.mse(X,M * X)
end

function γ(D)
    C = mapslices(softmax,D)
    P = C * C'
    return P
end

function dist(Δ::DeePWAK,X)
    mapslices(X,1

function predict(Δ::DeePWAK,X)
    D = dist(Δ,X)

function L(Δ::DeePWAK,X,Y)
    Ŷ = predict(Δ,X)
