using Flux, Zygote, Functors

# layer with unique connection for each input and output
struct OneToOne
    weight::AbstractArray   # weight matrix
    bias::AbstractArray  # bias vector
    σ::Function # activation fn
end
@functor OneToOne (weight,bias)

function OneToOne(d::Integer, σ = identity)
    #weight = randn(Float32, d)
    weight = ones(Float32, d)
    bias = zeros(Float32, d)
    return OneToOne(weight, bias, σ)
end

function (l::OneToOne)(x)
    return l.σ(l.weight .* x .+ l.bias)
end

function Zygote.pullback(l::OneToOne, x::AbstractArray)
    y = l.weight .* x .+ l.bias
    z = l.σ(y)
    
    function B(Δ)
        Δ = Δ .* gradient(layer.σ, y)[1]  # chain rule applied
        ∇_w = sum(Δ .* x, dims=2)[:]  # Gradient w.r.t. weight
        ∇_b = sum(Δ, dims=2)[:]       # Gradient w.r.t. bias
        return (weight=∇_w, bias=∇_b), nothing
    end
    
    return z, B
end

mutable struct Softmax
    weight::AbstractArray
    bias::AbstractArray
    σ::Function
end
@functor Softmax (weight,bias)

function Softmax(d::Integer)
    weight = randn(Float32,d)
    bias = randn(Float32,d)
    return Softmax(weight,bias)
end

function (θ::Softmax)(x)
    return θ.σ(θ.weight .* x .+ θ.bias)
end

#function Zygote.pullback(θ::Softmax,x::AbstractArray)
#    y = θ(x)
#    function back(Δ)
#        Δ = Δ .* gradient(θ,y)[1]
#        ∇_w = sum(Δ .* x,dims=2)

mutable struct SoftNN
    σ::Function
end
#
#function (δ::SoftNN)(x)
#    m,n = size(x)
#    D = euclidean(x)
#    D = D .* (1 .- I(n))

struct DEWAK
    θ_e::Dense
    ω::OneToOne
    κ::Parallel
    θ_d::Dense
end
@functor DEWAK

function (m::DEWAK)(X::AbstractMatrix)
    E = (m.ω ∘ m.θ_e)(X)
    D = 1 ./ (euclidean(E) .+ eps(Float32))
    D = wak(D)
    X̂ = m.θ_d((D * E')')
    return X̂
end

struct DeePWAK
    θ
    ω#::OneToOne
    ϕ
    #υ::OneToOne
    ψ
end
@functor DeePWAK

function embedding(m::DeePWAK,X::AbstractMatrix)
    return (m.ω ∘ m.θ)(X)
end

function dist(m::DeePWAK,X::AbstractMatrix)
    E = embedding(m,X)
    return 1 ./ (euclidean(E) .+ eps(Float32))
end

function clust(m::DeePWAK,X::AbstractMatrix)
    E = embedding(m,X)
    C = (softmax ∘ m.ϕ)(E)
    return C' * C
end

function g(m::DeePWAK,X::AbstractMatrix)
    D = dist(m,X)
    P = clust(m,X)
    return wak(D .* P)
end

function(m::DeePWAK)(X::AbstractMatrix)
    E = embedding(m,X)
    G = g(m,X)
    Ê = (G * E')'
    X̂ = m.ψ(Ê)
    return X̂
end

function H(m::DeePWAK)
    H_ω = (mean ∘ H)(m.ω.weight)
    return H_ω
end

function softmod(m::DeePWAK,X::AbstractMatrix,γ)
    D = dist(m,X)
    P = clust(m,X)
    return softmod(D,P,γ)
end

function loss(m::DeePWAK,X::AbstractArray,α,β,γ)
    L = mse(m,X)
    H_m = H(m)
    calH = softmod(m,X,γ)
    return L + α * H_m - β * calH
end

function mse(m::Union{Chain,DeePWAK},X::AbstractMatrix)
    return Flux.mse(X,m(X))
end
    
function update!(m::Union{Chain,DeePWAK},loss::Function,opt)
    state = Flux.setup(opt,m)
    l,∇ = Flux.withgradient(loss,m)
    Flux.update!(state,m,∇[1])
    return l
end

function train!(m::DeePWAK,X::AbstractMatrix,opt,γ)
    l = update!(m,θ->mse(θ,X),opt)
    entropy = update!(m,H,opt)
    modularity = update!(m,θ->1/softmod(θ,X,γ),opt)
    return l, entropy, modularity
end

function trainencoder!(m::DeePWAK,X::AbstractMatrix,opt)
    
    

function encoderlayers(m::Integer,d::Integer,l::Integer,σ=relu)
    #dims = size(X)
    #m = prod(dims[1:(length(dims)-1)])
    s = div(d-m,l)
    𝐝 = collect(m:s:d)
    𝐝[l+1] = d
    #𝐝 = vcat(𝐝,reverse(𝐝[1:length(𝐝)-1]))
    θ = foldl(𝐝[3:length(𝐝)],
              init=Chain(Dense(𝐝[1] => 𝐝[2],σ))) do layers,d
        d_0 = size(layers[length(layers)].weight)[1]
        return Chain(layers...,Dense(d_0 => d,σ))
    end
end
    
