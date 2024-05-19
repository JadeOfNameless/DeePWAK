using Pkg
Pkg.activate("julia/leiden")
#include("julia/DeePWAK.jl")
include("julia/fns.jl")

using Flux, ProgressMeter, CUDA, LinearAlgebra
using Distributions, Distances
using Zygote, Functors
using Zygote: @adjoint
using CSV, DataFrames, MLDatasets

# layer with unique connection for each input and output
mutable struct OneToOne
    weight::AbstractArray   # weight matrix
    bias::AbstractArray  # bias vector
    σ::Function # activation fn
end
@functor OneToOne (weight,bias)

mutable struct Softmax
    weight::AbstractArray
    bias::AbstractArray
end
@functor Softmax (weight,bias)

# Constructor for initializing the layer with a diagonal matrix
function OneToOne(in::Integer, out::Integer, σ = identity)
    # initializing with non-zero diagonal values
    weight = Flux.Zeros(out, in) .+ Diagonal(fill(0.01, min(in, out))) 
    bias = zeros(out)
    return OneToOne(weight, bias, σ)
end

# Forward pass for the custom layer
(layer::OneToOne)(x) = layer.σ(Diagonal(layer.weight) .* x .+ layer.bias)

function Zygote.pullback(layer::OneToOne,x::AbstractArray)
    y = layer(x)
    function back(Δ)
        # Here you ensure only diagonal elements are considered
        # Adjust this gradient calculation as necessary for your use case
        ddiag = diag(Δ * x')
        return y,ddiag
    end
    
    return y, back
end
    
function loss(X,θ)
    W_e = θ.layers[1].weight
    H_e = (mean ∘ H)(W_e)

    W_d = θ.layers[2].weight
    H_d = (mean ∘ H)(W_d)
    L = Flux.mse(X,θ(X))
    return ((H_e + H_d) / 2) + L
end

function H(W;dims=1)
    W = abs.(W) .+ eps(eltype(W))
    W = W ./ sum(W,dims=dims)
    return -sum(W .* log2.(W),dims=dims)
end

function wak(G::AbstractArray)
    m, n = size(G)
    G = G .* (1 .- I(n))
    G = G ./ (sum(G,dims=1) .+ eps(eltype(G)))
    return G
end
function wak(G::CuArray)
    m, n = size(G)
    G = G .* (1 .- (I(n)|>gpu))
    G = G ./ (sum(G,dims=1) .+ eps(eltype(G)))
    return G
end

function euclidean(x::CuArray{Float32})
    x2 = sum(x .^ 2, dims=1)
    D = x2' .+ x2 .- 2 * x' * x
    # Numerical stability: possible small negative numbers due to precision errors
    D = sqrt.(max.(D, 0) .+ eps(Float32))  # Ensure no negative values due to numerical errors
    return D
end


function loss(X,θ)
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
    L = Flux.mse(X,θ[3]((D * E')'))
    return ((H_E + H_D) / 2) * L
    #L = Flux.mse(X,θ(X))
    #return H_E * L
end



function OneToOne(in::Integer, out::Integer, σ = identity)
    weight = randn(Float32, out)
    bias = zeros(Float32, out)
    return OneToOne(weight, bias, σ)
end

function (layer::OneToOne)(x)
    return layer.σ(layer.weight .* x .+ layer.bias)
end

function Zygote.pullback(layer::OneToOne, x::AbstractArray)
    pre_activation = layer.weight .* x .+ layer.bias
    y = layer.σ(pre_activation)
    
    function OneToOne_pb(Δ)
        Δ = Δ .* gradient(layer.σ, pre_activation)[1]  # chain rule applied
        diag_grad = sum(Δ .* x, dims=2)[:]  # Gradient w.r.t. weight
        bias_grad = sum(Δ, dims=2)[:]       # Gradient w.r.t. bias
        return (weight=diag_grad, bias=bias_grad), nothing
    end
    
    return y, OneToOne_pb
end

dat = MNIST.traindata();
X = vcat(eachslice(dat[1],dims=1)...);

#dat = MNIST()
#X = vcat(eachslice(dat.features,dims=1)...);

m,n = size(X)

epochs = 1000
d = 50
η = 0.01
λ = 0.01
batchsize=1024

opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

μ = mean(X,dims=2);
X_0 = X .- μ;
Σ = cov(X_0,dims=2);
Λ,U = eigen(Σ);
W = U * Diagonal(sqrt.(1 ./(Λ .- minimum(Λ) .+ eps(Float32)))) * U';
X̃ = W * X |> gpu;

loader = Flux.DataLoader((X̃,X̃),batchsize=batchsize,shuffle=true) |> gpu

θ_md = Dense(m => d)
θ_dd = OneToOne(d,d)
θ_dm = Dense(d => m)

θ = Chain(θ_md,θ_dd,θ_dm) |> gpu

x,y = first(loader)
state = Flux.setup(opt,θ);
L,∇ = Flux.withgradient(x->loss(x,θ),x)

# Forward pass
y, back = Zygote.pullback(x->loss(x,θ), x)

# Backward pass
Δ = randn(Float32,size(y))|>gpu
grads = back(Δ)

𝐋 = []
@showprogress map(1:epochs) do _
    L = map(loader) do (x,y)
        state = Flux.setup(opt,θ)
        L,∇ = Flux.withgradient(θ->loss(x,θ),θ)
        Flux.update!(state,θ,∇[1])
        return L
    end
    push!(𝐋,L)
end


d = 2
l = 2

X = X |> gpu
α = autoencoder(X,d,l;λ=0.01)

δ = distenc(X,α,2*d-2;λ=0.01)

x,y = first(δ.loader)
E = α.encoder(x)

function loss(m)
    δ.loss(α.decoder(E*m(E)),y)
end
loss(δ)

∇ = Flux.gradient(loss,δ)
Flux.update!(state_δ,δ,∇[1]);

state = Flux.setup(α.opt, α);
L,∇ = Flux.withgradient(m->α.loss(m(x),y),α);
Flux.update!(state,α,∇[1]);

@showprogress for _ in 1:α.epochs
     map(α.loader) do (x,y)
        state = Flux.setup(α.opt, α);
        ∇ = Flux.gradient(α) do m
            α.loss(m(x),y)
        end
        Flux.update!(state,α,∇[1])
    end
end

δ =encoderlayers(d*2,1,d*2-2)|>gpu

@showprogress for _ in 1:δ.epochs
     map(δ.loader) do (x,y)
        E = α.encoder(x)

        state_δ = Flux.setup(δ.opt,δ)

        function loss(m)
            δ.loss(α.decoder(m(E)(E')'),y)
        end
         
        ∇ = Flux.gradient(loss,δ)
        Flux.update!(state_δ,δ,∇[1]);
    end
end

ψ = distenc(X,α,2*d-2;σ=σ,λ=0.01)

