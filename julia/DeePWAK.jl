using Pkg
Pkg.activate("leiden")
using Flux, CUDA, Functors, ProgressMeter

mutable struct ModelFamily
    params
    model
end
@functor ModelFamily (model)

mutable struct ModelSpace
    init::Function # ⊤ → ModelFamily
    update!::Function # ModelFamily → ModelFamily
end

mutable struct DeePWAK
    Enc::ModelSpace
    Dist::ModelSpace
    Part::ModelSpace
    enc::Autoencoder
    dist
    part
end
@functor DeePWAK (enc,dist,part)

function deepwak(Θ::ModelSpace,Δ::ModelSpace,Π::ModelSpace)
    return DeePWAK(Θ,Δ,Π,Θ.init(),Δ.init(),Π.init())
end

function update!(ω::DeePWAK)
    ω.Enc.update!(ω.enc)
    ω.Dist.update!(ω.dist)
    ω.Part.update!(ω.part)
end

mutable struct Autoencoder
    encoder
    decoder
    loader::Flux.DataLoader
    opt::Flux.Optimiser
    loss::Function
    epochs::Integer
end
@functor Autoencoder (encoder,decoder)

function encoderlayers(m::Integer,d::Integer,l::Integer,σ=relu)
    #dims = size(X)
    #m = prod(dims[1:(length(dims)-1)])
    s = div(d-m,l)
    𝐝 = m:s:d
    #𝐝 = vcat(𝐝,reverse(𝐝[1:length(𝐝)-1]))
    θ = foldl(𝐝[3:length(𝐝)],
              init=Chain(Dense(𝐝[1] => 𝐝[2],σ))) do layers,d
        d_0 = size(layers[length(layers)].weight)[1]
        return Chain(layers...,Dense(d_0 => d,σ))
    end
end
    
function train!(θ::Autoencoder)
    map(1:θ.epochs) do _
        

function autoencoder(X,d,l,epochs=100,σ=relu,loss=Flux.mse,η=0.01,λ=0,batchsize=1024)
    m,_ = size(X)
    encoder = encoderlayers(m,d,l)
    decoder = encoderlayers(d,m,l)
    loader = Flux.DataLoader((X,X),batchsize=batchsize,shuffle=true)
    opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))
    loss = (X,Y)->loss(Chain(encoder,decoder)(X),Y)
    return Autoencoder(encoder,decoder,loader,opt,loss,epochs)
end
    
