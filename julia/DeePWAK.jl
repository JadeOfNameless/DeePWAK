using Pkg
Pkg.activate("leiden")
using Flux, CUDA, Functors, ProgressMeter

macro train!(Model)
    quote
        function train!(δ::$Model)
            @showprogress for _ in 1:δ.epochs
                for l in δ.loader
                    x,y = l
                    state = Flux.setup(δ.opt, α);
                    ∇ = Flux.gradient(δ) do m
                        δ.loss(m(x),y)
                    end
                    Flux.update!(state,δ,∇[1])
                end
            end
        end
    end
end

mutable struct ModelFamily
    params
    model
end
@functor ModelFamily (model,)

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

function deepwak(A::ModelSpace,Δ::ModelSpace,Π::ModelSpace)
    return DeePWAK(A,Δ,Π,A.init(),Δ.init(),Π.init())
end

function update!(κ::DeePWAK)
    κ.Enc.update!(κ.enc)
    κ.Dist.update!(κ.dist)
    κ.Part.update!(κ.part)
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
@train! Autoencoder

function (α::Autoencoder)(X)
    return Chain(α.encoder,α.decoder)(X)
end


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
    
function train!(α::Autoencoder)
    @showprogress for _ in 1:α.epochs
        for l in α.loader
            x,y = l
            state = Flux.setup(α.opt, α);
            ∇ = Flux.gradient(α) do m
                α.loss(m(x),y)
            end
            Flux.update!(state,α,∇[1])
        end
    end
end

function autoencoder(X,d,l;epochs=100,σ=relu,loss=Flux.mse,η=0.01,λ=0,batchsize=1024)
    m,_ = size(X)
    encoder = encoderlayers(m,d,l)
    decoder = encoderlayers(d,m,l)
    loader = Flux.DataLoader((X,X),batchsize=batchsize,shuffle=true) |> gpu
    opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))
    α = Autoencoder(encoder,decoder,loader,opt,loss,epochs) |> gpu

    train!(α)
    return α
end
    
mutable struct DistEnc
    encoder
    loader
    opt
    loss
    epochs
end
@functor DistEnc (encoder,)
@train! DistEnc

function (δ::DistEnc)(E)
    m,n = size(E)
    pairs = genpairs(E)
    D = δ.encoder(pairs)
    D = reshape(D,(n,n)) .* (1 .- (I(n)|>gpu))
    D = D ./ (sum(D,dims=1) .+ eps(eltype(D)))
    return D
end

function train!(α::Autoencoder,δ::DistEnc)
    loss(m) = function(m)
    map(1:δ.epochs) do _
        map(α.loader) do (x,y)
            E = α.encoder(x)

function distenc(X::AbstractMatrix,α::Autoencoder,l::Integer;
                 epochs=100,σ=relu,loss=Flux.mse,η=0.01,λ=0,batchsize=1024)
    m,n = size(X)
    E = α.encoder(X)
    d,n = size(E)
    encoder = encoderlayers(d*2,1,l)
    loader = Flux.DataLoader((X,X),batchsize=batchsize,shuffle=true) |> gpu
    opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))
    δ = DistEnc(encoder,loader,opt,loss,epochs) |> gpu

    #train!(α,δ)
    return δ
end
    
function genpairs(X::CuArray)
    m, n = size(X)
    X_1 = repeat(X, 1, n)
    X_2 = reshape(repeat(X, n), m, n^2)
    return vcat(X_1, X_2)
end

function L_δ(α::Autoencoder,X,Y,D)
    m,n = size(Y)
    E = α.encoder(X)
    D = reshape(D,(n,n)) .* (1 .- (I(n)|>gpu))
    D = D ./ (sum(D,dims=1) .+ eps(eltype(D)))
    Ŷ = α.decoder(E * D)
    Flux.mse(Ŷ,Y)
end
