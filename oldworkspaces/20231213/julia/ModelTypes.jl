using Flux, CUDA, Functors, ProgressMeter

struct DAEWAK
    encoder::Chain
    decoder::Chain
end
@functor DAEWAK

# ∃ m,d:Int -> [Float m] -> [Float d]
function (M::DAEWAK)(X::AbstractArray)
    E = M.encoder(X)
    G = (wak ∘ normeucl)(E)
    Ehat = (G * E')'
    return M.decoder(Ehat)
end

struct ClustNetwork
    l::Chain
    γ::Real
end
@functor ClustNetwork (l)

# ∃ m,c:Int -> [Float m] -> [Float c]
function (M::ClustNetwork)(X::AbstractMatrix)
    C = (softmax ∘ M.l)(X)
    return C
end

struct DeePWAK
    autoencoder::DAEWAK
    partitioner::ClustNetwork
end
@functor DeePWAK

# ∀ n:Int ∃ m:Int -> [[Float]]{m,n} -> [[Float]]{m,n}
function (M::DeePWAK)(X::AbstractMatrix)
    #w = (softmax ∘ m.weigher)(X)
    #E = w .* m.encoder(X)
    E = M.autoencoder.encoder(X)
    D = normeucl(E)
    C = M.partitioner(E)
    P = C' * C
    G = wak(D .* P)
    Ehat = (G * E')'
    return M.autoencoder.decoder(Ehat)
end

function loss(f::DAEWAK,X::AbstractMatrix)
    return Flux.mse(f(X),X)
end

# ∀ n,m,d,c:Int -> Autoencoder m d -> ClustNetwork d c -> [Float m n] -> Float 
function loss(f::DAEWAK,π::ClustNetwork,X)
    E = f.encoder(X)
    D = normeucl(E)

    C = π(X)
    P = C' * C
    G = wak(D) .* P

    Ehat = (G * E')'
    Flux.mse(f.decoder(Ehat),X)
end

function loss(M::DeePWAK,X::AbstractMatrix)
    return loss(M.autoencoder,M.partitioner,X)
end

# ∀ n,m,d,c:Int -> ClustNetwork d c -> Autoencoder m d -> [Float m n] -> Float
function modularity(f::ClustNetwork,autoencoder::Chain,X::AbstractMatrix)
    E = autoencoder(X)
    D = normeucl(E)
    G = wak(D)

    C = f(X)
    P = C' * C
    return -softmod(G,P,f.γ)
end

function modularity(f::ClustNetwork,autoencoder::DAEWAK,X::AbstractMatrix)
    modularity(f,autoencoder.encoder,X)
end

function modularity(M::DeePWAK,X::AbstractMatrix)
    return modularity(M.partitioner,M.autoencoder,X)
end

function update!(M,loss::Function,opt)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(loss,M)
    Flux.update!(state,M,∇[1])
    return l
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
    
