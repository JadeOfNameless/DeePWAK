using Flux, Zygote, Functors

# layer with unique connection for each input and output
struct OneToOne
    weight::AbstractArray   # weight matrix
    bias::AbstractArray  # bias vector
    σ::Function # activation fn
end
@functor OneToOne (weight,bias)

# m:Int -> f:(Float -> Float) -> OneToOne m f
# OneToOne constructor
function OneToOne(d::Integer, σ = identity)
    #weight = randn(Float32, d)
    weight = ones(Float32, d)
    bias = zeros(Float32, d)
    return OneToOne(weight, bias, σ)
end

# ∀ n:Int ∃ m:Int -> [Float m n] -> [Float m n]
# OneToOne application
function (l::OneToOne)(x)
    return l.σ(l.weight .* x .+ l.bias)
end

# ∀ m,n:Int f:(Float -> Float)  -> g:(OneToOne m f) -> A:[Float m n] -> (g A, ∇ g A)
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

struct WeightNetwork
    l::Chain
end
@functor WeightNetwork

# ∀ n:Int ∃ m:Int -> [Float m n] -> [Float m n]
function (m::WeightNetwork)(X::AbstractMatrix)
    return (softmax ∘ m.l)(X)
end

struct WeightedEncoder
    encoder::Chain
    weigher::WeightNetwork
end
@functor WeightedEncoder

# ∀ n:Int ∃ m:Int -> [Float m n] -> [Float m n]
function (M::WeightedEncoder)(X::AbstractMatrix)
    E = M.encoder(X)
    w = M.weigher(E)
    return w .* E
end

struct ClustNetwork
    l::Chain
end
@functor ClustNetwork

# ∀ n:Int ∃ m,c:Int -> [Float m n] -> [Float c n]
function (m::ClustNetwork)(X::AbstractMatrix)
    C = (softmax ∘ m.l)(X)
    return C
end

struct DeePWAK
    encoder::WeightedEncoder
    partitioner::ClustNetwork
    decoder::Chain
    metric::Function
end
@functor DeePWAK (encoder,partitioner,decoder)

# ∀ n:Int ∃ m:Int -> [[Float]]{m,n} -> [[Float]]{m,n}
function (m::DeePWAK)(X::AbstractMatrix)
    #w = (softmax ∘ m.weigher)(X)
    #E = w .* m.encoder(X)
    E = m.encoder(X)
    D = m.metric(E)
    C = m.partitioner(E)
    P = C' * C
    G = wak(D .* P)
    Ehat = (G * E')'
    return m.decoder(Ehat)
end

#∀ n:Int ∃ m,d,c:Int -> DeePWAK m d c -> [Float m n] -> [Float d n]
function getw(M::DeePWAK,X::AbstractMatrix)
    #E = M.encoder(X)
    #w = (softmax ∘ M.weigher)(E)
    return M.encoder.weigher(X)
end

#∀ n:Int ∃ m,d,c:Int -> DeePWAK m d c -> [Float m n] -> [Float d n]
function embedding(M::DeePWAK,X::AbstractMatrix)
    E = M.encoder(X)
    w = (softmax ∘ M.weigher)(E)
    return w .* E
end

#∀ n:Int ∃ m,d,c:Int -> DeePWAK m d c -> [Float m n] -> [Float n n]
function dist(M::DeePWAK,X::AbstractMatrix)
    E = M.encoder(X)
    #return 1 ./ (euclidean(E) .+ eps(Float32))
    return M.metric(E)
end

#∀ n:Int ∃ m,d,c:Int -> DeePWAK m d c -> [Float m n] -> [Float c n]
function clust(M::DeePWAK,X::AbstractMatrix)
    E = embedding(M,X)
    C = (softmax ∘ M.partitioner)(E)
    return C
end

#∀ n:Int ∃ m,d,c:Int -> DeePWAK m d c -> [Float m n] -> [Float n]
function getclusts(M::DeePWAK,X::AbstractMatrix)
    return map(x->x[1],argmax(clust(M,X),dims=1))
end


#∀ n:Int ∃ m,d,c:Int -> DeePWAK m d c -> [Float m n] -> [Float n n]
function g(M::DeePWAK,X::AbstractMatrix)
    D = dist(M,X)
    C = clust(M,X)
    P = C' * C
    return wak(D .* P)
end

# ∀ n:Int ∃ m,d,c:Int -> DeePWAK m d c -> [Float m n] -> [Float m n]
function(M::DeePWAK)(X::AbstractMatrix)
    E = embedding(M,X)
    G = g(M,X)
    Ê = (G * E')'
    X̂ = M.dm(Ê)
    return X̂
end

# ∀ m,d,c:Int -> DeePWAK m d c -> Float
function H(m::DeePWAK)
    H_ω = (mean ∘ H)(m.ω.weight)
    return H_ω
end

# ∀ n:Int ∃ m,d,c:Int -> DeePWAK m d c -> [Float m n] -> Float -> Float
function softmod(M::DeePWAK,X::AbstractMatrix,γ)
    D = dist(M,X)
    C = clust(M,X)
    P = C' * C
    return softmod(D,P,γ)
end

function stats(M::DeePWAK,X::AbstractArray,γ)
    L = mse(M,X)
    w = (softmax ∘ M.dd ∘ M.dm)(X)
    H_w = (mean ∘ H)(w,dims=2)
    calH = softmod(M,X,γ)
    return L,H_w,calH
end

function loss(M::DeePWAK,X::AbstractArray,γ)
    #L = mse(M,X)
    #w = (M.dd ∘ M.dm)(X)
    #H_w = H(w)
    #calH = softmod(M,X,γ)
    L,H_w,calH = stats(M,X,γ)
    return L * H_w / calH#L + α * H_m - β * calH
end

function mse(m::Union{Chain,DeePWAK},X::AbstractMatrix)
    return Flux.mse(X,m(X))
end
    
function stats(M::DeePWAK,X::AbstractArray,Y::AbstractArray,γ)
    L = mse(M,X,Y)
    w = (softmax ∘ M.dd ∘ M.dm)(X)
    H_w = (mean ∘ H)(w,dims=2)
    calH = softmod(M,X,γ)
    return L,H_w,calH
end


function loss(M::DeePWAK,X::AbstractArray,Y::AbstractArray,γ)
    #L = mse(M,X)
    #w = (M.dd ∘ M.dm)(X)
    #H_w = H(w)
    #calH = softmod(M,X,γ)
    L,H_w,calH = stats(M,X,Y,γ)
    return L * H_w / calH#L + α * H_m - β * calH
end

function mse(m::Union{Chain,DeePWAK},X::AbstractMatrix,Y::AbstractMatrix)
    return Flux.mse(Y,m(X))
end
    
function update!(m::Union{Chain,ClustNetwork,WeightNetwork,WeightedEncoder},
                 loss::Function,opt)
    state = Flux.setup(opt,m)
    l,∇ = Flux.withgradient(loss,m)
    Flux.update!(state,m,∇[1])
    return l
end

function update!(M::DeePWAK,loss::Function,opt)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(loss,M)
    Flux.update!(state,M,∇[1])
    return l
end

function train!(M::DeePWAK,X::AbstractMatrix,opt,γ)
    l,entropy,modularity = stats(M,X,γ)
    update!(m,θ->loss(θ,X),opt)
    #entropy = update!(m,H,opt)
    #modularity = update!(m,θ->1/softmod(θ,X,γ),opt)
    return l, entropy, modularity
end

#function trainencoder!(m::DeePWAK,X::AbstractMatrix,opt)
    
    

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
    
