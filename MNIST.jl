include("julia/LayerTypes.jl")
include("julia/LayerFns.jl")

epochs = 100
d = 50
c = 20
η = 0.01
λ = 0.01
batchsize=1024

α = 0.1
β = 0.1
γ = 1
δ = 1

using MLDatasets, ProgressMeter

dat = MNIST.traindata();
X = vcat(eachslice(dat[1],dims=1)...);

m,n = size(X)

opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

X̃ = zfc(X) |> gpu

loader = Flux.DataLoader((X̃,X̃),batchsize=batchsize,shuffle=true) |> gpu

md = Dense(m => d, relu) |> gpu
dm = Dense(d => m, relu) |> gpu

autoencoder = Chain(md,dm)
state = Flux.setup(opt,m)
l,∇ = Flux.withgradient(loss,m)
Flux.update!(state,m,∇[1])


dd = OneToOne(d,σ) |> gpu

md = encoderlayers(m,d,5)
dm = encoderlayers(d,m,5)


L = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        loss = f->Flux.mse(f(x),y)
        l = update!(M,loss,opt)
        return l
    end
end

dd = OneToOne(d,d,σ)
dc = Dense(d => c)
cc = OneToOne(c,c,σ)

M = DeePWAK(md,dd,dc,cc,dm) |> gpu

L = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        loss = f->mse(f,x)
        l = update!(M,loss,opt)
        return l
    end
end

L,entropy,modularity = map(loader) do (x,y)
    train!(m,x,opt,γ)
end

x,y = first(loader)
E = embedding(m,x)
D = dist(m,x)
P = clust(m,x)
G = g(m,x)
X̂ = m(x)
state = Flux.setup(opt,m)

function loss(m,X)
    L_0 = mse(m,X)
    return L_0 + H(m)
end


L = m->mse(m,x)
l,∇ = Flux.withgradient(L,m)
push!(L_m,l)
Flux.update!(state,m,∇[1]);

h,∇ = Flux.withgradient(H,m)
push!(entropy,h)
Flux.update!(state,m,∇[1]);

L = m->-softmod(m,x,γ)
h,∇ = Flux.withgradient(L,m)
push!(modularity,h)
Flux.update!(state,m,∇[1]);

L = m->mse(m,x)
l,∇ = Flux.withgradient(L,m)
Flux.update!(state,m,∇[1])

L_m = Array{Float32}(undef,0)
entropy = Array{Float32}(undef,0)
modularity = Array{Float32}(undef,0)

function loss(X,m,α=α,β=β,γ=γ)
    X̂ = m(X)
    L_0 = log(Flux.mse(X,X̂) + eps(Float32))
    #push!(L_m,L_0)

    H_m = H(m)
    #push!(entropy,H_m)
    
    D = dist(m,X)
    P = clust(m,X)
    H_G = softmod(D,P,γ)
    #push!(modularity,H_G)

    L = α * H_m - β * H_G + L_0
    return L
end

L = @showprogress map(1:epochs) do _
    L = map(loader) do (x,y)
        state = Flux.setup(opt,m)
        l,∇ = Flux.withgradient(m->loss(x,m),m)
        Flux.update!(state,m,∇[1])
        #h = map(l->(mean∘H)(l.weight),θ.layers)
        #l = Flux.mse(X,pred(θ,X))
        return l
    end
    return L
end

θ = Chain(md,dd,dm) |> gpu

function dewak(θ,X)
    E = θ[1:2](X)
    D = 1 ./ (euclidean(E) .+ eps(Float32))
    D = wak(D)

    return 
end

function pred(θ,X)
    E = θ[1:2](X)
    D = 1 ./ (euclidean(E) .+ eps(Float32))
    D = wak(D)

    return θ[3]((D * E')')
end

function loss(X,θ,α=0.10,β=0.10,γ=1,δ=1)
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
    L_0 = Flux.mse(X,θ[3]((D * E')'))
    L = α * H_E + β * H_D + log(L_0+eps(Float32)) #- δ * ℍ
    return L,L_0,H_md,H_dd,H_dm,H_D
    #L = Flux.mse(X,θ(X))
    #return H_E * L
end

L = @showprogress map(1:epochs) do _
    L = map(loader) do (x,y)
        state = Flux.setup(opt,θ)
        l,∇ = Flux.withgradient(θ->loss(x,θ),θ)
        Flux.update!(state,θ,∇[1])
        #h = map(l->(mean∘H)(l.weight),θ.layers)
        #l = Flux.mse(X,pred(θ,X))
        return l
    end
    return L
end
