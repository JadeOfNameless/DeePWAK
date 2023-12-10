include("julia/LayerTypes.jl")
include("julia/LayerFns.jl")

epochs = 100000
d = 100
c = 100
η = 0.001
λ = 0.001
batchsize=1024

α = 0.01
β = 0.01
γ = 1
δ = 1

using MLDatasets, ProgressMeter, StatPlots
using ImageInTerminal

dat = MNIST.traindata();
X = vcat(eachslice(dat[1],dims=1)...);

m,n = size(X)

opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

X̃ = zfc(X) |> gpu

loader = Flux.DataLoader((X̃,X),batchsize=batchsize,shuffle=true) |> gpu

md = Dense(m => d, relu) |> gpu
dm = Dense(d => m, relu) |> gpu

state = Flux.setup(opt,m)
l,∇ = Flux.withgradient(loss,m)
Flux.update!(state,m,∇[1])


dd = OneToOne(d,σ) |> gpu

md = encoderlayers(m,d,10,σ)
dm = encoderlayers(d,m,10,σ)

autoencoder = Chain(md,dm) |> gpu

L = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        function loss(f)
            #E = f[1](x)
            #D = 1 ./ (euclidean(E) .+ eps(Float32))
            #G = wak(D)
            #Ê = (G * E')'
            #Ŷ = f[2](Ê)
            Flux.mse(f(x),y)
        end
        
        l = update!(autoencoder,loss,opt)
        return l
    end
end
p = scatter(1:epochs, (log ∘ mean).(L),
            xlabel="epoch",ylabel="logMSE",
            legend=:none)
savefig(p,"plots/autoencoderloss.pdf")

cc = encoderlayers(d,c,5,σ)

modularity = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        function loss(f)
            E = autoencoder[1](x)
            D = 1 ./ (euclidean(E) .+ eps(Float32))
            C = (softmax ∘ cc)(E)
            P = C' * C
            return -softmod(D,P,γ)
        end
            
L = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        loss = f->Flux.mse(f(x),y)
        l = update!(M,loss,opt)
        return l
    end
end

md = Dense(m => m, relu) 
dm = Dense(m => m, relu)
dd = OneToOne(m,σ)
dc = Dense(m => c)
cc = OneToOne(c,σ)

M = DeePWAK(md,dd,dc,cc,dm) |> gpu

L = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        loss = f->mse(f,x)
        l = update!(M,loss,opt)
        return l
    end
end
p = scatter(1:epochs, (log ∘ mean).(L),
            xlabel="epoch",ylabel="logMSE",
            legend=:none)
savefig(p,"plots/autoencoderloss.pdf")


entropy = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        l = update!(M,H,opt)
        return l
    end
end
p = scatter(1:epochs, mean.(entropy),
            xlabel="epoch",ylabel="entropy",
            legend=:none)
savefig(p,"plots/entropy.pdf")

modularity = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        loss = f->1/softmod(f,x,γ)
        l = update!(M,loss,opt)
        return l
    end
end
p = scatter(1:epochs, mean.(modularity),
            xlabel="epoch",ylabel="modularity",
            legend=:none)
savefig(p,"plots/modularity.pdf")

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
