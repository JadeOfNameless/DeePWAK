include("julia/DeePWAK.jl")
include("julia/LayerFns.jl")
include("julia/SAE.jl")
include("julia/Rmacros.jl")
include("julia/auxfns.jl")

using MLDatasets, StatsPlots, OneHotArrays
using Flux: logitcrossentropy

using JLD2,Tables,CSV
using ImageInTerminal,Images

path = "data/MNIST/"

epochs = 100
batchsize=512

η = 0.001
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

m = 3
d = 27
k = 12
h = 5

α = 0.001

dat = MNIST(split=:train)[:]
target = onehotbatch(dat.targets,0:9)

m_x,m_y,n = size(dat.features)
X = reshape(dat.features[:, :, :], m_x, m_y, 1, n)
colorview(Gray,X[:,:,1,1:2])

loader = Flux.DataLoader((X,target),
                         batchsize=batchsize,
                         shuffle=true)

kern = (3,3)
s = (2,2)
θ_conv = Chain(Conv(kern,1 => 3,relu,stride=s),
               Conv(kern,3 => 6,relu,stride=s),
               Conv(kern,6 => 9,relu,stride=s),
               Conv((2,2),9 => 12,relu))

θ_mlp = Chain(Dense(12 => 6,relu),
              Dense(6 => m,relu))

θ_outer = Chain(θ_conv,
                x->reshape(x,12,:),
                θ_mlp)
               
π_outer = Chain(Dense(m => 5,relu),
                Dense(5 => 10,relu),
                softmax)

M_outer = Chain(θ_outer,π_outer) |> gpu


θ_inner = Chain(Dense(m => 9,relu),
                Dense(9 => d,relu))
ϕ_inner = Chain(Dense(d => 9,relu),
                Dense(9 =>m,relu))
π_inner = Chain(Dense(m => 6,relu),
                Dense(6 => k,relu),
                softmax)
M_inner = DeePWAK(θ_inner,π_inner,ϕ_inner) |> gpu

sae = SAE(m,d) |> gpu
psae = PSAE(sae,π_inner) |> gpu

L_outer = []

train!(M_outer,loader,opt,epochs,logitcrossentropy,L_outer);

state_outer = Flux.state(M_outer) |> cpu;
jldsave(path*"state_outer.jld2";state_outer)
Tables.table(L_outer) |> CSV.write(path*"L_outer.csv")

L_inner = []

function loss_deepwak(M_outer,x,y,loss)
    x = gpu(x)
    y = gpu(y)
    yhat = M_outer(x)
    return m->loss((M_outer[2] ∘ m ∘ M_outer[1])(x),yhat)
end

function loss_SAE(M_outer,α,lossfn,x)
    x = gpu(x)
    yhat = M_outer(x)
    f = M->L1(M,α,x) + L2(M,lossfn,x,yhat)
    return m->lossfn((M_outer[2] ∘ m ∘ M_outer[1])(x),yhat)
end

@showprogress map(1:epochs) do _
    map(loader) do (x,y)
        f = loss_deepwak(M_outer,x,y,logitcrossentropy)
        state = Flux.setup(opt,M_inner)
        l,∇ = Flux.withgradient(f,M_inner)
        Flux.update!(state,M_inner,∇[1])
        push!(L_inner,l)
    end
end

state_inner = Flux.state(M_inner) |> cpu;
jldsave(path*"state_inner.jld2";state_inner)
Tables.table(L_inner) |> CSV.write(path*"L_inner.csv")

L_SAE = []
@showprogress map(1:epochs) do _
    map(loader) do (x,y)
        f = loss_SAE(M_outer,α,logitcrossentropy,x)
        state = Flux.setup(opt,sae)
        l,∇ = Flux.withgradient(f,sae)
        Flux.update!(state,sae,∇[1])
        push!(L_SAE,l)
    end
end

state_SAE = Flux.state(sae) |> cpu;
jldsave(path*"state_SAE.jld2";state_SAE)
Tables.table(L_SAE) |> CSV.write(path*"L_SAE.csv")

L_PSAE = []
@showprogress map(1:epochs) do _
    map(loader) do (x,y)
        f = loss_SAE(M_outer,α,logitcrossentropy,x)
        state = Flux.setup(opt,psae)
        l,∇ = Flux.withgradient(f,psae)
        Flux.update!(state,psae,∇[1])
        push!(L_PSAE,l)
    end
end
state_PSAE = Flux.state(psae) |> cpu;
jldsave(path*"state_PSAE.jld2";state_PSAE)
Tables.table(L_PSAE) |> CSV.write(path*"L_PSAE.csv")

state_outer = JLD2.load(path*"state_outer.jld2","state_outer");
Flux.loadmodel!(M_outer,state_outer)
state_inner = JLD2.load(path*"state_inner.jld2","state_inner");
Flux.loadmodel!(M_inner,state_inner)
state_SAE = JLD2.load(path*"state_SAE.jld2","state_SAE");
Flux.loadmodel!(sae,state_SAE)
state_PSAE = JLD2.load(path*"state_PSAE.jld2","state_PSAE");
Flux.loadmodel!(psae,state_PSAE)

x,y = first(loader)
x = gpu(x)
y = gpu(y)
labels = unhot(y)'

E_outer = M_outer[1](x)
E_inner = M_inner.encoder(E_outer)
E_SAE = encode(sae,E_outer)
E_PSAE = encode(psae,E_outer)

K_outer = M_outer(x)
K_inner = M_inner.partitioner(E_outer)
K_SAE = cluster(psae,E_outer)

clust_outer = unhot(K_outer)
clust_inner = unhot(K_inner)
clust_SAE = unhot(K_SAE)

function hm4(out,X,kwargs...)
    heatmap(out * "K_outer.pdf",X',
            (median ∘ not0)(X), split=clust_outer',kwargs...)
    heatmap(out * "K_inner.pdf",X',
            (median ∘ not0)(X), split=clust_inner',kwargs...)
    heatmap(out * "K_SAE.pdf",X',
            (median ∘ not0)(X), split=clust_SAE',kwargs...)
    heatmap(out * "_target.pdf",X',
            (median ∘ not0)(X), split=labels,kwargs...)
end

hm4(path*"E_outer",E_outer)   
hm4(path*"E_inner",E_inner)   
hm4(path*"E_SAE",E_SAE)   
hm4(path*"E_PSAE",E_PSAE)   

hm4(path*"K_outer",K_outer)   
hm4(path*"K_inner",K_inner)   
hm4(path*"K_SAE",K_SAE)   

clusthyper(clust_outer', clust_inner',
           path*"enrichment_inner")
clusthyper(labels, clust_outer',
           path*"enrichment_outer")

clusthyper(clust_outer',clust_SAE,
           path*"enrichment_SAE")
