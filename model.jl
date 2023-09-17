using Pkg
Pkg.activate("leiden")

using CSV, DataFrames, Tensors, CategoricalArrays, Distributions,
    Flux, LinearAlgebra, SparseArrays, Distances, ProgressMeter, CUDA
using Leiden
using ThreadTools
using JLD2

include("julia/fns.jl")

batchsize = 1024
epochs = 100
η = 0.01
λ = 0

frac = 10
b = 32

𝐝 = 1:32
𝐤 = 1:128
𝛄 = rand(Uniform(0.1,2),128);
𝐬 = 128

dat = (DataFrame ∘ CSV.File)("data/z_dat.csv",normalizenames=true);
dat = (scaledat ∘ Matrix)(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);

n,m = size(dat)
n_Y = Integer(2^round(log2(n) - log2(frac)))
n_X = n - n_Y

tmp = map(_->sampledat(dat',n_Y),1:b);

test = map(x->x[1],tmp);
train = map(x->x[2],tmp);

X = dat' |> gpu;

layers = map(d->Chain(Dense(m => d,relu),Dense(d => m)),𝐝);

Θ = Parallel(vcat,layers...) |> gpu;

loader = Flux.DataLoader((X,repeat(X,outer=(length(𝐝),1))),batchsize=batchsize,shuffle=true);

opt = Flux.Optimiser(Flux.AdamW(η), Flux.WeightDecay(λ))

@showprogress for _ in 1:epochs
    loss = (X,Y)->Flux.mse(Θ(X),Y)
    Flux.train!(loss,Flux.params(Θ),loader,opt)
end

Θ_s = Flux.state(Θ|>cpu);
jldsave("models/autoencoder.jld2",Θ_s=Θ_s)

Θ_s = JLD2.load("models/autoencoder.jld2", "Θ_s");
Flux.loadmodel!(Θ,Θ_s);

𝐄 = embedding(Θ,X);
𝐃 = tmap(distmat,𝐄);

𝐊 = tmap(𝐃) do D
    K = sortperm(D,dims=1,rev=true) .% n
    K[K .== 0] .= n
    return K
end;

𝐊_decomp = tmap(𝐊) do K
    map(1:maximum(𝐤)) do k
        sparse(1:n,K[:,k],1,n,n)
    end
end;

𝐆 = tmap(𝐊_decomp) do K
    map(𝐤) do k
        foldl(+,K[1:k])
    end
end;


L_dk = mapreduce(hcat,Θ.layers,𝐄,𝐃,𝐆) do θ,E,D,Gs
    E = E |> gpu
    D = D |> gpu
    map(Gs) do G
        G = G|>gpu
        return 𝕃(X,θ[2],E,D,G)
    end
end

Tables.table(L_dk) |> CSV.write("data/MSEdk.csv")

k,d = argmin(L_dk).I 

θ = Θ.layers[d];
E = 𝐄[d];
D = 𝐃[d];
G = 𝐆[d][k];
M = wak(D .* G)

Tables.table(E) |> CSV.write("data/E.csv")
Tables.table(G .* D) |> CSV.write("data/G.csv")
Tables.table(M) |> CSV.write("data/M.csv")

𝐂 = tmap(γ->Leiden.leiden(M,"mod++", γ = γ),𝛄);
(Tables.table ∘ hcat)(𝐂...) |> CSV.write("data/clusters.csv")

𝐏 = map(partitionmat,𝐂)

E = E |> gpu
D = D |> gpu

Lᵧ = map(𝐏) do P
    P = P |> gpu
    return 𝕃(X,θ[2],E,D,P)
end
argmin(Lᵧ)
nclust = map(maximum,𝐂)
C_df = DataFrame(gamma=𝛄,nclust=nclust,MSE=Lᵧ)
C_df |> CSV.write("data/MSEgamma.csv")

G = G |> gpu
L_γs = mapreduce(hcat,𝐏) do P
    P = P |> gpu
    _,L = diffuse(X,θ[2],E,D,G,P,𝐬)
    return L
end
s,γ = argmin(L_γs).I
Tables.table(L_γs) |> CSV.write("data/MSEgamma_s.csv")

P = 𝐏[γ]
Tables.table(P) |> CSV.write("data/P.csv")

𝐆_γ = tmapmap(G->G .* P,𝐆);
L_dkγ = mapreduce(hcat,Θ.layers,𝐄,𝐃,𝐆_γ) do θ,E,D,Gs
    E = E |> gpu
    D = D |> gpu
    map(Gs) do G
        G = G|>gpu
        return 𝕃(X,θ[2],E,D,G)
    end
end

argmin(L_dkγ).I

L_ks = mapreduce(hcat,𝐆[d]) do G
    G = G |> gpu
    _,L = diffuse(X,θ[2],E,D,G,1,𝐬)
    return L
end
Tables.table(L_ks) |> CSV.write("data/MSEks.csv")
