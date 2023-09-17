using Pkg
Pkg.activate("leiden")

using CSV, DataFrames, Tensors, CategoricalArrays, Distributions,
    Flux, LinearAlgebra, SparseArrays, Distances, ProgressMeter, CUDA
using Leiden
using ThreadTools
using JLD2

function zcat(args...)
    cat(args...,dims=3)
end

function mapmap(f,args...)
    map((args...)->map(f,args...),args...)
end

function tmapmap(f,args...)
    tmap((args...)->map(f,args...),args...)
end

function tmapreduce(f,rf,args...)
    rf(tmap(f,args...)...)
end

function scaledat(X::AbstractArray,dims=1)
    Y = X ./ maximum(abs.(X),dims=dims)
    Y[isnan.(Y)] .= 0
    return Y
end

function sampledat(X::AbstractArray,k)
    _,n = size(X)
    sel = sample(1:n,k,replace=false)
    test = X[:,sel]
    train = X[:,Not(sel)]
    return test,train
end

function wak(G)
    #Matrix -> Matrix
    #returns G with row sums normalized to 1
    W = sum(G,dims=2)
    K = G ./ W
    K[isnan.(K)] .= 0
    return K
end

function ehat(E,D,G)
    (wak(G .* D) * E')'
end

function 𝕃(X,θ,E,D,G)
    Flux.mse(X,(θ ∘ ehat)(E,D,G))
end

function partitionmat(C)
    (sum ∘ map)(1:maximum(C)) do c
        x = C .== c
        return x * x'
    end
end

function diffuse(X,θ,E,D,G,P,d)
    M = P .* G
    M = wak(M .* D)
    foldl(1:d,init=(M,[])) do (M⁺,L),_
        M⁺ = M⁺ * M
        L⁺ = Flux.mse(X,θ((M⁺ * E')'))
        L = vcat(L,L⁺)
        return M⁺,L
    end
end

frac = 10
b = 32
bfŋ = 1:32
𝐤 = 1:128
𝛄 = rand(Uniform(0.1,3),128);
𝐝 = 128

dat = (DataFrame ∘ CSV.File)("z_dat.csv",normalizenames=true);
dat = (scaledat ∘ Matrix)(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);

n,m = size(dat)
n_Y = Integer(2^round(log2(n) - log2(frac)))
n_X = n - n_Y

tmp = map(_->sampledat(dat',n_Y),1:b);

test = map(x->x[1],tmp);
train = map(x->x[2],tmp);

X = train |> gpu;
Y = test |> gpu;

#layers = map(ŋ->Chain(Dense(m => ŋ,relu),Dense(ŋ => m)),bfŋ);
𝚯 = map(1:b) do _
    l = map(ŋ->Chain(Dense(m => ŋ,relu),Dense(ŋ => m)),bfŋ)
    return Parallel(vcat,l...) |> gpu
end;

loader = map(X) do X_b
    Flux.DataLoader((X_b,repeat(X_b,outer=(length(bfŋ),1))), batchsize=1024, shuffle=true)
end;

opt = Flux.Optimiser(Flux.AdamW(0.01), Flux.WeightDecay(0))

@showprogress for _ in 1:100
    map(𝚯,loader) do Θ,l
        loss = (X,Y)->Flux.mse(Θ(X),Y)
        Flux.train!(loss,Flux.params(Θ),l,opt)
    end
end

𝚯_s = map(Flux.state,𝚯);
jldsave("bfTheta.jld2";𝚯_s)
𝚯_s = JLD2.load("bfTheta.jld2", "𝚯_s");
Flux.loadmodel!(𝚯,𝚯_s)

function embedding(Θ,X)
    map(Θ.layers) do θ
        θ[1](X)|>cpu
    end
end

function distmat(E)
    1 ./ (pairwise(Euclidean(),E,E) + (I*Inf));
end

function perm(D,n)
    K = sortperm(D,dims=1,rev=true) .% n
    K[K .== 0] .= n
    return K
end

function adjmat(K,𝐤,n)
    G = map(1:maximum(𝐤)) do k
        sparse(1:n,K[:,k],1,n,n)
    end
    G = map(𝐤) do k
        foldl(+,G[1:k])
    end
    return G
end
        
function 𝕃_ŋk(X,θ,E,D,𝐆)
    E = E |> gpu
    D = D |> gpu
    map(𝐆) do G
        G = G |> gpu
        return 𝕃(X,θ[2],E,D,G)
    end
end

𝐄_X = map(embedding,𝚯,X);
𝐄_Y = map(embedding,𝚯,Y);

trees = mapmap(KDTree,𝐄_X);

knn_Y = tmapmap(trees,𝐄_Y) do tree,E
    map(k->knn(tree,E,k),𝐤)
end;


𝐃 = mapmap(distmat,𝐄_X)

𝐃_X= tmap(E->1 ./ (pairwise(Euclidean(),E,E) + (I*Inf)),𝐄);
𝐊_X = tmap(𝐃) do D
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

#@showprogress 𝐆 = map(𝐊) do K
#    map(𝐤) do k
#        mapslices(x->sparsevec(x[1:k],1,n),K,dims=2)
#    end
#end
#@save "knns.bson" 𝐆
#
#𝐆 = map(𝐊, 𝐃) do K,D
#    map(𝐤) do k
#        G = mapslices(x->(wak ∘ sparsevec)(x[1:k,1],x[1:k,2],n),
#                      cat(K,D,dims=3),dims=(2,3))
#        return dropdims(G,dims=3)
#    end
#end

L_ŋk = mapreduce(hcat,Θ.layers,𝐄,𝐃,𝐆) do θ,E,D,Gs
    E = E |> gpu
    D = D |> gpu
    map(Gs) do G
        G = G|>gpu
        return 𝕃(X',θ[2],E,D,G)
    end
end

Tables.table(L_ŋk) |> CSV.write("MSEagma_k.csv")
k,ŋ = argmin(L_ŋk).I 

θ = Θ.layers[ŋ]
E = 𝐄[ŋ] |> gpu
D = 𝐃[ŋ]
G = 𝐆[ŋ][k]

𝐂 = tmap(γ->Leiden.leiden(wak(D .* G),"mod++", γ = γ),𝛄)
(Tables.table ∘ hcat)(𝐂...) |> CSV.write("clusters.csv")

𝐏 = map(partitionmat,𝐂)

D = D |> gpu

Lᵧ = map(𝐏) do P
    P = P |> gpu
    return 𝕃(X',θ[2],E,D,P)
end
argmin(Lᵧ)

DataFrame(gamma=𝛄,MSE=Lᵧ) |> CSV.write("MSEgamma.csv")

G = G |> gpu
L_γd = mapreduce(hcat,𝐏) do P
    P = P |> gpu
    _,L = diffuse(X',θ[2],E,D,G,P,𝐝)
    return L
end

Tables.table(L_γd) |> CSV.write("MSEgamma_d.csv")

DataFrame(gamma=𝛄,MSE=LᵧG) |> CSV.write("clustergraphloss.csv")

LᵧG² = map(𝐏) do P
    P = P |> gpu
    M = P .* G
    return Flux.mse(X',(θ_ŋ[2] ∘ ehat)(E,D,M * M))
end

tmp = G .* P
tmp=wak(tmp .* D)
diffuse(tmp,10)

L_diff = ((x->hcat(x...)) ∘ map)(𝐏) do P
    P = P |> gpu
    M = P .* G
    M = wak(M .* D)
    _,L = diffuse(M,50)
    return L
end

(Tables.table ∘ vcat)(𝛄,L_diff) |> CSV.write("clustdiffloss.csv")
