include("prelude.jl")
using ProgressMeter
path ="data/2024-05-03"

groups = (DataFrame ∘ CSV.File)("data/groups.csv",normalizenames=true)

dat = (DataFrame ∘ CSV.File)("data/z_dat.csv",normalizenames=true);
dat = Matrix(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);

X = scaledat(dat')
m,n = size(X)
l = 4

frac = 10
n_test = Integer(2^round(log2(n) - log2(frac)))
n_train = n - n_test
test,train = sampledat(X,n_test) |> gpu
X = gpu(X)

loader = Flux.DataLoader((train,train),batchsize=n_test,shuffle=true) 

function traindense(layers,path)
    d = last(layers)

    # M_dense learns a reduced dimension embedding
    θ_dense = mlp(layers,tanh) |> gpu
    ϕ_dense = mlp(reverse(layers),tanh) |> gpu

    M_dense = Autoencoder(θ_dense,ϕ_dense)
    train!(M_dense,path*"/dense/L2",
        (M,x,_)->l2_dense(M,test,x),loader,opt,epochs)
    
    E = encode(M_dense,X)
    TrainingIO.SaveTraining.writecsv(E',path*"/dense/L1_L2","E.csv")
    distenc = DistEnc(Chain(Dense(d => m)),
                    Chain(Dense(m => d)),inveucl) |> gpu
    dir = path*"/sparse/eucl/linear/L2/"
    mkpath(dir)
    L2_eucl = @showprogress map(_->TrainingIO.update!(distenc,
                            M->Flux.mse(decode(M_dense,(M ∘ encode)(M_dense,X)),X),
                            opt), 1:epochs)
    TrainingIO.SaveTraining.savemodel(distenc,dir*"/final")
    TrainingIO.SaveTraining.writecsv(L2_eucl,dir*"/loss.csv")
    F = encode(distenc,E)
    D = kern(distenc,F)
    N = perm(D |> cpu)

    TrainingIO.SaveTraining.writecsv(F,dir,"F.csv")
    TrainingIO.SaveTraining.writecsv(D,dir,"D.csv")
    TrainingIO.SaveTraining.writecsv(N,dir,"neighbors.csv")
end

traindense([m,58,29,14,8,4,2],path*"2")

layers = accumulate(÷,rep(2,l),init=2*m)
d = last(layers)

# M_dense learns a reduced dimension embedding
θ_dense = mlp(layers,tanh) |> gpu
ϕ_dense = mlp(reverse(layers),tanh) |> gpu

M_dense = Autoencoder(θ_dense,ϕ_dense)

Θ_dense = DistEnc(θ_dense,ϕ_dense,inveucl)
train!(Θ_dense,path*"/dense/eucl/L2",
       (M,x,_)->l2_dense(M,test,x),loader,opt,epochs)

train!(M_dense,path*"/dense/L2",
       (M,x,_)->l2_dense(M,test,x),loader,opt,epochs)
#L1 uses loss_cossim
#minimized when all embeddings are orthogonal
train!(M_dense,path*"/dense/L1_L2",
                 (M,x,_)->loss_dense(α,M,test,x),loader,opt,epochs)

train!(M_dense,path*"/dense/sparsecov",
       (M,x,_)->loss_sparsecov(α,M,x,x),loader,opt,epochs*2)

load!(M_dense,path*"/dense/L1_L2")

E = encode(M_dense,X)
TrainingIO.SaveTraining.writecsv(E',path*"/dense/L1_L2","E.csv")

θ_sparse = mlp(reverse(layers),relu) |> gpu
ϕ_sparse = mlp(layers,relu) |> gpu

θ_sparse = mlp(reverse(layers),tanh) |> gpu
ϕ_sparse = mlp(layers,tanh) |> gpu

distenc = DistEnc(Chain(Dense(d => m,relu)),
                  Chain(Dense(m => d)),inveucl) |> gpu
sae = SAE(d,m)|>gpu

train!(sae,path*"/sparse/sae/L2",
       (M,x,_)->l2_sparse(M,M_dense,test,x),
       loader,opt_wd,epochs)
train!(sae,path*"/sparse/sae/L1_L2",
       (M,x,_)->loss_sparse(β,M,M_dense,test,x),
       loader,opt_wd,epochs)

# distenc attempts to split the embedding space into a space where euclidean distance predicts similarity
train!(distenc,path*"/sparse/eucl/L2",
       (M,x,_)->l2_sparse(M,M_dense,test,x),loader,opt_wd,epochs)
train!(distenc,path*"/sparse/eucl/L1_L2",
       (M,x,_)->loss_sparse(β,M,M_dense,test,x),loader,opt_wd,epochs)

train!(M_sparse,loader,opt,epochs,loss_sparse,
       prefn=x->encode(M_dense,x),postfn=E->decode(M_dense,E),
       path=path*"/sparse/L2")

L2_eucl = @showprogress map(_->TrainingIO.update!(distenc,
                        M->Flux.mse(decode(M_dense,(M ∘ encode)(M_dense,X)),X),
                        opt), 1:epochs)
TrainingIO.SaveTraining.savemodel(distenc,path*"/sparse/eucl/L2/final")
TrainingIO.SaveTraining.writecsv(L2_eucl,path*"/sparse/eucl/L2/loss.csv")

distenc = DistEnc(Chain(Dense(d => m)),
                  Chain(Dense(m => d)),inveucl) |> gpu
dir = path*"/sparse/eucl/linear/L2/"
mkpath(dir)
L2_eucl = @showprogress map(_->TrainingIO.update!(distenc,
                        M->Flux.mse(decode(M_dense,(M ∘ encode)(M_dense,X)),X),
                        opt), 1:epochs)
TrainingIO.SaveTraining.savemodel(distenc,dir*"/final")
TrainingIO.SaveTraining.writecsv(L2_eucl,dir*"/loss.csv")
    
L_eucl = @showprogress map(_->update!(distenc,
                                      M->loss_cossim(β,lossfn_sparse(Flux.mse,M_dense),M,E,X),
                        opt_wd), 1:epochs)
TrainingIO.SaveTraining.savemodel(distenc,path*"/sparse/eucl/L1_L2/","final")
TrainingIO.SaveTraining.writecsv(mapreduce(x->hcat(x...),vcat,L_eucl),
                                 path*"/sparse/eucl/L1_L2/","loss")

load!(distenc,dir)
F = encode(distenc,E)
D = kern(distenc,F)
N = perm(D |> cpu)

TrainingIO.SaveTraining.writecsv(F,dir,"F.csv")
TrainingIO.SaveTraining.writecsv(D,dir,"D.csv")
TrainingIO.SaveTraining.writecsv(N,dir,"neighbors.csv")


k_min = 1
k_max = 200

γ_min = 0
γ_max = 5

s = 25

using Distributions
using ThreadTools

lims = Dict([(:k,(1,200)),
            (:γ,(0.0,10.0))])

L_clust = DataFrame(k=Vector{Int32}(),γ=Vector{Float32}(),L=Vector{Float32}())
L_clust = (DataFrame ∘ CSV.File)(dir*"loss_hyper.csv")

η_k = 10
η_γ = 0.1

function loss_kern(G,F,s)
    G = G |> CuArray{Float32}
    G = wak(G)
    F̂ = ((G^s) * F')'
    L_F = Flux.mse(F̂,F)
    Ê = distenc.decoder(F̂)
    L_E = Flux.mse(Ê,E)
    X̂ = decode(M_dense,Ê)
    L_X = Flux.mse(X̂,X)
    return [L_F,L_E,L_X]
end

function loss_k(k,N,F,s)
    K = knn(N,k)
    return loss_kern(K,F,s)
end

function loss_kdist(k,N,D,F,s)
    K = knn(N,k) |> CuArray{Float32}
    G = D .* K
    return loss_kern(G,F,s)
end

function loss_k(k)
    K = knn(N,k) |> CuArray{Float32}
    G = wak(D .* K)
    F̂ = ((G^s) * F')'
    L_F = Flux.mse(F̂,F)
    Ê = distenc.decoder(F̂)
    L_E = Flux.mse(Ê,E)
    X̂ = decode(M_dense,Ê)
    L_X = Flux.mse(X̂,X)
    return [L_F,L_E,L_X]
end

L_k = @showprogress mapreduce(k->loss_k(k,N,F,1),hcat,k_min:k_max)
TrainingIO.SaveTraining.writecsv(L_k,dir,"L_k.csv")

L_kdist = @showprogress mapreduce(k->loss_kdist(k,N,D,F,1),hcat,k_min:k_max)
TrainingIO.SaveTraining.writecsv(L_k,dir,"L_kdist.csv")
k = argmin(L_kdist[1,:])
K = knn(N,k)
G = cpu(D) .* K
D_E = inveucl(E)
G_E = cpu(D_E) .* K

function loss_clust(C,F,s)
    P = partitionmat(C) |> CuArray{Float32}
    return loss_kern(P,F,s)
end

function loss_clust(C,G,F,s)
    P = partitionmat(C) |> CuArray{Float32}
    G = G |> CuArray{Float32}
    G = G .* P
    return loss_kern(G,F,s)
end

function loss_clust(C,G,E)
    P = partitionmat(C) |> CuArray{Float32}
    G = G |> CuArray{Float32}
    G = G .* P
    Ê = (G * E')'
    return Flux.mse(Ê,E)
end
γs = rand(Uniform(γ_min,γ_max),200)
C = map(γ->Leiden.leiden(G,"mod++",γ=γ),γs)
C_K = map(γ->Leiden.leiden(K,"mod++",γ=γ),γs)
clusts_K = vcat(γs',hcat(C_K...))
TrainingIO.SaveTraining.writecsv(clusts_K,dir,"clusts_K.csv")

C_D = map(γ->Leiden.leiden(D,"mod++",γ=γ),γs)
s = 5

L_clust = @showprogress mapreduce(c->loss_clust(c,F,1),hcat,C)
L_clustk = @showprogress mapreduce(c->loss_clust(c,K,F,1),hcat,C)
L_clustkdist = @showprogress mapreduce(c->loss_clust(c,G,F,1),hcat,C)
L_clustdist = @showprogress mapreduce(c->loss_clust(c,D,F,1),hcat,C)

L_clust = @showprogress mapreduce(c->loss_clust(c,F,1),hcat,C_K)
L_clust = @showprogress mapreduce(c->loss_clust(c,K,F,1),hcat,C_K)
TrainingIO.SaveTraining.writecsv(L_clust,dir,"L_clusts.csv")
sel = argmin(L_clust[1,:])
cl = C_K[sel]

L_subcl = @showprogress map(1:maximum(cl)) do i
    sel = cl .== i
    X = G[sel,sel]
    F = F[sel,:]
    C = map(γ->Leiden.leiden(X,"mod++",γ=γ),γs)
    L_clust = @showprogress mapreduce(c->loss_clust(c,X,F,1),hcat,C)
    return L_clust
end


function sampleclust(lims,n)
    k_min,k_max = lims[:k]
    γ_min,γ_max = lims[:γ]
    ks = rand(k_min:k_max,n)
    γs = rand(Uniform(γ_min,γ_max),n)
    C = tmap((k,γ)->clusts(N,k,γ),ks,γs)
    L = map(loss_clust,C)
    return DataFrame(hcat(ks,γs,L),names(L_clust))
end

function trainclust!(lims)
    L = sampleclust(lims,8)
    append!(L_clust,L)
    i = argmin(L_clust[:,:L])
    Δk = (lims[:k] .- L_clust[i,:k]) .÷ η_k
    Δγ = (lims[:γ] .- L_clust[i,:γ]) .* η_γ
    lims[:k] = lims[:k] .- Δk
    lims[:γ] = lims[:γ] .- Δγ
end

@showprogress map(1:10) do i
    dir = dir*string(i)
    mkdir(dir)
    map(_->trainclust!(lims),1:10)
end

CSV.write(dir*"loss_hyper.csv",L_clust)
    
i = argmin(L_clust[:,:L])
k = L_clust[i,:k]
γ = L_clust[i,:γ]
C = clusts(N,k,γ)
P = partitionmat(C) |> gpu

f = M->begin
    F = encode(M,E)
    D = dist(M,F)
    G = wak(D .* P)
    F̂ = ((G^s) * F')'
    Ê = distenc.decoder(F̂)
    X̂ = decode(M_dense,Ê)
    return Flux.mse(X̂,X)
end
    
L2_eucl = @showprogress map(_->TrainingIO.update!(distenc, f, opt), 1:1000)
L_k = map(1:k_max) do k
    K = knn(N,k) |> gpu
    G = wak(D .* K)
    F̂ = (G * F')'
    Ê = distenc.decoder(F̂)
    X̂ = decode(M_dense,Ê)
    return Flux.mse(X̂,X)
end

clusts = Leiden.leiden(D,"mod++",γ = 1)
    

M = Chain(θ_dense,distenc,ϕ_dense)

f = (M,x,_)->begin
    E = M[1](x)
    F = encode(M[2],E)
    D = kern(M[2],F)
    Ê = decode(M[2],(D * F')')
    L1 = α * (sum ∘ sparsecov)(F)
    L2 = Flux.mse(M[3](Ê),x)
    return L1 + L2, L1,L2
end

train!(M,path*"/endtoend/sparsecov",(M,x,_)->loss_test(f,M,test,x),loader,opt_wd,epochs)
