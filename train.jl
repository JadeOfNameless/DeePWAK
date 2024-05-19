include("prelude.jl")

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

layers = accumulate(÷,rep(2,l),init=2*m)
d = last(layers)

# M_dense learns a reduced dimension embedding
θ_dense = mlp(layers,tanh) |> gpu
ϕ_dense = mlp(reverse(layers),tanh) |> gpu

M_dense = Autoencoder(θ_dense,ϕ_dense)

train!(M_dense,path*"/dense/L2",
       (M,x,_)->l2_dense(M,test,x),loader,opt,epochs)

E = encode(M_dense,X)

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

k_min = 1
k_max = 200

γ_min = 0
γ_max = 5

s = 25

using Distributions
using ThreadTools

function loss_k(k)
    K = knn(N,k) |> CuArray{Float32}
    G = wak(D .* K)
    F̂ = ((G^s) * F')'
    Ê = distenc.decoder(F̂)
    X̂ = decode(M_dense,Ê)
    return Flux.mse(X̂,X)
end

L_k = @showprogress map(loss_k,k_min:k_max)
k = argmin(L_k)
K = knn(N,k)
G = cpu(D) .* K

function loss_clust(C)
    P = partitionmat(C) |> gpu
    G = wak(D .* P)
    F̂ = ((G ^ s) * F')'
    Ê = distenc.decoder(F̂)
    X̂ = decode(M_dense,Ê)
    return Flux.mse(X̂,X)
end

γs = rand(Uniform(γ_min,γ_max),200)
C = map(γ->Leiden.leiden(G,"mod++",γ=γ),γs)
s = 5
L_clust = map(loss_clust,C)
