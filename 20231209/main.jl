include("julia/DeePWAK.jl")
include("julia/LayerFns.jl")

using CSV, DataFrames, StatsPlots
using JLD2

epochs = 10000


γ = 1.0

η = 0.001
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

groups = (DataFrame ∘ CSV.File)("data/groups.csv",normalizenames=true)

dat = (DataFrame ∘ CSV.File)("data/z_dat.csv",normalizenames=true);
dat = Matrix(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);

X = scaledat(dat')
m,n = size(X)
d = 14
c = 14
l_e = [m,58,29,d]
l_p = 5
h = 5

frac = 10
n_test = Integer(2^round(log2(n) - log2(frac)))
n_train = n - n_test
test,train = sampledat(X,n_test) |> gpu
X = gpu(X)

loader = Flux.DataLoader((train,train),batchsize=n_test,shuffle=true) 

θ = mlp(l_e,tanh)
ϕ = mlp(reverse(l_e),tanh)
π = Chain(mlp4x(m,c,l_p),softmax)

M = DeePWAK(θ,π,ϕ) |> gpu

L_train,L_test = train!(M,loader,opt,test,epochs*10)

O = Parallel(+,map(_->DeePWAK(mlp(l_e,tanh),
                                  Chain(mlp4x(m,c,l_p),softmax),
                                  mlp(reverse(l_e),tanh)),
                       1:h)...) |> gpu

L_Otrain,L_Otest = train!(O,loader,opt,test,epochs)

O_s = Flux.state(O) |> cpu
jldsave("DeePWAKBlock.jld2"; O_s)

C = mapreduce(vcat,O.layers) do M
    return (clusts ∘ M.partitioner)(X)
end

Tables.table(C') |> CSV.write("data/clusts.csv")

map(O.layers,1:h) do M,i
    E = M.encoder(X)
    Tables.table(E') |> CSV.write("data/embedding/$i.csv")
end

map(O.layers,1:h) do M,i
    C = M.partitioner(X)
    Tables.table(C') |> CSV.write("data/clust/$i.csv")
end

