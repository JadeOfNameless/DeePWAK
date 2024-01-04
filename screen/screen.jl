include("julia/DeePWAK.jl")
include("julia/LayerFns.jl")

using Pkg
using ThreadTools

using CSV, DataFrames, StatsPlots

Pkg.activate("julia/leiden")
using Leiden

epochs = 1000


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

M = Chain(θ,ϕ) |> gpu

L = train!(M,loader,opt,epochs*10)
