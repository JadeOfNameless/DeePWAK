include("julia/LayerTypes.jl")
include("julia/LayerFns.jl")

epochs = 1000
η = 0.01
λ = 0.0

α = 0.1
β = 0.1
γ = 1

using ProgressMeter, StatsPlots, DataFrames, CSV, DelimitedFiles

opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

dat = (DataFrame ∘ CSV.File)("data/z_dat.csv",normalizenames=true);
#dat = (scaledat ∘ Matrix)(dat[:,2:end]);
dat = Matrix(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);
X = dat'

X̃ = zfc(X)

X = X |> gpu
X̃ = X̃ |> gpu
m,n = size(X)
d = m
c = m

batchsize = n
l = 10

loader = Flux.DataLoader((X̃,X),batchsize=batchsize,shuffle=true) |> gpu

md = identity#Dense(m => d, relu)
dd = Dense(d => d)
#dc = Dense(d => c, relu)
#dm = Dense(d => m, relu)
dc = Chain(map(_->Dense(m => m,σ),1:l)...)
dm = Chain(map(_->Dense(m => m,σ),1:l)...)

M = DeePWAK(md,dd,dc,dm) |> gpu

state = Flux.setup(opt,M)

L = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        l,∇ = Flux.withgradient(m->loss(m,x,y,γ),M)
        Flux.update!(state,M,∇[1])
        return stats(M,x,y,γ)
    end
end

L = vcat(map(x->hcat(x[1]...),L)...)
p = scatter(1:epochs, (log ∘ mean).(L[:,1]),
            xlabel="epoch",ylabel="logMSE",
            legend=:none)
savefig(p,"plots/loss.pdf")

p = scatter(1:epochs, mean.(L[:,2]),
            xlabel="epoch",ylabel="entropy",
            legend=:none)
savefig(p,"plots/entropy.pdf")

p = scatter(1:epochs, mean.(L[:,3]),
            xlabel="epoch",ylabel="modularity",
            legend=:none)
savefig(p,"plots/modularity.pdf")

cl = getclusts(M,X̃)
writedlm("plots/clusters.txt",cl)
groups = readdlm("data/groups.csv",',')

D = dist(M,X̃)
C = clust(M,X̃)
P = C' * C
