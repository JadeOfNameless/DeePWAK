include("julia/DeePWAK.jl")
include("julia/LayerFns.jl")

using MLDatasets, StatsPlots
using ImageInTerminal
using JLD2

epochs = 10000
batchsize=1024

η = 0.001
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

h = 5
l = 3
s = 4

d = 10
c = 10

dat = MNIST.traindata();
X = vcat(eachslice(dat[1],dims=1)...) |> gpu;

m,n = size(X)
d = m

loader = Flux.DataLoader((X,X),batchsize=batchsize,shuffle=true) 

block = DeePWAKBlock(h,l,s,m,d,c,σ,σ) |> gpu

L = train!(block,loader,opt,epochs)
