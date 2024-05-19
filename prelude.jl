using Pkg
Pkg.activate("envs/leiden")
using Leiden
using Flux,CUDA,CSV,DataFrames,Dates

include("julia/SparseEncoders/SparseEncoders.jl")
using .DistEncoders,.SparsityLoss

include("julia/TrainingIO/TrainingIO.jl")
using .TrainingIO

include("julia/TrainingIO/Loaders.jl")
using .Preprocessing

include("julia/TrainingIO/DictMap.jl")
using .DictMap

include("julia/TrainingPlots/SaveDate.jl")
using .SaveDate

include("models.jl")
include("clustering.jl")
#include("lossfns.jl")
#include("trainingfns.jl")

path = "data/"*date()

epochs = 10000

η = 0.0001
λ = 0.0001
α = 0.001
β = 0.00001

opt = Flux.AdamW(η)
opt_wd = Flux.Optimiser(opt,Flux.WeightDecay(λ))

𝛄 = (0.01,3)
𝐤 = (1,30)
