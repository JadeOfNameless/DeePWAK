using CSV, DataFrames, #Tensors, CategoricalArrays, 
    #SparseArrays, Distances,
    Flux, ProgressMeter, CUDA
include("fns.jl")

epochs=100
batchsize=1024
η = 0.01
λ = 0

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

test = mapreduce(x->x[1],zcat,tmp);
train = mapreduce(x->x[2],zcat,tmp);

X = permutedims(train,(3,1,2)) |> gpu;
Y = permutedims(test,(3,1,2)) |> gpu;

𝚯_e = map(1:b) do _
    mapreduce(ŋ->Dense(m => ŋ,relu),
              (x...)->Parallel(x...,vcat),
              bfŋ)
end |> gpu;

𝚯_d = map(1:b) do _
    mapreduce(ŋ->Dense(ŋ => m,relu),
              (x...)->Parallel(x...,vcat),
              bfŋ)
end |> gpu;
    
loader = map(eachslice(X,dims=1)) do X_b
    Flux.DataLoader((X_b,repeat(X_b,outer=(length(bfŋ),1))),
                    batchsize=batchsize, shuffle=true)
end |>gpu;

opt = Flux.Optimiser(Flux.AdamW(η), Flux.WeightDecay(λ))

@showprogress map(1:epochs) do _
    map(𝚯_e,𝚯_d,loader) do Θ_e,Θ_d,l
        function loss(X,Y)
            Ŷ = (Θ_d ∘ Θ_e)(X)
            L = Flux.mse(Ŷ,Y)
            return L
        end
        Flux.train!(loss,Flux.params(Θ_e,Θ_d),l,opt)
    end
end

        
            
