using CSV, DataFrames, ProgressMeter, StatsPlots
using JLD2

epochs = 10000

l = 3

γ = 1.0

η = 0.001
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

dat = (DataFrame ∘ CSV.File)("data/z_dat.csv",normalizenames=true);
dat = Matrix(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);

X = scaledat(dat')
m,n = size(X)
d = 14
c = 14

frac = 10
n_test = Integer(2^round(log2(n) - log2(frac)))
n_train = n - n_test
test,train = sampledat(X,n_test) |> gpu

loader = Flux.DataLoader((train,train),batchsize=n_test,shuffle=true) 

θ = Chain(Dense(m => 58, tanh),
          Dense(58 => 29, tanh),
          Dense(29 => 14, tanh)) |> gpu
ϕ = Chain(Dense(14 => 29, tanh),
          Dense(29 => 58, tanh),
          Dense(58 => m, tanh),
          Dense(m => m, tanh) |> gpu

θ = mlp4x(m,d,l,tanh) |> gpu
ϕ = mlp4x(d,m,l,tanh) |> gpu
autoencoder = DAEWAK(θ,ϕ)

L_train,L_test = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        l_train = update!(autoencoder,f->Flux.mse(f(x),x),opt)
        l_test = Flux.mse(autoencoder(test),test)
        return l_train,l_test
    end
end

π = (ClustNetwork ∘ mlp4x)(m,c,l) |> gpu
X = gpu(X)

H_train,H_test = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        l_train = update!(autoencoder,f->loss(f,π,x),opt)
        l_test = loss(autoencoder,π,test)

        mod = update!(π,f->modularity(f,autoencoder,X),opt)
        #H_test = modularity(π,test)

        return l_train,l_test,mod
    end
end
