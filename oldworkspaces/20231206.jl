include("julia/LayerTypes.jl")
include("julia/LayerFns.jl")

using CSV, DataFrames, ProgressMeter, StatsPlots
using JLD2

epochs = 10000

l = 3

γ = 1.0

η = 0.001
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

function trainencoder!(M,loader,opt,test,epochs)
    @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            l_test = Flux.mse(M(test),test)
            l_train = update!(M,f->Flux.mse(f(x),x),opt)
            return l_train,l_test
        end
    end
end

function trainpartitioner!(π,autoencoder,loader,opt,test,epochs)
    @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            H_test = modularity(π,autoencoder,test)
            H_train = update!(π,f->modularity(f,autoencoder,x),opt)
            return H_train,H_test
        end
    end
end

dat = (DataFrame ∘ CSV.File)("data/z_dat.csv",normalizenames=true);
dat = Matrix(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);

X = scaledat(dat')
m,n = size(X)
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
          Dense(58 => m, tanh)) |> gpu

autoencoder = Chain(θ,ϕ)

L_train,L_test = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        l_train = update!(autoencoder,f->Flux.mse(f(x),x),opt)
        l_test = Flux.mse(autoencoder(test),test)
        return l_train,l_test
    end
end

daewak = DAEWAK(autoencoder[1],autoencoder[2])

L_Dtrain,L_Dtest = trainencoder!(daewak,loader,opt,test,epochs)

π = (ClustNetwork ∘ mlp4x)(m,c,l) |> gpu
X = gpu(X)

H_train,H_test = trainpartitioner!(π,daewak,loader,opt,test,epochs)

L_train,L_test,H_train,H_test = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        l_train = update!(autoencoder,f->loss(f,π,x),opt)
        l_test = loss(autoencoder,π,test)

        mod = update!(π,f->modularity(f,autoencoder,X),opt)
        #H_test = modularity(π,test)

        return l_train,l_test,mod
    end
end

autoencoder_s = Flux.state(autoencoder) |> cpu
jldsave("autoencoder.jld2"; autoencoder_s)

pi_s = Flux.state(π) |> cpu
jldsave("pi.jld2"; pi_s)

ω = mlp4x(d,d,l) |> gpu
L_wtran,L_wtest = @showprogress map(1:epochs) do _
    map(loader) do (x,y)
        function loss(M,X)
            E = daewak.encoder(X)
            D = (normeucl ∘ M)(E)
            G = (softmax ∘ zerodiag)(D)
            Ehat = (G * E')'
            return Flux.mse(daewak.decoder(Ehat),X)
        end
        l_test = loss(ω,x)
        l_train = update!(ω,f->loss(f,x),opt)
    end
end

            

