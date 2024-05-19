include("julia/LayerTypes.jl")
include("julia/LayerFns.jl")

using CSV, DataFrames, ProgressMeter, StatsPlots
using JLD2

epochs = 100000
batchsize=1024

l = 10

α = 0.001
β = 0.01

η = 0.001
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

dat = (DataFrame ∘ CSV.File)("data/z_dat.csv",normalizenames=true);
dat = Matrix(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);

X = scaledat(dat')
m,n = size(X)
d = 50
c = 50

frac = 10
n_test = Integer(2^round(log2(n) - log2(frac)))
n_train = n - n_Y
test,train = sampledat(X,n_test) |> gpu

Y = zfc(X) |> gpu
X = gpu(X)

loader = Flux.DataLoader((train,train),batchsize=n_test,shuffle=false) 

θ = mlp4x(m,m,l,tanh) |> gpu
ϕ = mlp4x(m,m,l,tanh) |> gpu

θ = Chain(Dense(m => 58, tanh),
          Dense(58 => 29, tanh),
          Dense(29 => 14, tanh)) |> gpu
ϕ = Chain(Dense(14 => 29, tanh),
          Dense(29 => 58, tanh),
          Dense(58 => m, tanh)) |> gpu

autoencoder = Chain(θ,ϕ)

L_train,L_test = @showprogress map(1:10000) do _
    map(loader) do (x,y)
        function loss(f)
            Flux.mse(f(x),x)
        end

        l_train = update!(autoencoder,loss,opt)
        l_test = Flux.mse(autoencoder(test),test)

        return l_train,l_test
    end
end

L = @showprogress map(1:1000) do _
    map(loader) do (x,y)
        function loss(f)
            E = f[1](x)
            D = normeucl(E)
            G = wak(D)
            Ehat = (G * E')'
            Flux.mse(f[2](Ehat),x)
        end

        l_train = update!(autoencoder,loss,opt)
        l_test = Flux.mse(autoencoder(test),test)

        return l_train,l_test
    end
end

ω = WeightNetwork(mlp4x(m,m,l)) |> gpu
π = (ClustNetwork ∘ mlp4x)(m,c,l) |> gpu
encoder = WeightedEncoder(autoencoder[1],ω)

M = Chain(encoder,autoencoder[2])

L_w = @showprogress map(1:1000) do _
    map(loader) do (x,y)
        function loss(f)
            w = f[1].weigher(x)
            entropy = (mean ∘ H)(w)
            
            E = f[1](x)
            D = normeucl(E)
            G = wak(D)
            Ehat = (G * E')'
            L = Flux.mse(f[2](Ehat),x)
            return L + α * entropy
        end

        update!(M,loss,opt)
        
        l_train = Flux.mse(M(x),x)
        H_train = (mean ∘ H ∘ M[1].weigher)(x)
        l_test = Flux.mse(M(test),test)
        H_test = (H ∘ M[1].weigher)(test)

        return (l_train,H_train),(l_test,H_test)
    end
end

Tables.table(hcat(L...)') |> CSV.write("encoderMSE")
M_s = Flux.state(M) |> cpu
jldsave("M.jld2"; M_s)

M_s = JLD2.load("M.jld2","M_s")
Flux.loadmodel!(M,M_s)
M = gpu(M)

W = DeePWAK(M[1],Chain(normeucl),π,M[2])

L = @showprogress map(1:1000000) do _
    map(loader) do (x,y)
        function loss(f)
            Flux.mse(f(x),y)
        end
        
        l = update!(W,loss,opt)
        return l
    end
end

E = W.encoder(X̃)
D = W.metric(E)
C = W.partitioner(E)
    P = C' * C
    G = wak(D .* P)
    Ehat = (G * E')'
W.decoder(Ehat)

function loss(m)
