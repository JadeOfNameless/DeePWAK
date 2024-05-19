include("julia/DeePWAK.jl")
include("julia/LayerFns.jl")

using MLDatasets, StatsPlots
using ImageInTerminal
using JLD2

epochs = 25
batchsize=512

η = 0.001
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

h = 5
l = 3
s = 4

d = 64
c = 10

dat = MNIST.traindata();
X = vcat(eachslice(dat[1],dims=1)...) |> gpu;

m,n = size(X)

loader = Flux.DataLoader((X,dat[2]),batchsize=batchsize,shuffle=true) 

θ = Dense(m => d,relu)
ϕ = Dense(d => m,σ)

M = Chain(θ,ϕ) |> gpu

L = train!(M,loader,opt,epochs*10)

π = Chain(Dense(m => 4*m),Dense(4*m => 4*m,σ),Dense(4*m => m),softmax)

M = DeePWAK(Chain(θ),π,Chain(ϕ)) |> gpu
L_M = train!(M,loader,opt,epochs)

x,y = first(loader)
reshape(x[:,2],(28,28)) |> x->map(Gray,x)
reshape(M(x)[:,2],(28,28)) |> x->map(Gray,x)

clusts = map(x->x[1],argmax(M.partitioner(x),dims=2))

π = Chain(Dense(m => 4*m,relu),Dense(4*m => 4*m,σ),Dense(4*m => 2*c,relu),softmax) |> gpu

L_P = map(1:epochs) do _
    map(loader) do (x,_)
        function loss(π)
            E = M[1](x)
            C = π(E)
            P = C' * C
            G = wak(P)
            Ehat = (G * E')'
            return Flux.mse(M[2](Ehat),x)
        end
        return update!(π,loss,opt)
    end
end

deepwak = DeePWAK(Chain(θ),π,Chain(ϕ)) |> gpu
L_deepwak = train!(deepwak,loader,opt,epochs)

block = Parallel(vcat,map(_->DeePWAK([m,d],[m,4*m,c],[d,m],relu,relu,σ),1:h)...)
κ = Dense(h * m => m,relu)
dblock = Chain(block,κ) |> gpu

L_block = train!(dblock,loader,opt,epochs)
             
block = DeePWAKBlock(h,l,s,m,d,c,relu,σ,σ) |> gpu

c = 2

f_e = ()->Chain(Dense(m => d,relu))

f_p = ()->Chain(Dense(m => 4*m),
                Dense(4*m => 4*m,σ),
                Dense(4*m => 4*m,σ),
                Dense(4*m => c),
                softmax)

f_d = ()->Chain(Dense(d => m,σ))

block = Parallel(vcat,map(_->DeePWAK(f_e(),f_p(),f_d()),1:h)...)
κ = Dense(h*m => m,σ)
dwakblock = Chain(block,κ) |> gpu

L_dwakblock = train!(dwakblock,loader,opt,epochs)

C = mapreduce(vcat,dwakblock[1].layers) do M
    clusts(M.partitioner(x))
end
        
