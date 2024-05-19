include("julia/LayerTypes.jl")
include("julia/LayerFns.jl")

epochs = 1000
d = 50
η = 0.01
λ = 0.01
batchsize=1024

α = 1
β = 1
γ = 1

using MLDatasets, ProgressMeter

dat = MNIST.traindata();
X = vcat(eachslice(dat[1],dims=1)...);

m,n = size(X)

opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

X̃ = zfc(X)|>gpu

loader = Flux.DataLoader((X̃,X̃),batchsize=batchsize,shuffle=true) |> gpu

md = Dense(m => m)
dd = OneToOne(m,m,σ)#Softmax(m)
dm = Dense(m => m)

θ = Chain(md,dd,dm) |> gpu


function loss(X,θ,ω_E=[0.1,0.1,0.1],ω_D=0)
    H_θ = sum(map(l->(mean∘H)(l.weight),θ.layers) .* ω_E)

    E = θ[1:2](X)
    D = 1 ./ (euclidean(E) .+ eps(Float32))
    D = wak(D)

    H_D = (mean ∘ H)(D)
    #ℍ = softmod(D,P,γ) 
    L = Flux.mse(X,θ[3]((D * E')'))
    return H_θ + ω_D * H_D + log(L) #- δ * ℍ
    #L = Flux.mse(X,θ(X))
    #return H_E * L
end

H_θ = []
L = @showprogress map(1:epochs) do _
    L = map(loader) do (x,y)
        state = Flux.setup(opt,θ)
        l,∇ = Flux.withgradient(θ->loss(x,θ),θ)
        Flux.update!(state,θ,∇[1])
        H_θ = map(l->(mean∘H)(l.weight),θ.layers)
        push!(H_θ,H_θ)
        return l,H_θ
    end
    return L
end
