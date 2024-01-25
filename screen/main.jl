include("julia/DeePWAK.jl")
include("julia/LayerFns.jl")

using CSV, DataFrames, StatsPlots
using JLD2

epochs = 10000


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

θ = mlp(l_e,tanh)
ϕ = mlp(reverse(l_e),tanh)
π = Chain(mlp4x(m,c,l_p),softmax)

M = DeePWAK(θ,π,ϕ) |> gpu

L_train,L_test = train!(M,loader,opt,test,epochs*10)

O = Parallel(+,map(_->DeePWAK(mlp(l_e,tanh),
                                  Chain(mlp4x(m,c,l_p),softmax),
                                  mlp(reverse(l_e),tanh)),
                       1:h)...) |> gpu

L_Otrain,L_Otest = train!(O,loader,opt,test,epochs)

O_s = Flux.state(O) |> cpu
jldsave("DeePWAKBlock.jld2"; O_s)

O_s = JLD2.load("DeePWAKBlock.jld2","O_s");
Flux.loadmodel!(O,O_s)
O = gpu(O);

C = mapreduce(vcat,O.layers) do M
    return (clusts ∘ M.partitioner)(X)
end

Tables.table(C') |> CSV.write("data/clusts.csv")

map(O.layers,1:h) do M,i
    E = M.encoder(X)
    Tables.table(E') |> CSV.write("data/embedding/$i.csv")
end

map(O.layers,1:h) do M,i
    C = M.partitioner(X)
    Tables.table(C') |> CSV.write("data/clust/$i.csv")
end

p_cat = Parallel(vcat,map(M->M.partitioner,O.layers)) |> gpu
p_consensus = Chain(mlp4x(h*c,c,l_p),softmax) |> gpu

function loss_consensus(O,p_cat,p_consensus,X)
    E = p_cat(X)
    C = p_consensus(E)
    P = C' * C
    G = wak(P)
    M = Parallel(+,map(m->Chain(m.encoder,E->(G * E')',
                                m.decoder),
                       O.layers)...)
    return Flux.mse(M(X),X)
end

L_consensus = @showprogress map(1:100) do _
    map(loader) do (x,y)
        loss = m->loss_consensus(O,p_cat,m,x)
        l_test = loss_consensus(O,p_cat,p_consensus,test)
        l_train = update!(p_consensus,loss,opt)
        return l_train,l_test
    end
end

p_s = Flux.state(p_consensus) |> cpu;
jldsave("p_consensus.jld2"; p_s)

C_consensus = (p_consensus ∘ p_cat)(X)
Tables.table(C_consensus') |> CSV.write("data/C_consensus.csv")
clusts(C_consensus)' |> Tables.table |> CSV.write("data/clusts_consensus.csv")

e_cat = Parallel(vcat,map(M->M.encoder,O.layers)) |> gpu
e_consensus = mlp([h*d,35,d],tanh) |> gpu

function loss_Ec(O,e_cat,e_consensus,X)
    M = Parallel(+,map(m->DeePWAK(Chain(e_cat,e_consensus),
                                  m.partitioner,
                                  m.decoder),
                       O.layers)...) |> gpu
    return Flux.mse(M(X),X)
end

L_Ec = @showprogress map(1:100) do _
    map(loader) do (x,y)
        loss = m->loss_Ec(O,e_cat,m,x)
        l_test = loss_Ec(O,e_cat,e_consensus,test)
        l_train = update!(e_consensus,loss,opt)
        return l_train,l_test
    end
end

e_s = Flux.state(e_consensus) |> cpu;
jldsave("e_consensus.jld2"; e_s)

E_consensus = (e_consensus ∘ e_cat)(X)
Tables.table(E_consensus') |> CSV.write("data/E_consensus.csv")

include("julia/knn.jl")

G_10 = knn(E_consensus,10)

Tables.table(G_10) |> CSV.write("data/10NN.csv")

Ehat_10 = wak(G_10) * E_consensus
Tables.table(Ehat_10) |> CSV.write("data/Ehat_10.csv")
