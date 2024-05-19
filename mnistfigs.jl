include("julia/DeePWAK.jl")
include("julia/LayerFns.jl")
include("julia/SAE.jl")
include("julia/Rmacros.jl")
include("julia/auxfns.jl")
include("julia/KEheatmap.jl")

using MLDatasets, StatsPlots, OneHotArrays
using Flux: logitcrossentropy

using JLD2, Tables, CSV, DataFrames
using ImageInTerminal,Images

path = "data/MNIST/"

batchsize = 128
m = 3
d = 27
k = 12

dat = MNIST(split=:train)[:]
target = onehotbatch(dat.targets,0:9)

m_x,m_y,n = size(dat.features)
X = reshape(dat.features[:, :, :], m_x, m_y, 1, n)
colorview(Gray,X[:,:,1,1:2])

loader = Flux.DataLoader((X,target),
                         batchsize=batchsize,
                         shuffle=true)

kern = (3,3)
s = (2,2)
θ_conv = Chain(Conv(kern,1 => 3,relu,stride=s),
               Conv(kern,3 => 6,relu,stride=s),
               Conv(kern,6 => 9,relu,stride=s),
               Conv((2,2),9 => 12,relu))

θ_mlp = Chain(Dense(12 => 6,relu),
              Dense(6 => m,relu))

θ_outer = Chain(θ_conv,
                x->reshape(x,12,:),
                θ_mlp)

π_outer = Chain(Dense(m => 5,relu),
                Dense(5 => 10,relu),
                softmax)

M_outer = Chain(θ_outer,π_outer) |> gpu


θ_inner = Chain(Dense(m => 9,relu),
                Dense(9 => d,relu))
ϕ_inner = Chain(Dense(d => 9,relu),
                Dense(9 =>m,relu))
π_inner = Chain(Dense(m => 6,relu),
                Dense(6 => k,relu),
                softmax)
M_inner = DeePWAK(θ_inner,π_inner,ϕ_inner) |> gpu

sae = SAE(m,d) |> gpu
psae = PSAE(sae,π_inner) |> gpu

state_outer = JLD2.load("data/MNIST/state_outer.jld2","state_outer");
Flux.loadmodel!(M_outer,state_outer)
state_inner = JLD2.load("data/MNIST/state_inner.jld2","state_inner");
Flux.loadmodel!(M_inner,state_inner)
state_SAE = JLD2.load("data/MNIST/state_SAE.jld2","state_SAE");
Flux.loadmodel!(sae,state_SAE)
state_PSAE = JLD2.load("data/MNIST/state_PSAE.jld2","state_PSAE");
Flux.loadmodel!(psae,state_PSAE)

x,y = first(loader)
x = gpu(x)
y = gpu(y)
labels = unhot(y)'
E_outer = M_outer[1](x)

E_inner = M_inner.encoder(E_outer)
E_SAE = encode(sae,E_outer)

K_outer = M_outer(x)
K_inner = M_inner.partitioner(E_outer)

x = x[:,:,:,1:10]

E_outer = M_outer[1](x)
E = encode(psae,E_outer)
K = cluster(psae,E_outer)
KEheatmap("tmp.pdf",K,E)

P = pwak(K)

Ehat = (pwak(K)*E')'
Chat = (K * Ehat')' ./ sum(K,dims=1)'
C = K_SAE * E_PSAE' ./ batchsize

topleftfill = zeros(k,k)
bottomrightfill = zeros(d,d)

hmvals = Dict([(:topleftfill,topleftfill),
               (:K,K_SAE),
               (:C,C),
               (:KT,K_SAE'),
               (:P,P),
               (:E,E_PSAE'),
               (:Chat,Chat),
               (:Ehat,Ehat),
               (:bottomrightfill,bottomrightfill)])

colorkey = Dict([(:topleftfill,:K),
              (:K,:K),
              (:C,:E),
              (:KT,:K),
              (:P,:P),
              (:E,:E),
              (:Chat,:E),
              (:Ehat,:E),
              (:bottomrightfill,:E)])

colsp = vcat(rep("K^T",k),
             rep("PWAK(K)",batchsize),
             rep("KE^T",d))
rowsp = vcat(rep("K",k),
             rep("PWAK(K)",batchsize),
             rep("KEhat^T",d))

hmdat = Dict([(:E,E_PSAE),
            (:K,K_SAE),
            (:P,P)])
cols = Dict([(:E,"red"),
             (:K,"blue"),
             (:P,"black")])
legend = Dict([(:E,"embedding"),
               (:K,"P(cluster)"),
               (:P,"pairwise weight")])
colfns = mapkey((M,c)->circlize.colorRamp2([extrema(M)...],
                                           ["white",c]),
                hmdat,cols)

f = (key,val)->colfns[colorkey[key]](val)
hmfill = maplab(f,hmvals)

colfns = mapkey(hmdat,cols) do (M,c)
    lims = [extrema(M)...]
    return circlize.colorRamp2(lims,["white",cols[i]])
end

layout = [:topleftfill :K :C;
          :KT :P :E;
          :Chat :Ehat :bottomrightfill]

Kfill = vcat(hcat(zeros(k,k),K_SAE,zeros(k,d)),
             hcat(K_SAE',zeros(batchsize,batchsize+d)),
             zeros(d,k+batchsize+d))

Pfill = vcat(zeros(k,k+batchsize+d),
             hcat(zeros(batchsize,k),P,zeros(batchsize,d)),
             zeros(d,k+batchsize+d))

Efill = vcat(hcat(zeros(k,k+batchsize),C),
             hcat(zeros(batchsize,k+batchsize),E_PSAE'),
             hcat(Chat,Ehat,zeros(d,d)))

Kmask = vcat(hcat(zeros(k,k),ones(k,batchsize),zeros(k,d)),
             hcat(ones(batchsize,k),zeros(batchsize,batchsize+d)),
             zeros(d,k+batchsize+d))

Pmask = vcat(zeros(k,k+batchsize+d),
             hcat(zeros(batchsize,k),
                  ones(batchsize,batchsize),
                  zeros(batchsize,d)),
             zeros(d,k+batchsize+d))

Emask = vcat(hcat(zeros(k+batchsize,k+batchsize),
                  ones(k+batchsize,d)),
             hcat(ones(d,k+batchsize),zeros(d,d)))

fill = Dict([("K",Kfill),
             ("P",Pfill),
             ("E",Efill)])
mask = Dict([("K",Kmask),
             ("P",Pmask),
             ("E",Emask)])
mask = map(x->Matrix{Bool}(x),mask)
cols = Dict([("K","blue"),
             ("P","black"),
             ("E","red")])
legend = Dict([("K","P(cluster)"),
               ("P","pairwise weight"),
               ("E","embedding")])

interclust = K_SAE * K_SAE'
interclust = interclust ./ sum(interclust,dims=1)

maskedheatmap("tmp.pdf",fill,mask,cols,legend;
              split=rowsp,column_split=colsp,
              show_row_dend=false,show_column_dend=false)

clust_outer = unhot(K_outer)
clust_inner = unhot(K_inner)
clust_SAE = unhot(K_SAE)

E = Dict([("bottleneck",E_outer),
          ("DeePWAK",E_inner),
          ("SAE",E_SAE),
          ("PSAE",E_PSAE)])

K = Dict([("bottleneck",K_outer),
          ("DeePWAK",K_inner),
          ("PSAE",K_SAE)])

C = map(unhot,K)
C["labels"] = labels'

hyper = batchhyper(C)

combinedheatmap(path*"E_labels",E;split=labels,name="embedding")
combinedheatmap(path*"E_predicted",E;split=clust_outer',name="embedding")
combinedheatmap(path*"K_labels",K;split=labels,name="P(cluster)")
combinedheatmap(path*"K_predicted",K;split=clust_outer',name="P(cluster)")

combinedhyper(path*"enrichment.pdf",C)

L_outer = (DataFrame ∘ CSV.File)(path*"L_outer.csv")[:,1]
L_SAE = (DataFrame ∘ CSV.File)(path*"L_SAE.csv")[:,1]
L_PSAE = (DataFrame ∘ CSV.File)(path*"L_PSAE.csv")[:,1]
L_DeePWAK = (DataFrame ∘ CSV.File)(path*"L_inner.csv")[:,1]

p = scatter(1:length(L_outer), L_outer,
            xlabel="batch",ylabel="loss",
            label="outer");
savefig(p,path*"loss_outer.svg")
savefig(p,path*"loss_outer.pdf")

p = scatter(1:length(L_SAE), L_SAE,
            xlabel="batch",ylabel="loss",
            label="SAE");
scatter!(1:length(L_PSAE), L_PSAE,label="PSAE");
scatter!(1:length(L_DeePWAK), L_DeePWAK,label="DeePWAK");
savefig(p,path*"loss_inner.svg")
savefig(p,path*"loss_inner.pdf")


heatmap(path*"K_SAEmult.pdf",Kmult';
        split=rowsp, column_split=colsp,
        name="P(cluster)",
        show_row_dend=false,show_column_dend=false,
        #row_title_rot=1,
        border=true)

@rput CEhat
@rput CE
R"""
hm <- do.call(Heatmap,args)
colE <- col.abs(CE,cols=c("white","red"))
hmCEhat <- Heatmap(CEhat,colE,"embedding",show_row_dend=F)
hmCE <- Heatmap(CE,colE,show_column_dend=F)
pdf("tmp.pdf")
draw(hm %v% hmCEhat + hmCE)
dev.off()
"""

