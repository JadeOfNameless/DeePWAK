include("julia/knn.jl")
include("julia/Rmacros.jl")
include("julia/fns.jl")

using SparseArrays

dat = (DataFrame ∘ CSV.File)("data/z_dat.csv",normalizenames=true);
dat = Matrix(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);
X = scaledat(dat')

m,n = size(X)

X = X[:,sample(1:n,100)]

n = 100

@rput X
pcs = R"prcomp(t(X))" |> rcopy
pcs10 = pcs[:x][:,1:10]

G_10 = knn(pcs10',10);
pred = (wak(G_10)*X')';

E_c = (Matrix ∘ DataFrame ∘ CSV.File)("data/E_consensus.csv")';
Ehat_c = (wak(G_10)*E_c')';

colsp = vcat(rep("adjacency matrix",n),
             rep("params",m),)
rowsp = vcat(rep("matmul",n),
             rep("predicted",m))

M = vcat(hcat(G_10,X'),
         hcat(pred,zeros(m,m)))

hmargs = Dict([("matrix",M),
               ("name"," "),
               ("show_column_names",false),
               ("show_row_names",false),
               ("show_row_dend",false),
               ("show_column_dend",false),
               ("row_split",rowsp),
               ("column_split",colsp),
               ("row_title_rot",0)])

@rput hmargs
@Heatmap hmargs "10NN100.pdf"

@rput G_10
clust = @leiden G_10;
c = maximum(clust)
C = sparse(1:n,clust,1,n,c)
P = C*C'

M = vcat(hcat(zeros(c,c),C'),
         hcat(C,P))

colsp = vcat(rep("clusts",c),
             rep("partition matrix",n),)
rowsp = vcat(rep("transpose",c),
             rep("matmul",n))

hmargs = Dict([("matrix",M),
               ("name"," "),
               ("show_column_names",false),
               ("show_row_names",false),
               ("show_row_dend",false),
               ("show_column_dend",false),
               ("row_split",rowsp),
               ("column_split",colsp),
               ("row_title_rot",0)])
@rput hmargs
@HeatmapScale [0,1] ["white","black"] hmargs "leiden.pdf"

@rput E_c
@Heatmap E_c
@Heatmap G_10

@Rfn :Heatmap ["matrix","name"] [E_c," "]
