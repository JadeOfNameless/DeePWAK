using RCall

@rimport stats
@rimport ComplexHeatmap as CH
@rimport circlize
@rimport grDevices as grD
@rimport grid as rgrid

function KEheatmap(out,K,E)
    k,n = size(K)
    d,_ = size(E)
    
    P = pwak(K)
    Ksum = sum(K,dims=2)

    C = K * E' ./ Ksum

    Ehat = (P * E')'
    Chat = K * Ehat' ./ Ksum

    topleftfill = zeros(k,k)
    bottomrightfill = zeros(d,d)
    
    hmdat = Dict([(:E,E),
                (:K,K),
                (:P,P)]),

    cols = Dict([(:E,"red"),
                (:K,"blue"),
                (:P,"black")])
    legend = Dict([(:E,"embedding"),
                (:K,"P(cluster)"),
                (:P,"pairwise weight")])

    colsp = vcat(rep("(1) K^T",k),
                rep("(2) PWAK(K)",n),
                rep("(3) KE^T",d))
    rowsp = vcat(rep("(1) K",k),
                rep("(2) PWAK(K)",n),
                rep("(3) KEhat^T",d))

    hmvals = Dict([(:topleftfill,topleftfill),
                (:K,K),
                (:C,C),
                (:KT,K'),
                (:P,P),
                (:E,E'),
                (:Chat,Chat'),
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

    layout = [:topleftfill :K :C;
            :KT :P :E;
            :Chat :Ehat :bottomrightfill]

    colfns = mapkey((M,c)->circlize.colorRamp2([extrema(M)...],
                                            ["white",c]),
                    hmvals,cols)

    f = (key,val)->colfns[colorkey[key]](val) |> rcopy
    hmfill = maplab(f,hmvals)

    hclust = map(stats.hclust ∘ stats.dist,hmdat)
                 
    ord = map(x->rcopy(x[:order]),hclust)

    sel = vcat(ord[:K],
               ord[:P] .+ k,
               ord[:E] .+ (k + n))

    mat = mapreduce(vcat,eachrow(layout)) do row
        hcat(select(hmvals,row)...)
    end
    colmat = mapreduce(vcat,eachrow(layout)) do row
        hcat(select(hmfill,row)...)
    end
    
    @rput mat
    @rput colmat
    @rput sel
    @rput rowsp
    @rput colsp
    @rput lgd

    
R"""
cellfn <- function(j,i,x,y,width,height,fill){
  grid.rect(x=x,y=y,height=height,width=width,gp = gpar(fill = colmat[i, j], col = NA))
}

hm <- Heatmap(mat,col=colmat,
                cell_fun=cellfn,
                split=rowsp,column_split=colsp,
                row_order=sel, column_order=sel,
                cluster_rows=FALSE, cluster_columns=FALSE,
                cluster_row_slices=FALSE, cluster_column_slices=FALSE,
                show_heatmap_legend=FALSE,border=TRUE);
pdf($out)
draw(hm,annotation_legend_list=lgd)
dev.off()
"""
end


