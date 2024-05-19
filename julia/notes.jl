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

    C = K * E' ./ n

    Ehat = (P * E')'
    Chat = K * Ehat' ./ n

    topleftfill = zeros(k,k)
    bottomrightfill = zeros(d,d)
    
    hmdat = Dict([(:E,E),
                (:K,K),
                (:P,P)])

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
    
    cellfn = (j,i,x,y,width,height)->rgrid.grid_rect(x=x,y=y,width=width,
                                                     gp=rgrid.gpar(fill=colmat[i,j]))
    hm = CH.Heatmap(mat,col=colmat,
                    cell_fun=rcopy(cellfn),
                    split=rowsp,column_split=colsp,
                    row_order=sel, column_order=sel,
                    cluster_rows=false, cluster_columns=false,
                    cluster_row_slices=false, cluster_column_slices=false,
                    show_heatmap_legend=false);

    grD.pdf(out)
    CH.draw(hm,annotation_legend_list=lgd)
    grD.dev_off()
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


macro cellfn()
R"""
cellfn <- function(j,i,x,y,width,height,fill){
  grid.rect(x=x,y=y,height=height,width=width,gp = gpar(fill = $(colmat)[i, j], col = NA))
}
"""
end

macro KEheatmap(mat,colmat,
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

R"""
layerfn <- function(j,i,x,y,width,height,slice_r,slice_c){
        grid.fill(
}
"""
R"""
library(grid)
layerfn <- function(j,i,x,y,width,height,fill,slice_r,slice_c){
  fillmat <- hmfill[[layout[slice_r,slice_c]]]
  print(fillmat)
  n <- nrow(fillmat)
  m <- ncol(fillmat)
  
  pushViewport(viewport(layout = grid.layout(m, n)))
  
  for (i in 1:m) {
    for (j in 1:n) {
      pushViewport(viewport(layout.pos.row = i, layout.pos.col = j))
      grid.rect(gp = gpar(fill = fillmat[i, j], col = NA))
      popViewport()
    }
  }
}

cellfn <- function(j,i,x,y,width,height,fill){
  grid.rect(x=x,y=y,height=height,width=width,gp = gpar(fill = colmat[i, j], col = NA))
}


draw_color_heatmap <- function(color_matrix) {
  nrows <- nrow(color_matrix)
  ncols <- ncol(color_matrix)
  
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(nrows, ncols)))
  
  for (i in 1:nrows) {
    for (j in 1:ncols) {
      pushViewport(viewport(layout.pos.row = i, layout.pos.col = j))
      grid.rect(gp = gpar(fill = color_matrix[i, j], col = NA))
      popViewport()
    }
  }
}

# Example usage
color_matrix <- matrix(c("#FF0000", "#00FF00", "#0000FF", "#FFFFFF"), nrow = 2, byrow = TRUE)
draw_color_heatmap(color_matrix)
"""
