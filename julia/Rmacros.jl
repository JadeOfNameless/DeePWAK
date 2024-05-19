using RCall, Match

@rimport stats
@rimport ComplexHeatmap as CH
@rimport circlize

macro rsource()
    R"""
    f <- list.files("R","\\.R$",full.names=T)
    sapply(f,source)
    """
end

@rsource

macro leiden(G)
    q = R"""
    library(leiden)
    leiden($G)
    """
    return rcopy(q) |> Array{Int}
end

macro Heatmap(args,out)
    R"""
    library(ComplexHeatmap)
    pdf($(out))
    draw(do.call(Heatmap,$(args)))
    dev.off()
    """
end

macro HeatmapScale(breaks,cols,args,out)
    return quote
        breaks = $(esc(breaks))
        cols = $(esc(cols))
        args = $(esc(args))
        out = $(esc(out))
        R"""
        library(ComplexHeatmap)
        library(circlize)
        col <- colorRamp2($(breaks),$(cols))
        args <- $(args)
        args$col <- col

        pdf($(out))
        draw(do.call(Heatmap,args))
        dev.off()
        """
    end
end


macro Rfn(f,argnames,args)
    R"""
    argnames <- names(formals(args($(f))))
    """
    @rget argnames
    return quote
        f = $(esc(f))
        argnames = $(esc(argnames))
        args = $(esc(args))
        R"""
        args <- $(args)
        argnames <- $(argnames)
        names(args) <- argnames
        do.call($f,args)

        """
    end
end

function clusthyper(out,cond,clust;kwargs...)
    args = [cond,clust,out]
    rsendargs(args,kwargs...)
    R"""
args[[1]] <- as.data.frame(args[[1]])
do.call(clusthyper,args)
    """
end

function rsendargs(args;kwargs...)
    args = map(1:length(args)) do i
        ("arg"*string(i),args[i])
    end |> Dict

    kwargs = Dict([kwargs...])
    @rput args
    @rput kwargs
    R"""
names(args) <- NULL
args <- append(args,kwargs)
    """
end

function heatmap(out, dat, mid; kwargs...)
    rsendargs([dat],kwargs...)
    R"""
args$col <- col.z(args[[1]],mid=$mid)
pdf($(out))
draw(do.call(Heatmap,args))
dev.off()
    """
end

function heatmap(out, X; kwargs...)
    rsendargs([X]; kwargs...)

    if minimum(X) >= 0
        #mid = (median ∘ not0)(X)
        R"""
args$col <- col.abs(args[[1]])
         """
    else
        R"""
args$col <- col.z(args[[1]])
         """
    end
    #@rput mid

    R"""
pdf($(out))
draw(do.call(Heatmap,args))
dev.off()
    """
end

function combinedheatmap(out,dict;kwargs...)
    mat = vcat(values(dict)...)
    colsp = mapreduce(vcat,zip(values(dict),keys(dict))) do (M,lab)
        m,n = size(M)
        return rep(lab,m)
    end
    heatmap(out*".pdf",mat',column_split = colsp;kwargs...)
end

function batchhyper(dict)
    @rput dict
    R"""
batchsize <- length(dict[[1]])
hyper <- sapply(dict,function(cond) lapply(dict, function(clust) condhyper(1:batchsize,cond,clust)))
xlab <- sapply(hyper[,1],function(x) colnames(x$log2OR))
ylab <- sapply(hyper[1,],function(x) row.names(x$log2OR))
    """
    @rget hyper
    @rget xlab
    @rget ylab
    return hyper,xlab,ylab
end


function combinedhyper(out,dict;kwargs...)

    hyper,xlab,ylab = batchhyper(dict)
    xlab = vcat(values(xlab)...)
    ylab = vcat(values(ylab)...)
                
    clust = dictcat(hyper)
    sp = vcat(repkey_clust(dict)...)
    
    rsendargs([clust[:log2OR],clust[:FDR],clust[:q]],
              split = sp, column_split=sp,border = true,
              #column_title = "cluster",row_title = "enriched in";
              kwargs...)
    @rput xlab
    @rput ylab
    R"""
row.names(args[[1]]) <- ylab
colnames(args[[1]]) <- xlab
pdf($out)
do.call(dotplot,args)
dev.off()
    """
end
    

function rcall(f,args,kwargs...)
    rsendargs(args, kwargs...)
    R"""
res <- do.call($f,args)
    """
    @rget res
    return res
end


function maskedheatmap(out,fill,mask,cols,legend;kwargs...)
    sel = (collect ∘ keys)(fill)
    lims = map(extrema,fill)

    mat = map(sel) do i
        return fill[i] ./ maximum(abs.(lims[i]))
    end |> sum

    m,n = size(mat)
    colfns = map(i->rcall("col.abs",[fill[i],0,["white",cols[i]]]),sel)

    colfns = map(i->circlize.colorRamp2([lims[i]...],["white",cols[i]]),sel)
    colmat = map((f,i)->(rcopy ∘ f)(fill[i]),colfns,sel)

    fillmat = (rcopy ∘ colfns[1])(zeros(m,n))
    for (M,i) in zip(colmat,sel)
        fillmat[mask[i]] .= M[mask[i]]
    end
                            
    fillmat = reduce((fill,x)->begin
                         M,i = x
                         fill[mask[i]] = M[mask[i]]
                         return fill
                     end,zip(colmat,sel),init=fillmat)
    
    hm = ComplexHeatmap.Heatmap(mat,col=fillmat,
                                split=rowsp, column_split=colsp,
                                show_column_dend=false,show_row_dend=false,
                                show_heatmap_legend=false);
    
    lgd = map((f,i)->ComplexHeatmap.Legend(col_fun=f,title=legend[i]),
              colfns,sel)
    rdraw("tmp.pdf",hm,annotation_legend_list=lgd)
end

macro rpdfopen(out)
    quote
    R"""pdf($(esc(out)))"""
    end
end

macro rdevoff()
    quote
    R"""dev.off()"""
    end
end


macro rpdf(out, expr)
    return quote
        @rpdfopen ($(esc(out)))
        try
            $(esc(expr))
        finally
            @rdevoff
        end
    end
end

    

macro rpdfopen(out)
    quote
        R"pdf($(esc(out)))"
    end
end

macro rdevoff()
    quote
        R"dev.off()"
    end
end

macro rpdf(out, expr)
    esc_out = esc(out)
    esc_expr = esc(expr)

    return quote
        @rpdfopen $esc_out
        $esc_expr
        @rdevoff
    end
end
using RCall

function rdraw(filename, hm;kwargs...)
    rsendargs([hm];kwargs...)
    R"""
pdf($filename)
do.call(draw,args)
dev.off()
    """
end

