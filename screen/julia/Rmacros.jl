using RCall

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


