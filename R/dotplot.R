#' Wrapper for \code{Heatmap()} which allows specifying cell dimensions and resizing the heatmap accordingly.
#'
#' @param x A numeric matrix to be plotted.
#' @param ... Additional arguments to \code{Heatmap()}.
#' @param cell.h The cell height.
#' @param cell.w The cell width.
#' @param height The heatmap height. Ignored if \code{cell.h} is specified.
#' @param width The heatmap width. Ignored if \code{cell.w} is specified.
#' @param units The unit scale to be used for \code{cell.h} and \code{cell.w}.
#' @return A ComplexHeatmap.
#' @import ComplexHeatmap
#' @export
hm.cell <- function(
		x,...,
		cell.h=NULL,cell.w=NULL,
		height=NULL,width=NULL,
		#                 heatmap_height=NULL, heatmap_width=NULL,
		units='in'
){
	if(!is.null(cell.h)) height <- unit(nrow(x)*cell.h,units)
	if(!is.null(cell.w)) width <- unit(ncol(x)*cell.w,units)
	return(Heatmap(x,...,height=height,width=width))
}

#' accepts the results of an enrichment test applied to each cell in a matrix
#' and writes a dotplot of the results

#' @param mat A matrix of values shown by the dot color.
#' @param outl A matrix of values shown by the dot outlline.
#' @param size A matrix of values shown by the dot size.
#' @param col.mat A color scale for \code{mat}.
#' @param col.outl A color scale for \code{outlline}.
#' @param scale A vector of length 2 giving the min and max values to scale the size of the dots.
#' @param cell.dim The width & height of each heatmap cell in inches.
#' @param ... Additional arguments to \code{hm.cell()}.
#' @export
#' @importFrom grid gpar unit grid.points
hmdot <- function(
	mat, outl, size, 
	col.mat, col.outl, scale, size.breaks,
	mat.name="log2(OR)", outl.name="size", size.name="-log10(FDR)", 
	#file, path='.',
        cell.dim=.15,#, width=12, height=12, append.date=F,
	...
){
    require(ComplexHeatmap)
	#         outl <- -log10(outl)
	#         mat[mat<0] <- 0
	mat[is.na(mat)] <- 0
	mat[mat==Inf] <- max(mat[is.finite(mat)])
	mat[mat==-Inf] <- min(mat[is.finite(mat)])
	#         outl[!is.finite(outl)] <- 0

	#         cexfn <- function(x) unit(min(x/-log10(0.001),1)*cell.dim,'in')
	cexfn <- function(x) unit((1.2*x/max(size))*cell.dim,'in')
	#         col.mat <- col.z(mat)
	#         col.outl <- col.abs(outl)
	#         col.outl <- colorRamp2(c(0,2),c('white','black'))
	cellfn <- function(j, i, x, y, width, height, fill) {
            grid.points(
		x = x, y = y, 
		size=cexfn(size[i, j]),
		pch=16,
                gp = gpar(
			col = col.mat(mat[i, j]) 
			#                         col = col.outl(outl[i, j])
		)
	    )
            grid.points(
		x = x, y = y, 
		size=cexfn(size[i, j]),
		pch=1,
                gp = gpar(
			col = col.outl(outl[i, j])
		)
	    )
        }

	#         mat.breaks <- round(seq(range(attr(col.mat,'breaks'), length.out=6)))
	#         outl.breaks <- round(seq(range(attr(col.outl,'breaks'), length.out=6)))

	lgd <- list(
		Legend(col_fun = col.mat, title = mat.name),# at=mat.breaks),
		Legend(col_fun = col.outl, title = outl.name),# at=outl.breaks),
		Legend(
			title=size.name,
			at=size.breaks,
			type='points',
			background=0,
			pch=16,
			size=unit(sapply(size.breaks,cexfn),'in'),
			legend_gp=gpar(col=1,fill=0)
		)
	)

	hm <- hm.cell(
		mat,
		cell_fun=cellfn,
		#                 name='log2OR',
		#                 col=col.mat,
		rect_gp = gpar(type = "none"),
		cell.w=cell.dim,
		cell.h=cell.dim,
		#                 show_column_dend=F,
		#                 show_row_dend=F,
		show_heatmap_legend=F,
		...
	)

	#dir.pdf(
	#  file,path,
	#  append.date=append.date,
	#  width=width,
	#  height=height
	#)
	draw(hm, annotation_legend_list=lgd)
	#dev.off()
}

#' @param dot dot color matrix
#' @param size dot size matrix
#' @param outline dot outline color matrix
#' @param sizelim Maximum value on the size scale. Values above \code{sizelim} are set to \code{sizelim}.
#' @param ... Additional arguments to \code{hmdot()}.
#' @export
dotplot <- function(dot, size, outline, sizelim=-log10(0.01), outl.name='size', size.name='-log10(FDR)', ...){
	size[which(size<sizelim)] <- sizelim
	#logFDR <- -log10(size)
	dot[is.na(dot)] <- 0
	dot[dot==Inf] <- sizelim
	dot[dot==-Inf] <- 0
	size[!is.finite(size)] <- 0

	col.dot <- col.z(dot)

	outline[!is.finite(outline)] <- 0
	outlinescale <- c(0,quantile(as.matrix(outline)[as.matrix(outline)!=0], 0.95))
	col.outl <- colorRamp2(outlinescale ,c('white','black'))

	size.breaks <- seq(0, sizelim, length.out=6)

	hmdot(
	      dot, 
	      outline, 
	      size, 
	      col.mat=col.dot, 
	      col.outl=col.outl, 
	      scale=outlinescale, 
	      size.breaks=size.breaks,
	      outl.name=outl.name,
	      size.name=size.name,
	      ...
	)
}
