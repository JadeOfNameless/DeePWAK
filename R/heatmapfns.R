#' Color scale for a specified quantile. This scale is intended for heatmaps containing negative and positive values, so the range is set to \code{c(quant,1-quant)}.
#'
#' @param x A numeric matrix or vector.
#' @param quant The quantile to be used as the scale limits.
#' @param mid The midpoint of the scale.
#' @return A \code{colorRamp2} scale which can be passed to \code{Heatmap()}.
#' @importFrom circlize colorRamp2
#' @export
col.z <- function(x,quant=.01, mid=0, cols=c('blue', 'white', 'red')) {
	breaks <- c(quantile(x, quant, na.rm=T),
		    mid,
		    quantile(x, 1-quant, na.rm=T))
	colorRamp2(breaks, cols)
}


#' Color scale for a specified quantile. This scale is intended for heatmaps containing only positive values, so the range is set to \code{c(0,1-quant)}.
#'
#' @param x A numeric matrix or vector.
#' @param quant The quantile to be used as the upper limit.
#' @param cols The colors used for the color scale.
#' @return A \code{colorRamp2} scale which can be passed to \code{Heatmap()}.
#' @importFrom circlize colorRamp2
#' @export
col.abs <- function(x,quant=.05, cols=c('white','black')){
	breaks <- c(0,quantile(x[x!=0],1-quant, na.rm=T))
	breaks <- seq(breaks[1], breaks[2], length.out=length(cols))
	colorRamp2(breaks, cols)
}

#' Color scale for categorical data.
#'
#' @param cond A vector that can be coerced to a factor.
#' @param colfn A function that returns a color map for each level in \code{cond}.
#' @param ... Additional arguments to \code{colfn}.
#' @return A named vector of colors corresponding to the levels of \code{cond}.
#' @export
cond.col <- function(cond,colfn=rainbow,...){
	cond <- as.factor(cond)
	cols <- colfn(length(levels(cond)),...)
	names(cols) <- levels(cond)
	return(cols)
}

#' Creates a color scale for the levels in a vector, then returns a vector assigning a color to each element of the input vector.
#'
#' @param cond A vector that can be coerced to a factor.
#' @param colfn A function that returns a color map for each level in \code{cond}.
#' @param ... Additional arguments to \code{colfn}.
#' @return A vector of colors corresponding to the elements of \code{cond}.
#' @export
cond.col.vec <- function(cond,colfn=rainbow,...) {
	cond.col(cond,colfn=colfn,...)[as.numeric(as.factor(cond))]
}

