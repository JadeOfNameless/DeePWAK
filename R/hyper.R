writepdf <- function(expr,file,out='.',...){
    pdf(paste0(out,'/',file),...)
    tryCatch(expr,finally=dev.off())
}
        
clusthyper <- function(dat, clusts, out,...){
	# run hypergeometric tests for enrichment of conditions and phenotypes
	hyper <- lapply(dat, 
			function(x) condhyper(row.names(dat),
					      x,clusts))

	# extract fields from test & reformat as matrices
	odds <- do.call(rbind,lapply(hyper,'[[','log2OR'))
	fdr <- do.call(rbind,lapply(hyper,'[[','FDR'))
	qval <- as.matrix(do.call(rbind,
				  lapply(hyper,'[[','q')))

	# split phenotype & condition into separate panels
	#         if(length(dat)>1){
	rowsplit <- unlist(
		mapply(
		       function(x,y) rep(y,nrow(x$log2OR)),
		       hyper,
		       names(hyper)
		)
	)
	#         } else rowsplit <- NULL

	# write matrices to csv
	dir.csv(cbind(
		parameter=rowsplit,
		condition=row.names(odds),
		as.data.frame(odds)
	),'log2OR', out, append.date=F)
	dir.csv(cbind(
		parameter=rowsplit,
		condition=row.names(odds),
		as.data.frame(fdr)
	),'FDR', out, append.date=F)
	dir.csv(cbind(
		parameter=rowsplit,
		condition=row.names(odds),
		as.data.frame(qval)
	),'size', out, append.date=F)

        writepdf({
    pdf(paste0(out,'/hyper.pdf'))
            dotplot(
                odds,
                fdr,
                qval,
                row_split=rowsplit,
                row_title_rot=0,...)
    dev.off()
        },'hyper.pdf',out)
}


#' Hypergeometric test for enrichment of conditions in a cluster.
#' @param id A vector of sample IDs.
#' @param conds A vector of the same length as ID giving the condition of each sample.
#' @param clusts A vector of the same length as ID giving the cluster ID of each sample.
#' @param padj.method Method passed to \code{\link{p.adjust}} for multiple hypothesis correction.
#' @seealso \code{\link{phyper2}}, \code{\link{p.adjust}}
#' @export
condhyper <- function(id,conds,clusts,padj.method='fdr'){
	#         id <- id[!is.na(conds)]
	#         cols <- unique(clusts)
	#         clusts <- clusts[!is.na(conds)]
	test <- split(id,conds)
	clusts <- split(id,clusts)
	#         fn <- function(x) sum(!is.na(x))
	m <- sapply(test,length)
	n <- length(id)-m
	k <- sapply(clusts,length)
	q <- as.data.frame(sapply(clusts,
		  function(m) {
			  sapply(test, 
				 function(k){
					sum(m%in%k) 
				 })
		  }))
	log2OR <- mapply(
	  function(q.i,k.i) mapply(
	    function(q.ij,m.j) log2(
	      (q.ij/(k.i-q.ij))/(m.j/(length(id)-m.j))
	    ),
	    q.i,m
	  ),
	  q,k
	)
	row.names(log2OR) <- names(test)

	testHyper <- mapply(function(q,k) mapply(
	  phyper2,q=q-1,k=k,m=m,n=n
	),q=q,k=k)
	testFdr <- apply(testHyper,2,p.adjust,method=padj.method)
	row.names(testHyper) <- names(test)
	return(list(log2OR=log2OR,FDR=testFdr,q=as.matrix(q)))
}

#' Two-tailed version of \code{\link{phyper}}. It simply takes the minimum of the upper and lower tails and multiplies the result by 2.
#' @param ... Arguments to \code{phyper()}.
#' @export
#' @seealso \code{\link{phyper}}
phyper2 <- function(...) min(
	phyper(...,lower.tail=T),
	phyper(...,lower.tail=F)
)*2
