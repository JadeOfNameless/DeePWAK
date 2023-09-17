silhouettePlot <- function(dat, clusts, distmat=NULL, out){
	require(dirfns)
	require(cluster)
	pcs <- dat[,grep('pca', names(pcs))]
	pcs <- dat[,!apply(pcs==0,2,all)]

	row.names(pcs) <- row.names(dat)

	if(is.null(distmat)) {
		   dists <- dist(pcs)
	}else dists <- distmat

	sil <- silhouette(clusts, dists)

#	if(!is.na(sil)){
		dir.pdf('silhouette', out, append.date=F)
		plot(sil)
		dev.off()
#	}
	return(sil)
}

plotDist <- function(dat,conds,distmat,out,clustcol='gray',...){
	require(dirfns)

	dat <- dat[,grep('umap', names(dat))]
	cols <- c(Depdc='royalblue',Tyrosinase='saddlebrown')
	condcol <- cols[conds]
	#         dists <- distmat[row.names(dat), row.names(dat)]

	ix <- which(distmat>0, arr.ind=T)
	clustline <- do.call(rbind, 
			     apply(ix,1, 
				   function(x) rbind(dat[x[1],],dat[x[2],])))

	dir.pdf('Tyr_Depdc', out, append.date=F)
	plot(NULL,
	     xlim=range(dat[,1]), ylim=range(dat[,2]), 
	     xlab=NA, ylab=NA, xaxt='n', yaxt='n')
	lines(clustline, col=clustcol)
	points(dat,cex=1.0,col=condcol,pch=19)
	legend(
	       'bottomleft',
	       names(cols), 
	       col=cols,
	       pch=19,
	       cex=1.0
	)
	dev.off()
}

clustplots <- function(dat, clusts, conds, distmat, 
		       out='.', append.date=T,
		       legend.ncol=2,legend.cex=0.5){
	require(dirfns)
	require(cluster)
	row.names(dat) <- as.character(1:nrow(dat))
	umap <- as.data.frame(dat)
	if(length(clusts)>0) {
		clustpts <- split(umap, clusts)
	}else clustpts <- NULL
	if(length(conds)>0) {
		condpts <- split(umap, conds)
	} else condpts <- NULL

	clustcol <- rainbow(length(clustpts))
	condcol <- rainbow(length(condpts))

	clustdist <- lapply(clustpts, 
			    function(x) {
				    distmat[row.names(x), 
					    row.names(x)]
			    })

	clustline <- mapply(function(dists, pts){
		ix <- which(dists>0, arr.ind=T)
		do.call(rbind, 
			apply(ix,1, 
			      function(x) { 
				      rbind(pts[x[1],], 
					    pts[x[2],])
			      }))
	}, clustdist, clustpts, SIMPLIFY=F)

	dir.pdf('clustedge', out, append.date=append.date)
	plot(NULL, xlim=range(umap[,1]), 
	     ylim=range(umap[,2]), xlab=NA, ylab=NA, 
	     xaxt='n', yaxt='n')
	mapply(lines, clustline, col=clustcol)
	mapply(points, condpts, col=condcol, cex=0.5)
	legend(
	       'bottomleft',
	       c(names(condpts), names(clustpts)),
	       col=c(condcol, clustcol),
	       lty=c(rep(NA, length(condpts)), 
		     rep(1, length(clustpts))),
	       pch=c(rep(1, length(condpts)), 
		     rep(NA, length(clustpts))),
	       ncol=legend.ncol, cex=legend.cex
	)
	dev.off()

	dir.pdf('clustplot', out, append.date=append.date)
	plot(NULL, xlim=range(umap[,1]), 
	     ylim=range(umap[,2]), xlab=NA, ylab=NA, 
	     xaxt='n', yaxt='n')
	mapply(points, condpts, col=condcol, cex=0.5, pch=19)
	mapply(points, clustpts, col=clustcol, cex=0.5)
	legend(
	       'bottomleft',
	       c(names(condpts), names(clustpts)),
	       col=c(condcol, clustcol),
	       pch=c(rep(19, length(condpts)), 
		     rep(1, length(clustpts))),
	       ncol=2, cex=0.5
	)
	dev.off()
}

clustparam <- function(dat, clusts, out){
	require(moreComplexHeatmap)
	clustdat <- split(dat,clusts)
	# test for significance of each feature in each cluster
	testfn <-function(x, clust, dat){
	      mu <- mean(dat[,x],na.rm=T)
	      FC <- mean(clust[,x],na.rm=T)/mu
	      if(FC!=1&length(unique(clust[,x]))>1){
		      utest <- wilcox.test(clust[,x],mu=mean(dat[,x], na.rm=T))$p.value
		      ttest <- t.test(clust[,x],mu=mean(dat[,x], na.rm=T))$p.value
	      }else{
		      utest <- NaN
		      ttest <- NaN
	      }
	      return(c(FC=FC,u=utest,t=ttest))
	} 
	clusttest <- function(clust){
	       sapply(colnames(clust), testfn, clust, dat)
	}
	test <- lapply(clustdat, clusttest)

	# select fields from output
	fcdat <- sapply(test,'[',"FC",)
	p.t <- sapply(test,'[','t',)
	p.u <- sapply(test,'[','u',)

	# convert p-values to FDR values
	fdr.t <- apply(p.t,2,function(x) p.adjust(unlist(x)))

	# define color scale
	clustdat <- lapply(clustdat, function(x) t(x)[row.names(fcdat),])
	# col.z creates a color scale centered on 0 ranging from the 0.01 to 0.99 quantiles of the input data
	if(length(unique(clusts))>1){
	# This scale will be used for the fold changes between average feature values in each cluster from the background.
		colfc <- col.z(fcdat,.05,1)
		# creates a color scale from 0 to 2
		# This will give a log10 scale for FDR values between 1 and 0.01
		colfdr <- colorRamp2(c(0,2),c('white','black'))

		# show feature values within clusters as boxplots color-coded by FDR value
		getAnn <- function(x) {
			m <- clustdat[[x]]
			fc.cols <- colfc(fcdat[,x])
			fdr.cols <- colfdr(fdr.t[,x])
			return(anno_boxplot(
				m, 
				which='row',
				width=unit(1,'in'),
				box_width=0.9,
				gp=gpar(
					fill=fc.cols,
					col=fdr.cols
				)
			))
		}

		# apply annotation function to each cluster
		ha <- lapply(colnames(fcdat),getAnn)
		names(ha) <- paste('cluster',colnames(fcdat))
		# bind annotation into single object
		ha <- do.call(HeatmapAnnotation,append(ha,list(which='row')))

		# assign names to color scales
		lgd <- list(
			Legend(col_fun = colfc, title = "FC"),
			Legend(col_fun = colfdr, title = "-log10(FDR)")
		)

		hm <- hm.cell(fcdat,
			right_annotation = ha,
			cell.w=.15,
			cell.h=.15,
			show_heatmap_legend=F,
			col=colfc
		)

		dir.pdf('t', out, height=24, width=10+ncol(fcdat), append.date=F)
		draw(hm, annotation_legend_list=lgd)
		dev.off()
	}
}

clusthyper <- function(dat, clusts, out){
	require(moreComplexHeatmap)
	# run hypergeometric tests for enrichment of conditions and phenotypes
	hyper <- lapply(dat, 
			function(x) condHyper(row.names(dat),
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

	if(length(unique(clusts))>1){
		dotPscale(
			odds, 
			fdr, 
			qval, 
			file='condition', 
			path=out, 
			row_split=rowsplit, 
			row_title_rot=0
		)
	}
}

getPois <- function(cond,dists){
	#select distances between points in the same condition
	condsel <- sapply(unique(cond), function(x) cond==x)
	#condition permutations to test
	conds <- combn(unique(cond),2)
	conds <- cbind(sapply(unique(cond),rep,2),conds)

	#number of embryos per condition
	ncond <- table(cond)
	#neighbors have dist>0
	n <- dists>0

	#get matrix of neighbors for each cond
	kn <- split(as.data.frame(dists>0),cond)

	#poisson test
	pois <- lapply(kn,function(x){
		       #split columns by condition
		       conds <- split(as.data.frame(t(x)),cond)
		       #total number of neighbors
		       k <- sum(as.matrix(x))
		       #test each condition
		       res <- lapply(conds,function(y){
				y <- as.matrix(y)
				#number of neighbors in test condition
				ct <- sum(y)
				#proportion of test condition to total embryos
				p <- nrow(y)/sum(ncond)
				pval <- poisson.test(ct,k,p)$p.value
				#proportional overrepresentation of test condition among all neighbors vs. expected
				odds <- log2((ct/k)/p)
				return(c(p=pval,log2OR=odds,count=ct))
			})
		       return(do.call(rbind,res))

		})
	odds <- sapply(pois,'[',,'log2OR')
	fdr <- sapply(pois,'[',,'p')
	qval <- sapply(pois,'[',,'count')
	return(list(log2OR=odds,p=fdr,count=qval))
}

dotPois <- function(pois,out,append.date=T){
	require(moreComplexHeatmap)

	dotPscale(
		pois$log2OR, 
		pois$p, 
		pois$count, 
		file='conditionEdgePois', 
		path=out, 
		row_title_rot=0,
		append.date=append.date
	)

	dir.csv(pois$log2OR,'conditionPoisLog2OR', 
		out, append.date=append.date)
	dir.csv(pois$p,'conditionPoisFDR', 
		out, append.date=append.date)
	dir.csv(pois$count,'conditionPoisCount', 
		out, append.date=append.date)
}

poisGraph <- function(pois,cutoff=0.05,up=T,mode='max'){
	require(igraph)

	edgeval <- pois$log2OR
	edgeval[which(!is.finite(edgeval))] <- 0
	edgeval[which(pois$p>cutoff)] <- 0
	if(up){
		edgeval[which(edgeval<0)] <- 0
	}
	g <- graph_from_adjacency_matrix(edgeval,
					 mode,T,F)
	if(up) { 
		colfn <- colorRamp2(c(0,max(E(g)$weight)),
				   c('white','red')) 
	}else colfn <- colorRamp2(c(min(E(g)$weight),0,
				    max(E(g)$weight)),
				  c('blue','white','red'))
	E(g)$edge.color <- colfn(E(g)$weight)

	return(g)
}

networkPois <- function(g,file,out='.',
			colfn = col.z(E(g)$weight),
			fn=plotNetwork,
			title='log2OR',
			layout=layout_nicely,...){
	require(igraph)

	E(g)$edge.color <- colfn(E(g)$weight)

	lgd <- seq(round(quantile(E(g)$weight,0.99)),
		   min(0,round(quantile(E(g)$weight,
					0.01))),
		   length.out=6)
	fn(g,file,out,colfn,lgd,title,layout,...)
}

plotNetwork <- function(g,file,out='.',colfn,lgd,
			title='log2OR',layout=layout_nicely,
			vertex.shape='none',
			vertex.size=10,...,
			append.date=T){
	tmp <- do.call(rbind,strsplit(as_ids(E(g)),'\\|'))
	curved <- sapply(1:nrow(tmp),function(x) { 
		y <- which(tmp[,1]==tmp[x,2]&tmp[,2]==tmp[x,1]) 
		if(length(y)>0) return(c(x,y))
	})

	curves <- curve_multiple(g)
	curves[unique(unlist(curved))] <- 0.3
	#         lgd <- seq(lgd[1],0,length.out=6)

	#         ldist <- distances(g,weights = abs(E(g)$weight))
	#         ldist[is.finite(ldist)&ldist>0] <- 1
	#         l <- layout_with_mds(g, dist = ldist, dim = 2, 
	#                         options = arpack_defaults)

	dir.pdf(file,out,append.date = append.date)
	E(g)$weight <- 1
	plot(g,vertex.shape=vertex.shape,
	     vertex.size=vertex.size,
	     layout=layout,
	     edge.curved=curves,
	     #autocurve=T,
	     edge.arrow.size=0.5,
	     #layout=l,
	     edge.color=E(g)$edge.color,#colfn(E(g)$weight),
	     ...)
	legend('topleft',as.character(lgd),fill=colfn(lgd),title=title)
	dev.off()

}

plotNetworkCircle <- function(g,file,out='.',colfn,lgd,title='log2OR',layout=layout.circle,append.date=T){
	dir.pdf(file,out,append.date = append.date)
	plot(g,vertex.shape='none',layout=layout.circle,edge.color=E(g)$edge.color)
	legend('topleft',as.character(lgd),fill=colfn(lgd),title=title)
	dev.off()
}

enrichCond <- function(cond,dists,out,layout=layout_nicely){
	require(moreComplexHeatmap)
	require(igraph)

	#number of embryos per condition
	ncond <- table(cond)

	pois <- getPois(cond,dists)
	dotPois(pois,out)

	g <- poisGraph(pois,up=F)
	colfn <- col.z(E(g)$weight)
	networkPois(g,'conditionEdgeNetwork',out,colfn,layout=layout.circle,vertex.shape='none')

	g.up <- poisGraph(pois)
	networkPois(g.up,'conditionEdgeNetworkUp',
		    out,colfn,layout=layout)
}

enrichCondDeprecated <- function(cond,dists,out){

	lgd <- seq(round(quantile(E(g)$weight,0.99)),
		   round(quantile(E(g)$weight,0.01)),length.out=6)

	dir.pdf('conditionEdgeNetwork',out,append.date = F)
	plot(g,vertex.shape='none',layout=layout.circle,edge.color=E(g)$edge.color)
	legend('topleft',as.character(lgd),fill=colfn(lgd),title='log2OR')
	dev.off()

	edgeval[edgeval<0] <- 0
	g.up <- graph_from_adjacency_matrix(edgeval,'directed',T,F)

	tmp <- do.call(rbind,strsplit(as_ids(E(g.up)),'\\|'))
	curved <- sapply(1:nrow(tmp),function(x) { 
		y <- which(tmp[,1]==tmp[x,2]&tmp[,2]==tmp[x,1]) 
		if(length(y)>0) return(c(x,y))
	})


	curves <- curve_multiple(g.up)
	curves[unique(unlist(curved))] <- 0.3
	lgd <- seq(lgd[1],0,length.out=6)

	ldist <- distances(g.up)
	ldist[is.finite(ldist)&ldist>0] <- 1
	l <- layout_with_mds(g.up, dist = ldist, dim = 2, 
			options = arpack_defaults)

	dir.pdf('conditionEdgeNetworkUp',out,append.date = F)
	plot(g.up,vertex.shape='circle',
	     layout=layout_nicely,
	     edge.curved=curves,
	     #autocurve=T,
	     edge.arrow.size=0.5,
	     #layout=l,
	     edge.color=colfn(E(g.up)$weight))
	legend('topleft',as.character(lgd),fill=colfn(lgd),title='log2OR')
	dev.off()

	test <- lapply(1:nrow(n),function(x){
			x <- n[x,]
			conds <- split(x,cond)
			k <- sum(x)
			n <- length(x)
			res <- lapply(conds,function(y){
				q <- sum(y)
				m <- length(y)
				p <- 1-phyper(q,m,n-m,k)
				odds <- log2((q/k)/(m/n))
				return(c(p=p,log2OR=odds))
			})
			return(do.call(rbind,res))
		})
	test <- split(test,cond)
	pois <- lapply(test,function(x){
		       res <- sapply(x,function(y) y[,1]<0.1&y[,2]>1.5)
		       ct <- apply(res,1,sum)
		       res <- mapply(poisson.test,ct,sum(ct),
			      ncond/sum(ncond),SIMPLIFY=F)
		       p <- sapply(res,'[[','p.value')
		       odds <- log2((ct/sum(ct))/(ncond/sum(ncond)))
		       return(cbind(
				bgCount=ncond,
				testCount=ct,
				bgConnectionFreq=ct/sum(ct),
				testConnectionFreq=ncond/sum(ncond),
				log2OR=odds,
				p=p,
				FDR=p.adjust(p)))
		})

	odds <- sapply(pois,'[',,'log2OR')
	fdr <- sapply(pois,'[',,'FDR')
	qval <- sapply(pois,'[',,'testCount')
	dotPscale(
		odds, 
		fdr, 
		qval, 
		file='conditionConnections', 
		path=out, 
		row_title_rot=0
	)

	edgeval <- odds
	edgeval[!is.finite(edgeval)] <- 0
	edgeval[fdr>0.05] <- 0
	g <- graph_from_adjacency_matrix(edgeval,'directed',T,F)

	colfn <- col.z(E(g)$weight)
	E(g)$edge.color <- colfn(E(g)$weight)

	lgd <- seq(round(quantile(E(g)$weight,0.99)),
		   round(quantile(E(g)$weight,0.01)),length.out=6)

	#l <- layout.sphere(g)
	dir.pdf('conditionNetwork',out,append.date = F)
	#plot(g,vertex.shape='none',layout=layout_with_lgl,edge.color=E(g)$edge.color)
	#plot(g,vertex.shape='rectangle',layout=l,edge.color=E(g)$edge.color)
	plot(g,vertex.shape='none',layout=layout.circle,edge.color=E(g)$edge.color)
	legend('topleft',as.character(lgd),fill=colfn(lgd),title='log2OR')
	dev.off()

	edgeval[edgeval<0] <- 0
	g.up <- graph_from_adjacency_matrix(edgeval,'directed',T,F)

	lgd <- seq(round(quantile(E(g)$weight,.99)),0,length.out=6)

	dir.pdf('conditionNetworkUp',out,append.date = F)
	plot(g.up,vertex.shape='circle',
	     #layout=layout_nicely,
	     #layout=layout.circle,
	     layout=layout.fruchterman.reingold,
	     edge.color=colfn(E(g.up)$weight))
	legend('topleft',as.character(lgd),fill=colfn(lgd),title='log2OR')
	dev.off()

	dir.pdf('conditionNetworkUpRing',out,append.date = F)
	plot(g.up,vertex.shape='none',
	     layout=layout.circle,
	     edge.color=colfn(E(g.up)$weight))
	legend('topleft',as.character(lgd),fill=colfn(lgd),title='log2OR')
	dev.off()
}

condDist <- function(clust, cond, dists, out){
	require(moreComplexHeatmap)

	condsel <- sapply(unique(cond), function(x) cond==x)
	conds <- combn(unique(cond),2)
	conds <- cbind(sapply(unique(cond),rep,2),conds)
	sel <- sapply(unique(clust), function(x) clust==x)
	colnames(sel) <- unique(clust)

	tmp <- lapply(as.data.frame(sel), function(x){
		x <- setNames(apply(conds,2,function(y){
			as.data.frame(dists[x&condsel[,y[1]],
				     x&condsel[,y[2]],drop=F])
		}),apply(conds,2,paste,collapse='.'))
		x <- lapply(x, function(x) x[x!=0])
		x
	})

	mat <- sapply(tmp,sapply,mean,na.rm=T)
	sp <- sub('\\..*','',row.names(mat))

	avg <- apply(conds, 2, function(x) mean(dists[condsel[,x[1]],condsel[,x[2]]]))
	avg <- split(avg,sp)
	avg <- do.call(c,sapply(avg,'[',-1))

	ct <- sapply(tmp,sapply,length)
	ct <- split(as.data.frame(ct),sp)
	ct <- do.call(rbind,lapply(ct,'[',-1,,drop=F))

	dat <- split(as.data.frame(mat),sp)
	lapply(dat, function(x) as.matrix(x)/x[1,])
	self <- lapply(tmp, function(x) x[1:length(unique(cond))])
	self <- sapply(self,sapply,mean,na.rm=T)
	row.names(self) <- sub('\\..*','',row.names(self))
	dat <- sapply(names(dat), function(x) t(t(dat[[x]])/self[x,])[-1,,drop=F])
	dat <- do.call(rbind,dat)
	sp <- sub('\\..*','',row.names(dat))
	row.names(dat) <- sub('.*\\.','',row.names(dat))

	n <- table(cond)
	n <- paste0(names(n),' (',as.character(n),')')
	sp <- Reduce(function(x,y) sub(sub('\\s.*','',y),y,x),n,sp)

	size <- apply(sel,2,sum)
	score <- apply(dat*ct,2,sum,na.rm=T)/apply(ct,2,sum)
	total <- log2(sum(score*size)/sum(size))

	#         topann <- list(anno_barplot(size),anno_barplot(score))
	#         names(topann) <- c('size',paste('score =',as.character(total)))
	#         topann <- do.call(columnAnnotation,topann)

	topann <- columnAnnotation(size=anno_barplot(size),
				   score=anno_barplot(score),
				   name=paste("score =",as.character(total)))
	quantHeatmap(
		log2(dat),'cond_dist',path=out,
		conds=row.names(dat), show_row_names=F,
		top_annotation=topann,
		right_annotation=rowAnnotation(avg.dist=anno_barplot(avg)),
		column_title=paste('total score =',as.character(total)),
		split=sp,row_title_rot=0
	)

	out <- lapply(tmp, function(x) x[length(unique(cond))+1:length(x)])
	mean(unlist(self))/mean(unlist(out))
}

#' Hypergeometric test for enrichment of conditions in a cluster.
#' @param id A vector of sample IDs.
#' @param conds A vector of the same length as ID giving the condition of each sample.
#' @param clusts A vector of the same length as ID giving the cluster ID of each sample.
#' @param padj.method Method passed to \code{\link{p.adjust}} for multiple hypothesis correction.
#' @seealso \code{\link{phyper2}}, \code{\link{p.adjust}}
#' @export
condHyper <- function(id,conds,clusts,padj.method='fdr'){
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
	return(list(log2OR=log2OR,FDR=testFdr,q=q))
}

#' Two-tailed version of \code{\link{phyper}}. It simply takes the minimum of the upper and lower tails and multiplies the result by 2.
#' @param ... Arguments to \code{phyper()}.
#' @export
#' @seealso \code{\link{phyper}}
phyper2 <- function(...) min(
	phyper(...,lower.tail=T),
	phyper(...,lower.tail=F)
)*2


