descend <- function(range,f,...,breaks,rate,maxiter,
		    x=0,fwhich=which.max,discrete=F,min=0,
		    trace=matrix(nrow=0,ncol=2),parallel=T){
	require(parallel)
	if(parallel){
		applyfn <- function(...){
			ncore <- min(breaks,detectCores()-2)
			cl <- makeCluster(ncore,"FORK")
			res <- parSapply(cl,...)
			stopCluster(cl)
			return(res)
		}
	}else{applyfn <- sapply}

	res <- list(argmax=x,trace=trace)
	if(nrow(trace)/breaks==maxiter) return(res)

	if(discrete){
		i <- setdiff(round(range[1]):round(range[2]),
			     trace[,1])
		if(length(i)==0) return(res)
		if(length(i)>breaks){
			i <- sample(i,breaks)
		}
	}else{
		i <- runif(breaks,range[1],range[2])
	}
	print(as.character(i))

	vals <- applyfn(i,f,...)
	if(length(dim(vals)>1)) vals <- t(vals)
	res$trace <- rbind(trace,cbind(i,vals))
	res$trace <- res$trace[order(res$trace[,1]),]
	sel <- fwhich(res$trace[,2])
	res$argmax <- res$trace[sel,1]
	win <- (range[2]-range[1])/2*rate

	range.o <- c(res$argmax-win,res$argmax+win)
	if(range.o[1]<min) range.o[1] <- min

	descend(range.o,f,...,breaks=breaks,rate=rate,
		maxiter=maxiter,x=res$argmax,fwhich=fwhich,
		discrete=discrete,min=min,trace=res$trace)
}

gsea.prot <- function(k,dists,groups,int,mode='directed'){
	require(fgsea)
	require(igraph)

	g <- get.knn(dists,k,mode=mode)

	ge <- get.edgelist(g)
	edges <- cbind(groups[ge[,1]],groups[ge[,2]])
	edges <- apply(edges,1,paste,collapse='->')
	ct <- fgsea(list(edge.ct=int),table(edges))
	return(ct$ES)
}

test.leiden <- function(res,k,g,dat,reps,groups,int,...){
	require(cluster)
	require(leiden)

	dists <- as.matrix(dist(dat))

	clust <- leiden(g,resolution_parameter=res)
	if(all(clust==1)) return(c(rep(0,6),clust))

	err <- test.knn(dat,k,clust,reps=reps)

	es <- test.leiden.gsea(g,clust,groups,int)
	sil <- mean(silhouette(clust,dists)[,3])

	g.gene <- gene.network(g,res,groups)
	recall <- score.gene.network(g.gene,int)

	stat <- err*es*sil*recall
	return(c(stat=stat,ES=es,recall=recall,log2error=err,
		 mean_silhouette=sil,
		 nclust=max(clust),clust))
}

test.knn <- function(dat,k,clust,sample=0.9,reps=1000,...){
	n.train <- round(nrow(dat)*sample)
	cv <- replicate(reps,knn.err(dat,k,clust,n.train))
	cv[!is.finite(cv)] <- 1/(2+n.train)
	err <- mean(cv)
	mse <- mean(cv^2)
	#         return(c(error=err,MSE=mse))
	return(-log2(err))
}

knn.err <- function(dat,k,clust,n.train){
	require(class)

	trainsel <- sample(1:nrow(dat),n.train)
	traindat <- dat[trainsel,]
	testdat <- dat[-trainsel,]

	cv <- class::knn(traindat,testdat,clust[trainsel],k)
	err <- sum(cv!=clust[-trainsel])/n.train
	return(err)
}

test.leiden.gsea <- function(g,clust,groups,int,...){
	require(fgsea)
	require(leiden)

	e.group <- edge.factor(group.edge(g,groups),T)

	e.clust <- group.edge(g,clust)

	is.cis <- e.clust[,1]==e.clust[,2]
	scores <- table(e.group[is.cis])/table(e.group[!is.cis])
	scores[!is.finite(scores)] <- max(Filter(is.finite,
						 scores))
	cl <- fgsea(list(interactions=int),scores)

	return(cl$ES)
}

group.edge <- function(g,groups){
	require(igraph)
	e <- get.edgelist(g)
	e <- cbind(groups[e[,1]],groups[e[,2]])
	return(e)
}

edge.factor <- function(e,as.factor=T) {
	res <- apply(e,1,paste, collapse='->')
	if(as.factor) res <- as.factor(res)
	return(res)
}

get.res <- function(range,breaks,rate,maxiter,
		    k,dat,int,groups,...){
	require(igraph)
	dists <- as.matrix(dist(dat))
	g <- get.knn(dists,k)
	descend(range,test.leiden,
		k=k,g=g,dat=dat,reps=1000,groups=groups,
		int=int,breaks=breaks,rate=rate,
		maxiter=maxiter,min=0.05,
		trace=matrix(nrow=0,ncol=7+nrow(dat)),...)
}

gene.network <- function(g,res,groups,
			 mode='directed',weighted=T,diag=F,...){
	require(igraph)
	gene.edge <- group.edge(g,groups)
	K.out <- sapply(split(igraph::degree(g,mode='out'),groups),sum)
	K.in <- sapply(split(igraph::degree(g,mode='in'),groups),sum)

	fy <- function(y,x){ 
		e <- sum(gene.edge[,1]==x&gene.edge[,2]==y) 
		e-(res*((K.out[[x]]*K.in[[y]])/(2*sum(igraph::degree(g)))))
	}

	fx <- function(x) { 
		sapply(unique(groups),fy,x)
	}

	clust.modularity <- sapply(unique(groups),fx)

	clust.modularity[clust.modularity<0] <- 0
	g.gene <- graph_from_adjacency_matrix(clust.modularity,mode,weighted,diag,...)
	return(g.gene)
}

score.gene.network <- function(g,int){
	e.gene <- edge.factor(get.edgelist(g),F)
	recall <- length(intersect(e.gene,int))/length(int)
	return(recall)
}

get.knn <- function(dists,k,mode='directed'){
	require(igraph)
	neighbors <- apply(dists,2,order)
	adj <- sapply(1:ncol(neighbors),function(i){
			      r <- dists[,i]
			      sel <- neighbors[-1:-k,i]
			      r[sel] <- 0
			      return(r)
	})
	g <- graph_from_adjacency_matrix(adj,mode,T,F)
}

par.apply <- function(...,f=parSapply){
	require(parallel)
	ncore <- detectCores()-2
	cl <- makeCluster(ncore,"FORK")
	res <- f(cl,...)
	stopCluster(cl)
	return(res)
}

get.k <- function(range,dists,group,int,mode='plus'){
	res <- par.apply(range,gsea.prot,dists,group,int,
			  mode=mode)
	return(cbind(k=range,ES=res))
}

res.unif <- function(range,k,dat,int,groups,reps=1000){
	require(parallel)
	require(igraph)

	dists <- as.matrix(dist(dat))
	g <- get.knn(dists,k,'plus')

	res <- runif(reps,range[1],range[2])
	out <- par.apply(res,test.leiden,
		k=k,g=g,dat=dat,reps=1000,groups=groups,
		int=int)
	return(cbind(res,t(out)))
}

get.leidens <- function(range,k,dat){
	require(igraph)
	require(leiden)
	 
	dists <- as.matrix(dist(dat))
	g <- get.knn(dists,k,'plus')

	res <- runif(reps,range[1],range[2])
	clust <- par.apply(leiden,resolution_parameter=res,
			   MoreArgs=list(object=g),
			   f=clusterMap)

	return(cbind(res,t(clusts)))
 }

graph.pdf <- function(out,g,layout=layout_nicely,...,append.date=T){
	dir.pdf(out,append.date=append.date)
	plot(g,layout=layout,...)
	dev.off()
}

#' Two-tailed version of \code{\link{phyper}}. It simply takes the minimum of the upper and lower tails and multiplies the result by 2.
#' @param ... Arguments to \code{phyper()}.
#' @export
#' @seealso \code{\link{phyper}}
phyper2 <- function(...) min(
	phyper(...,lower.tail=T),
	phyper(...,lower.tail=F)
)*2


graph.hyper <- function(g,conds,x,y){
	require(igraph)
	gene.edge <- group.edge(g,conds)
	e <- sum(gene.edge[,1]==x&gene.edge[,2]==y) 
	m <- sum(degree(g)[conds==x])
	k <- sum(degree(g)[conds==y])
	n <- sum(degree(g))-m
	p <- phyper(e-1,m,n,k)
	odds <- (e/(k-e))/(m/n)
	return(c(log2OR=odds,p=p,q=e))
}

get.hyper <- function(g,conds,padj.method="fdr"){
	require(purrr)
	condl <- unique(conds)
	hyper <- mapply(partial(mapply,
				partial(graph.hyper,knn,
					conds),
				condl),
			condl,SIMPLIFY=F)
	hyper <- sapply(c('log2OR','p','q'),
			partial(sapply,hyper,'['),,
			simplify=F)
	hyper$p <- apply(hyper$p,2,p.adjust,
			 method=padj.method,
			 n=length(condl))
	return(hyper)
}
