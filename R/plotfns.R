#source('modelfns.R')
#source('descend.R')
#source('clustplots.R')

dot.stat <- function(y,dat,...){
	require(ggplot2)
	ggplot(dat,
	       aes_string(x=names(dat)[1],
			  y=y,...))+geom_point()+
	  theme(plot.margin = margin(20))
}

dot.col <- function(y,dat,cols='',id='group',...){
	require(ggplot2)
	dat <- do.call(cbind,append(list(dat),setNames(list(cols),id)))
	dot.stat(y,dat,col=id,...)
}


dir.f <- function(f,file.arg='filename'){
	require(dirfns)
	function(...,filename='',ext='',path='.',append.date=T){
		out <- mkdate(filename,ext=ext,path=path,append.date=append.date)
		arglist <- append(list(...),setNames(out,file.arg))
		do.call(f,arglist)
	}
}

dir.plot <- function(out,outfn = dir.pdf){
	require(dirfns)
	function(f,...){
		outfn(out)
		tryCatch(f(...),finally=dev.off())
	}
}

arrange.stats <- function(plots,filename,ncols=3,
			  nrows=ceiling(length(plots)/ncols),
			  ...){
	require(ggpubr)
	require(dirfns)

	#         nrows <- ceiling(length(plots)/ncols)
	dat <- lapply(plots,'[[','data')
	#         rho <- sapply(dat,
	#                       function(x) cor(x[,1],
	#                                       x[,2],
	#                                       method='spearman'))

	#         title <- paste('Spearman correlation =',
	#                        as.character(rho))

	plots <- ggarrange(plotlist=plots,
			   labels=names(plots),
			   ncol=ncols,nrow=nrows)

	dir.f(ggexport)(plots,
			filename=paste0(filename,'.pdf'),
			...)
}

plot.edge <- function(dat,g,alpha=0.1,clusts=NULL,...){
	require(igraph)
	dists <- get.edgelist(g)
	f <- function(d,col) apply(d,1,
				   function(x) lines(dat[x,],
					col=col))

	#         dir.pdf(out)
	plot(NULL, xlim=range(dat[,1]), 
	     ylim=range(dat[,2]),
	     xlab=colnames(dat)[1],
	     ylab=colnames(dat)[2],...) 
	f(dists, col=rgb(0,0,0,alpha))

	if(length(clusts>0)){
		clustn <- unique(clusts)
		clustn <- clustn[order(clustn)]
		clustsel <- lapply(clustn, 
				 purrr::compose(which,
					 partial(`==`,clusts)))
		clustl <- lapply(clustsel,
				 function(x){
					 dists[do.call(`&`, as.data.frame(apply(dists, 2,`%in%`,x))),]
				 })
		cols <- rainbow(length(clustsel))
		mapply(f,clustl,cols)
		if(length(clustl)>1){
			legend(
			       'bottomleft',
			       as.character(clustn),
			       col=cols,
			       lty=1,
			)
		}
		#         dev.off()
	}
}

statplot <- function(clusts,out,...){
	plots <- lapply(names(clusts)[2:7],dot.stat,clusts)
	arrange.stats(plots,paste0(out,'.optimization'),...)
}


cutoff <- function(g,genes){
	edges <- get.edgelist(g)
	edges <- cbind(edges,E(g)$weight)
	edges <- edges[order(E(g)$weight,decreasing = T),]

	f.cutoff <- function(genes,i){
		genes <- setdiff(genes,edges[i,1:2])
		if(length(genes)==0) return(edges[1:i,])
		f.cutoff(genes,i+1)
	}

	e <- f.cutoff(genes,1)
	res <- graph_from_edgelist(e[,1:2],F)
	E(res)$weight <- as.numeric(e[,3])
	return(res)
}


f.connected <- function(nodes,res){
	connected <- sapply(nodes,intersect,res)
	res <- connected[[which.max(sapply(connected,length))]]
	nodes <- nodes[setdiff(names(nodes),res)]
	if(length(nodes)==0) return(res)
	f.connected(nodes,res)
}

write.dot <- function(g,filename,...){
	write.graph(g,format='dot',file='tmp.dot')
	out <- mkdate(filename,'dot')
	system2('sed',paste("'s/name/label/' tmp.dot >",out))
	system2('dot',paste('-Tsvg -O',out))
	#         V(g)$label <- names(V(g)) 
	# 
	#         clust <- sapply(names(V(g)),neighbors,graph=g)
	#         clust <- clust[order(sapply(clust,length),decreasing = F)]
	# 
	#         colfn <- col.abs(E(g)$weight)
	#         E(g)$color <- colfn(E(g)$weight)
	#         dir.f(write.graph,'file')(g,format='dot',filename=filename)
}


