source("R/modelfns.R")
source("R/descend.R")
source("R/clustplots.R")
source("R/plotfns.R")

library(leiden)
library(optparse)
library(igraph)
library(dirfns)
library(purrr)
library(umap)


dot.stat <- function(y,dat,...){
	require(ggplot2)
	ggplot(dat,
	       aes_string(x=names(dat)[1],
			  y=y,...))+geom_point()
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

arrange.stats <- function(plots,filename,ncols=3,...){
	require(ggpubr)

	nrows <- ceiling(length(plots)/ncols)
	dat <- lapply(plots,'[[','data')
	#         rho <- sapply(dat,
	#                       function(x) cor(x[,1],
	#                                       x[,2],
	#                                       method='spearman'))

	#         title <- paste('Spearman correlation =',
	#                        as.character(rho))

	plots <- ggarrange(plotlist=plots,
			   labels=names(plots),
			   ncol=3,nrow=nrows)

	dir.f(ggexport)(plots,
			filename=paste0(filename,'.pdf'),
			...)
}

plot.knn <- function(dat,g){
	require(igraph)
	dists <- get.edgelist(g)
	f <- function(d,col) apply(d,1,
				   function(x) lines(dat[x,],
					col=col))

	#         dir.pdf(out)
	plot(NULL, xlim=range(dat[,1]), 
	     ylim=range(dat[,2]),
	     xlab=colnames(dat)[1],
	     ylab=colnames(dat)[2]) 
	f(dists, col=rgb(0,0,0,0.1))
	return(dists)
}

plot.edge <- function(dat,g,clusts=NULL){
	dists <- plot.knn(dat,g)

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

plot.pt <- function(dat,g,clusts=NULL,pch=19,cex=0.8,...){
	require(purrr)
	f <- function(sel,col) points(dat[sel,], col=col,pch=pch,cex=cex,...)
	dists <- plot.knn(dat,g)

	clustn <- unique(clusts)
	clustn <- clustn[order(clustn)]
	clustsel <- lapply(clustn, 
			 purrr::compose(which,
				 partial(`==`,clusts)))
	cols <- rainbow(length(clustsel))
	mapply(f,clustsel,cols)
	legend(
	       'bottomleft',
	       as.character(clustn),
	       col=cols,
	       pch=pch,
	)
}



statplot <- function(clusts,out){
	plots <- lapply(names(clusts)[2:7],dot.stat,clusts)
	arrange.stats(plots,paste0(out,'.optimization'))
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

parser <- OptionParser()
parser <- add_option(parser, '--modeldir', action = 'store',
		     default = 'data/2024-05-03')
parser <- add_option(parser, '--clustdir', action = 'store',
		     default = '2024-05-08')
parser <- add_option(parser, '--groups', action = 'store',
		     default = 'data/groups.csv')
parser <- add_option(parser, '--pheno', action = 'store',
		     default = 'data/phenotype.csv')
opts <- parse_args(parser)


groups <- read.csv(opts$groups)
pheno <- read.csv(opts$pheno,row.names=1)

dat <- as.matrix(read.csv(paste0(opts$modeldir,"/dat.csv"),row.names=1))

model <- read.model(paste0(opts$modeldir,'/encode2.model'),dat)

model <- list() 
encoded <- read.csv("data/2024-05-03/dense/L1_L2/E.csv")
dists <- as.matrix(dist(encoded))
model$dists <- dists
model$encoded <- encoded

model$k <- read.csv(paste0(opts$clustdir,'/k.csv'),row.names=1)
names(model$k)[1:2] <- c('k','ES')


clusts <- lrtab(opts$clustdir,read.csv,'leiden',row.names=1)

# which.k <- which.max(sapply(clusts, 
#                       purrr::compose(max, '['), ,
#                       'stat'))
# k <- as.numeric(sub('leiden.k','',names(clusts)[which.k]))

which.k <- which.max(model$k$ES)
k <- model$k$k[which.k]
#model$umap <- umap(encoded,n_neighbors=k)$layout
model$umap <- umap(encoded)$layout
colnames(model$umap) <- c("UMAP1","UMAP2")

knn <- get.knn(model$dists,k,'plus')
enrichCond(groups$Condition,
	   as.matrix(as_adjacency_matrix(knn)),
	   'knn.network.fr',layout.fruchterman.reingold)
enrichCond(groups$Condition,
	   as.matrix(as_adjacency_matrix(knn)),
	   'knn.network')

hyper <- get.hyper(knn, groups$Condition)
g <- poisGraph(hyper)
networkPois(g,'hyper.k',
	    colfn=colorRamp2(c(0,max(E(g)$weight)),
			     c('white','red')))

model$clusts <- clusts$leiden

names(model$clusts)[1:2] <- c('resolution','combined_score')

model$clusts <- model$clusts[model$clusts$nclust>1,]

# model$stats <- read.csv(paste0(opts$dir,'/stats.csv'),row.names=1)

# model$clusts <- clusts$clusts[,c(1,7:nrow(clusts$encoded)+7)]

mapply(statplot,clusts,sub('leiden.','',names(clusts)))
statplot(model$clusts,'leiden.k17',nrows=3,ncols=3)

dir.plot('knn')(plot.edge,model$umap,knn)

dir.f(ggexport)(dot.col('encoding2',
			as.data.frame(model$encoded),
			col=groups$Condition,'condition'),
		filename='encoding.pdf')

plots <- lapply(names(model$clusts)[2:7],dot.stat,clusts$clusts)

es <- dot.stat('ES',model$k)
dir.f(ggexport)(ggarrange(plotlist=list(es),ncol=3,nrow=3),
		filename="ES.pdf")
 
plots <- append(plots,list(es))
arrange.stats(plots,'optimization')

arrange.stats(lapply(names(model$stats)[-1],dot.stat,clusts$stats),'stats')
 
leidens <- t(model$clusts[,-1:-7])
sel <- sapply(model$clusts[,c(2:6)],which.max)
sel['recall'] <- which(model$clusts[,1]==max(model$clusts[model$clusts[,4]==max(model$clusts[,4]),1]))

leidens <- leidens[,sel]
colnames(leidens) <- names(model$clusts[2:6])
label <- paste('resolution =',model$clusts[sel,1])

ggdat <- cbind(as.data.frame(model$umap),sapply(as.data.frame(leidens),as.character),Depdc=as.character(groups$Condition=="Depdc"))

g <- ggplot(ggdat, aes_string(x='UMAP1', y='UMAP2'))

f <- function(fill) g + 
	geom_point(aes_string(color="Depdc", fill=fill), 
		   shape=21) + 
	scale_color_manual(values=c("white","black"))

plots <- lapply(colnames(leidens),f)

dir.f(arrange.plots,'out')(plots, labs=label,
			   filename='leidens.depdc')

dists <- as.matrix(as_adjacency_matrix(knn))
row.names(dists) <- as.numeric(1:nrow(dists))
colnames(dists) <- as.numeric(1:nrow(dists))

dir.plot('knn.clust')(plot.pt,model$umap,knn,leidens[,1])

clustknn <- function(cond,clusts,...)
	plot
clustcond <- function(cond,clusts=NULL,...){
	dir.pdf(paste0('umap/',gsub("/","_",cond)))
	plot.edge(model$umap,knn,clusts)
	points(model$umap[groups$Condition==cond,],...)
	dev.off()
}
sapply(unique(groups$Condition),clustcond,clusts=leidens[,1])

knncond <- function(cond,clusts=NULL,...){
	dir.pdf(paste0('umap/',gsub("/","_",cond)))
	plot.knn(model$umap,knn)
#	points(model$umap,pch=1,col=1,cex=0.8)
	points(model$umap[groups$Condition==cond,],pch=19,cex=0.8,col=1)
	dev.off()
}
sapply(unique(groups$Condition),knncond)


dir.pdf('clust.depdc')
plot.edge(model$umap,knn,leidens[,1])
points(model$umap[groups$Condition=="Depdc",])
dev.off()

clustplots(model$encoded,
	   leidens[,1],
	   groups$Phenotype,
	   dists,'pheno')

clustplots(model$encoded,
	   leidens[,1],
	   NULL,
	   dists,legend.ncol=1,legend.cex=1)

pheno.sel <- sapply(compose(unique, unlist, 
			strsplit)(groups$Comment.detailed,
			split=' / '),
		grep,
		groups$Comment.detailed)

pheno <- replicate(5,rep('WT',nrow(groups)))
colnames(pheno) <- c("TVC.division","TVC.migration",
		     "ATM.division","ATM.migration",
		     "other")
row.names(pheno) <- row.names(groups)
pheno[pheno.sel$`inhibited TVC division`,
      "TVC.division"] <- "inhibited"
pheno[pheno.sel$`enhanced TVC division`,
      "TVC.division"] <- "enhanced"
pheno[pheno.sel$`inhibited ATM division`,
      "ATM.division"] <- "inhibited"
pheno[pheno.sel$`enhanced ATM division`,
      "ATM.division"] <- "enhanced"
pheno[pheno.sel$`inhibited TVC migration`,
      "TVC.migration"] <- "inhibited"
pheno[pheno.sel$`enhanced TVC migration`,
      "TVC.migration"] <- "enhanced"
pheno[pheno.sel$`inhibited ATM migration`,
      "ATM.migration"] <- "inhibited"
pheno[pheno.sel$`enhanced ATM migration`,
      "ATM.migration"] <- "enhanced"
pheno[pheno.sel$`problem disposition TVC`,
      "other"] <- "TVC disposition"
pheno[pheno.sel$`TVC cell alignment`,
      "other"] <- "TVC alignment"
# pheno[pheno.sel$`problem disposition ATM`,
#       "other"] <- "ATM disposition"
# pheno <- cbind(Condition=groups$Condition,pheno)

dir.f(clusthyper,'out')(groups[,"Condition",drop=F],leidens[,1])
dir.f(clusthyper,'out')(as.data.frame(pheno),leidens[,1],
			filename='pheno')

res <- model$clusts[which.max(model$clusts[,2]),1]
g <- gene.network(knn,res,groups$Condition,mode='directed')

edgelist <- cbind(as.data.frame(get.edgelist(g)),E(g)$weight)
edgelist <- do.call(rbind,lapply(split(edgelist,edgelist[,1]),function(x) x[order(x[,3],decreasing=T)[1:min(nrow(x),5)],]))
reduced <- graph_from_edgelist(as.matrix(edgelist[,-3]),F)
write.dot(reduced,'gene_network')
graph.pdf('gene_network',reduced)

sel <- E(g)$weight>quantile(E(g)$weight,0.9)
E(reduced)$weight <- E(g)$weight[sel]

dir.f(networkPois,'out')(reduced,'gene_network',colfn=col.abs(E(reduced)$weight),title='modularity')

dir.f(write.graph,'file')(reduced,format='dot',filename='gene.network.dot')
V(g)$cluster <- leiden(g,resolution_parameter=1.05)
	
reduced <- cutoff(g,unique(groups$Condition))
write.dot(g,'gene.network.dot')

