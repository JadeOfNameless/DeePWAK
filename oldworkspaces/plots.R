source("R/modelfns.R")
source("R/descend.R")
source("R/clustplots.R")
source("R/plotfns.R")

library(dirfns)
library(purrr)
library(igraph)
library(moreComplexHeatmap)
library(ggpubr)

hmMSE <- function(out,...){
	dir.pdf(out)
	draw(Heatmap(...,name="MSE",cluster_columns=F,cluster_rows=F,
		     row_title="k",column_title="d"))
	dev.off()
}

groups <- read.csv("data/groups.csv")
pheno <- read.csv("data/phenotype.csv",row.names=1)
intx <- read.csv('data/interactions.csv',row.names=1)[,3:4]
intx <- apply(intx,1,paste,collapse='->')

E <- t(read.csv("data/E.csv"))
colnames(E) <- c('embedding1','embedding2')
E <- as.data.frame(E)
D <- as.matrix(dist(E))

G <- read.csv("data/G.csv")
G <- as.matrix(G)
G[is.na(G)] <- 0
row.names(G) <- NULL
colnames(G) <- NULL
M <- t(read.csv("data/M.csv"))
P <- t(read.csv("data/P.csv"))

dkMSE <- read.csv("data/MSEdk.csv")
dkMSE <-as.matrix(dkMSE) 
row.names(dkMSE) <- as.character(1:nrow(dkMSE))
colnames(dkMSE) <- as.character(1:ncol(dkMSE))

bfC <- read.csv("data/clusters.csv")
bfgamma <- read.csv("data/MSEgamma.csv")
gammasMSE <- as.matrix(read.csv("data/MSEgamma_s.csv"))

sel <- bfgamma$nclust>1
bfgamma <- bfgamma[sel,]
bfC <- bfC[,sel]
gammasMSE <- gammasMSE[,sel]

bfd <- ncol(dkMSE)
bfk <- nrow(dkMSE)
ij_dk <- which.min(dkMSE)
k <- ij_dk %% bfk
d <- ceiling(ij_dk / bfk)

bfs <- nrow(gammasMSE)
ij_gammas <- which.min(gammasMSE)
s <- ij_gammas %% bfs
i_gamma <- ceiling(ij_gammas / bfs)
gamma <- bfgamma[i_gamma,1]
C <- bfC[,i_gamma]

knn <- graph_from_adjacency_matrix(as.matrix(G),'directed',T,F)
sil <- sapply(bfC,function(C) mean(silhouette(C,D)[,3]))
ES <- sapply(bfC,partial(test.leiden.gsea,knn),groups$Condition,intx)

G.gene <- sapply(bfgamma$gamma,partial(gene.network,knn),groups$Condition)
recall <- sapply(G.gene,score.gene.network,intx)

dir.pdf('knn')
plot.edge(E,knn,0.005)
dev.off()

dir.pdf("knnC")
plot.edge(E,knn,clusts=C)
dev.off()

p.cond <- ggplot(cbind(E,Condition=groups$Condition))+geom_point()+aes(col=Condition)
dir.f(ggexport)(dot.col("embedding2",
			as.data.frame(E),
			col=groups$Condition,'condition'),
		filename='encoding.pdf')
dir.f(ggexport)(dot.col("embedding2",
			as.data.frame(E),
			col=as.character(C),'cluster'),
		filename='clusts.pdf')

hmMSE("dkMSE",dkMSE,show_row_names=F)

p.d <- dot.stat("MSE",data.frame(d=1:bfd,MSE=dkMSE[k,]))
p.k <- dot.stat("MSE",data.frame(k=1:bfk,MSE=dkMSE[,d]))
p.gamma <- dot.stat("MSE",data.frame(gamma=bfgamma[,1],MSE=gammasMSE[s,]))
p.s <- dot.stat("MSE",data.frame(s=1:bfs,MSE=gammasMSE[,i_gamma]))
p.nclust <- dot.stat("nclust",bfgamma[,1:2])


plots=list(p.d,p.k,p.gamma,p.s,p.nclust)
lab = c(sprintf("k=%g",k),sprintf("d=%g",d),sprintf("d=%g,k=%g,s=%g",d,k,s),
	sprintf("d=%g,k=%g,gamma=%0.2f",d,k,gamma),
	sprintf("d=%g,k=%g,s=%g",d,k,s))
plots <- ggarrange(plotlist=plots,labels=lab,ncol=3,nrow=3,label.x=0)


dir.f(ggexport)(plots,filename='hyperparams.pdf')

plots=list(p.d,p.k,p.gamma,p.s,p.nclust)
lab = c(sprintf("k=%g",k),sprintf("d=%g",d),sprintf("d=%g,k=%g,s=%g",d,k,s),
	sprintf("d=%g,k=%g,γ=%0.2f",d,k,gamma),
	sprintf("d=%g,k=%g,s=%g",d,k,s))
plots <- ggarrange(plotlist=plots,labels=lab,ncol=3,nrow=3)
dir.f(ggexport)(plots,filename='hyperparams.pdf')

dir.f(clusthyper,'out')(groups[,"Condition",drop=F],C)
dir.f(clusthyper,'out')(as.data.frame(pheno),C,
			filename='pheno')

p.sil <- dot.stat("mean_silhouette",
		  data.frame(gamma=bfgamma$gamma,mean_silhouette=sil))
p.ES <- dot.stat("ES",data.frame(gamma=bfgamma$gamma,ES=ES))
p.recall <- dot.stat("recall",data.frame(gamma=bfgamma$gamma,recall=recall))

plots <- list(p.sil,p.ES,p.recall)
plots <- ggarrange(plotlist=plots,ncol=3,nrow=3,label.x=0)
dir.f(ggexport)(plots,filename='validation.pdf')

dir.f(ggexport)(p.gamma,filename='gammaMSE.pdf')

model <- read.model(paste0(opts$modeldir,'/encode2.model'),dat)

model$k <- read.csv(paste0(opts$clustdir,'/k.csv'),row.names=1)
names(model$k)[1:2] <- c('k','ES')


clusts <- lrtab(opts$clustdir,read.csv,'leiden',row.names=1)

# which.k <- which.max(sapply(clusts, 
#                       purrr::compose(max, '['), ,
#                       'stat'))
# k <- as.numeric(sub('leiden.k','',names(clusts)[which.k]))

which.k <- which.max(model$k$ES)
k <- model$k$k[which.k]

knn <- get.knn(model$dists,k,'plus')
enrichCond(groups$Condition,
	   G,
	   'knn.network.fr',layout.fruchterman.reingold)
enrichCond(groups$Condition,
	   G,
	   'knn.network')

hyper <- get.hyper(knn, groups$Condition)
g <- poisGraph(hyper)
networkPois(g,'hyper.k',
	    colfn=colorRamp2(c(0,max(E(g)$weight)),
			     c('white','red')))

model$clusts <- clusts[[which.k]]

names(model$clusts)[1:2] <- c('resolution','combined_score')

model$clusts <- model$clusts[model$clusts$nclust>1,]

# model$stats <- read.csv(paste0(opts$dir,'/stats.csv'),row.names=1)

# model$clusts <- clusts$clusts[,c(1,7:nrow(clusts$encoded)+7)]

mapply(statplot,clusts,sub('leiden.','',names(clusts)))
statplot(model$clusts,'leiden.k17',nrows=3,ncols=3)

dir.plot('knn')(plot.edge,model$encoded,knn)
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

ggdat <- cbind(as.data.frame(model$encoded),sapply(as.data.frame(leidens),as.character),Depdc=as.character(groups$Condition=="Depdc"))

g <- ggplot(ggdat, aes_string(x='encoding1', y='encoding2'))

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

dir.plot('knn.clust')(plot.edge,model$encoded,knn,leidens[,1])


dir.pdf('clust.depdc')
plot.edge(model$encoded,knn,leidens[,1])
points(model$encoded[groups$Condition=="Depdc",])
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
