library(keras)
library(dirfns)

source('modelfns.R')
source('clustplots.R')
source('descend.R')

plot.leidens('2022-10-03/encode2')
plot.leidens('2022-10-03/encode3')
plot.leidens('2022-10-03/encode7')
plot.leidens('2022-10-03/encode14')

plot.leidens('2022-10-03/encode2','encoded')
plot.leidens('2022-10-03/encode3' ,'encoded')
plot.leidens('2022-10-03/encode7' ,'encoded')
plot.leidens('2022-10-03/encode14','encoded')

groups <- read.csv('out/groups.csv')
cols <- groups[,c("Condition","Phenotype")]
dat <- read.csv('out/z_dat.csv',row.names=1)
dat <- dat[,apply(abs(dat),2,max)!=0]
x_train <- t(t(dat)/apply(abs(dat),2,max))

corHeatmap(x_train,method='spearman')

enrichK <- function(cond,dists,k,out){
	out <- paste(out,
		     paste0('k',as.character(k)), 
		     sep='/')
	dists <- get.knn(dists,k)
	enrichCond(cond,dists,out)
}


hyper <- function(cond,clusts,res,k,out){
	out <- paste(out,res,k,sep='/')
	clusthyper(cond,clusts[[res]][,k],out)
}

gethyper <- function(cond,clusts,out){
	sapply(names(clusts),
	       function(res) sapply(names(clusts[[res]]),
				    function(k) hyper(cond,
						      clusts,
						      res,k,
						      out)))
}

getplots <- function(group,clustdir){
	clusts <- read.clusts(clustdir)
	sapply(3:21,
	       function(k) {
		       enrichK(groups$Condition,
			       clusts$dists,k,
			       paste0(clustdir,'/network'))
	       })
	gethyper(group,clusts$clusts,clustdir)
}

tmp <-clusts
tmp$clusts <- setNames(do.call(function(...) mapply(cbind,...,SIMPLIFY=F),clusts$clusts),colnames(tmp$clusts[[1]]))

tmp2 <- mapply(gsea.edges,3:21,tmp$clusts,MoreArgs = list(tmp$dists,groups$Condition),SIMPLIFY = F)

clusts2 <- read.clusts('2022-10-03/encode2')
int <- read.csv('interactions.csv',row.names=1)[,3:4]
int <- rbind(as.matrix(int),t(sapply(unique(groups$Condition),rep,2)))
int <- apply(int,1,paste,collapse='->')

tmpk=descend(23,c(3,53),gsea.prot,clusts$dists,groups$Condition,int,breaks=5,rate=.95,maxiter=100,discrete=T,min=3)

res.sil2 <- get.res(1,c(0.01,2),10,.95,10,1,tmpk$argmax,clusts$encoded,int,groups$Condition)

res.sil2 <- get.res(1,c(0.01,5),10,.95,10,1,tmpk$argmax,clusts$encoded,int,groups$Condition)

res <- get.res(c(0.01,2),10,.99,100,1,tmpk$argmax,clusts$encoded,int,groups$Condition)

res2 <- get.res(c(0.01,4),10,.95,10,1,tmpk$argmax,clusts$encoded,int,groups$Condition)

getplots(cols,'2022-10-03/encode2')
getplots(cols,'2022-10-03/encode3')
getplots(cols,'2022-10-03/encode7')
getplots(cols,'2022-10-03/encode14')

sapply(3:21,function(k) enrichK(groups$Condition,dists,k,'2022-09-06/encode2/condnetwork'))
test <- clusts2[grep('leiden',names(clusts2))]
gethyper(cols,test,'2022-09-06/encode2')

clusthm <- function(input,group,clustdir){
	require(moreComplexHeatmap)
	clusts <- lrtab(clustdir,pattern='\\.txt$')

	clustleiden <- names(clusts)[grep('leiden',
					  names(clusts))]
	dat <- cbind(input,clusts$encoded)
	colsp <- c(rep('input',ncol(input)),
		   rep('encoded',ncol(clusts$encoded)))

	hmfn <- function(file,path=clusts,...){
		quantHeatmap(dat,file, 
			     cell.w=0.005,cell.h=0.005, 
			     show_row_names=F, 
			     path=path,column_split=colsp,...)
	}

	hmfn('input',conds=group[,'Phenotype',drop=F], 
	     split=group$Condition)

	leidenplots <- function(res){
		leidenplot <- function(k){ 
			hmfn(k,paste0(clusts,'/',res),
			     conds=group,
			     split=clusts[[res]][,k])
		}
	}
}

clusthm(x_train,cols,'2022-09-06/encode2')

clusts2 <- lrtab('2022-09-06/encode2')

clusts <- clusts2[2:11]

clusts <- apply(clusts,2,as.character)
colnames(clusts) <- paste0('k',as.character(seq(3,21,2)))


plot.leiden <- function(dat,clust,err,out,title=''){
	cols <- as.data.frame(apply(clust,2,as.character))
	names(cols) <- colnames(clust)
	lab <- paste('k =',sub('k','',names(cols)),' -log2(err) =',as.character(round(err,2)))
	plot.clust(dat,cols,out,labs=lab)
}
	
plot.leiden(clusts2$encoded,clusts2$leiden0.1,clusts2$log2error$leiden0.1,'2022-09-06/encode2/leiden0.1',paste('res =',sub('leiden','','leiden0.1')))




