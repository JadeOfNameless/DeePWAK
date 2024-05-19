source('R/modelfns.R')
source('R/descend.R')
source('R/clustplots.R')

library(optparse)
library(igraph)
library(dirfns)

leiden_reps = 1000

groups <- read.csv('data/groups.csv')
encoded <- read.csv("data/2024-05-03/dense/L1_L2/E.csv")
#dists <- read.csv("data/2024-05-03/sparse/eucl/linear/L2/D.csv")
dists <- as.matrix(dist(encoded))

int <- read.csv('data/interactions.csv',row.names=1)[,3:4]
g <- graph_from_edgelist(as.matrix(int))
write_graph(g,format='dot',file='tmp.dot')
out <- mkdate('interactions','dot')

int <- rbind(int,setNames(int[,2:1],names(int)))
int <- rbind(as.matrix(int),t(sapply(unique(groups$Condition),rep,2)))
int <- int[!duplicated(int),]

int <- apply(int,1,paste,collapse='->')

# k <- get.k(3:53,m$dists,groups$Condition,int)
k <- get.k(3:53,dists,groups$Condition,int,'directed')
# k2 <- get.k(3:53,clusts$dists,groups$Condition,int,'directed')
dir.csv(k,'k')
k.max <- k[which.max(k[,'ES']),'k']

leidens <- res.unif(c(0.01,3),31,encoded,
		    int,groups$Condition,
		    leiden_reps)
dir.csv(leidens,'leiden_31')

leidens <- res.unif(c(0.01,3),k.max,encoded,
		    int,groups$Condition,
		    leiden_reps)
dir.csv(leidens,'leiden')
