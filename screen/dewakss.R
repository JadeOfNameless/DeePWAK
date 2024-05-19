source("R/clustplots.R")

library(purrr)
library(optparse)
library(umap)
library(ggplot2)
library(moreComplexHeatmap)
library(leiden)

parser <- OptionParser()
parser <- add_option(parser, '--dir', action = 'store',
		     default='data')
parser <- add_option(parser, '--embeddingdir', action = 'store',
		     default = 'data/embedding')
parser <- add_option(parser, '--clustdir', action = 'store',
		     default = 'data/clust')
parser <- add_option(parser, '--clusts', action = 'store',
		     default = 'data/clusts.csv')
parser <- add_option(parser, '--groups', action = 'store',
		     default = 'data/groups.csv')
parser <- add_option(parser, '--pheno', action = 'store',
		     default = 'data/phenotype.csv')
opts <- parse_args(parser)


groups <- read.csv(opts$groups)
pheno <- read.csv(opts$pheno,row.names=1)
clusts <- read.csv(opts$clusts)
names(clusts) <- sub("Column","Head",names(clusts))

E <- lrtab(opts$embeddingdir,read.csv)
C <- lrtab(opts$clustdir,read.csv)

E_c  <- read.csv(paste0(opts$dir,'/E_consensus.csv'))
C_c  <- read.csv(paste0(opts$dir,'/C_consensus.csv'))
logits_c <- log(C_c/(1-C_c))
clusts_c  <- read.csv(paste0(opts$dir,'/clusts_consensus.csv'))

G_10 <- read.csv(paste0(opts$dir,'/10NN.csv'))
Ehat_10 <- read.csv(paste0(opts$dir,'/Ehat_10.csv'))
D <- dist(E_c)

C_10 <- leiden(G_10)
