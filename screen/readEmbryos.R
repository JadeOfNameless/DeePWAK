source("imageFns.R")
library(dirfns)
library(purrr)

dirs <- list.files(
  "segdat",
  pattern = "_Statistics$",
  recursive = T,
  include.dirs = T,
  full.names = T
)

#sgrna <- read.csv("sgRNA.csv")

meta <- strsplit(dirs,'/')

meta <- data.frame(
  condition=sub('_','.',sub(' .*','',sapply(meta,'[',3))),
  grouping=dirs,
  batch=sapply(meta,'[',2),
  slide='',
  date=sub('.*([0-9]{2})([0-9]{2})([0-9]{4})','\\3_\\1_\\2',sapply(meta,'[',2)),
  #date=sub('.*([0-9]{2})([0-9]{2})([0-9]{4})_\\[.*','\\3_\\1_\\2',sapply(meta,'[',4)),
  image=paste0('image',sub('.*Image_([0-9]+)\\].*','\\1',dirs)),
  half='',
  file=dirs,
  stringsAsFactors = F
)
sel <- grep('slide',dirs)
meta$slide[sel] <- sub('.* (slide)\\s?([0-9]+).*','\\1\\2',dirs[sel])

meta[grep('next part',dirs),'slide'] <- 'nextpart'
meta[grep('right',dirs,ignore.case=T),'half'] <- "right"
meta[grep('left',dirs,ignore.case=T),'half'] <- "left"

sel <- grep('normal',meta$batch)
meta[sel,'batch'] <- 'old'
meta[-sel,'batch'] <- 'new'

meta$cond <- sub('^[0-9]+[A-Z]\\-','',meta$condition)
meta[meta$cond=="EPH",'cond'] <- "Eph"
meta[meta$cond=="GNA12.13",'cond'] <- "Gna12.13"
meta[meta$cond=="GNA.L.S",'cond'] <- "Gna.L.S"

#tmp <- sgrna[!duplicated(sgrna[,c(1,3)]),c(1,3)]
#sapply(unique(meta$cond),grep,sgrna$Gene.Name,ignore.case=T)

meta$id <- sub('^_','',sub('_$','',paste(meta$slide,meta$image,meta$half,sep='_')))

meta[grep('grouped',meta[,2],ignore.case = T),2] <- 'surface'
meta[grep('surf[az]ce',meta[,2],ignore.case = T),2] <- 'surface'
meta[grep('individual',meta[,2],ignore.case = T),2] <- 'cell'
meta[grep('nuclei',meta[,2],ignore.case = T),2] <- 'cell'
meta[grep('cell',meta[,2],ignore.case = T),2] <- 'cell'


meta$name <- make.names(paste(
			      meta$cond,meta$id,
			      meta$date,
			      sep='.'))

embryos <- split(meta[,c('name','file')],meta$grouping)
embryos <- lapply(embryos,function(x) data.frame(name=make.unique(x$name),file=x$file))

#hack to fix names
sel <- setdiff(embryos$cell$name,embryos$surface$name)
embryos$cell$name[embryos$cell$name%in%sel] <- setdiff(embryos$surface$name,embryos$cell$name)

embryos <- do.call(merge,append(unname(embryos),list(by='name')))
names(embryos) <- c('name','cell','surface')

celldat <- lapply(embryos$cell,readEmbryos)
surfacedat <- lapply(embryos$surface,readEmbryos)

#dat <- lapply(dirs,readEmbryos)
#
#sel <- meta$grouping=='surface'
#surface <- sel&!duplicated(meta$name[sel])
#cell <- !sel&!duplicated(meta$name[!sel])
#meta.surface <- meta[surface,]
#meta.cell <- meta[cell,]
#celldat <- dat[cell]
#surfacedat <- dat[surface]
#

celldat <- lapply(celldat,normvol)
ndists <- sapply(celldat,getDists)
cdists <- sapply(celldat,getDists,F)
stats <- sapply(celldat,groupStats)
dists <- rbind(ndists,cdists,stats)

cellID <- sapply(celldat,function(x) factor(x[,2],levels=c("ATM","TVC")))
ncell <- sapply(cellID,table)
row.names(ncell) <- c('nATM','nTVC')
cell <- rbind(dists,ncell)
colnames(cell) <- embryos$name

membrane <- sapply(surfacedat,membraneStats)
colnames(membrane) <- embryos$name

params <- merge(t(cell),t(membrane),0)
params[sapply(params,function(x) !is.finite(x)&is.numeric(x))] <- 0
dir.csv(params, 'params', 'out', row.names=F, append.date=F)

z <- zdf(params[,-1])
z <- cbind(Row.names=params[,1],as.data.frame(z))
dir.csv(z, 'z_dat', 'out', append.date=F, row.names=F)

phenotype <- read.csv('phenotype.csv',stringsAsFactors=F)
names(phenotype)[1] <- 'Date'
phenotype$Date <- gsub('-','_',phenotype$Date)
# phenotype$Condition <- gsub('\\s','',phenotype$Condition)
# 
# phenotype$Cond <- sub('^[0-9]+[A-Z][-\\/]','',phenotype$Cond)
# phenotype$Cond <- sub("Gna12\\/[0-9]+","Gna12.13",phenotype$Cond)
# phenotype$Cond <- sub("EPB4\\.[0-9]+","EPB4.1",phenotype$Cond)
# 
# phenotype$Cond <- gsub('_','.',phenotype$Cond)

phenotype$Condition <- sub('.*\\s','',phenotype$Condition)
phenotype$Condition[phenotype$Condition=="EPH"] <- "Eph"
phenotype$Condition[phenotype$Condition=="GNA-L/S"] <- "Gna-L/S"
# phenotype$Condition[phenotype$Condition=="GNA12.13"] <- "Gna12.13"
phenotype$Condition[phenotype$Condition=="12/13"] <- "Gna12/13"
phenotype$ID.embryo <- tolower(phenotype$ID.embryo)
phenotype$ID.embryo <- sub("^[0-9-]+_",'',phenotype$ID.embryo)
#phenotype <- phenotype[-grep("with_lightning",phenotype$ID.embryo)]
phenotype$ID.embryo <- sub("with_lightning_","",phenotype$ID.embryo)
phenotype$Threshold.membrane <- as.numeric(phenotype$Threshold.membrane)

phenotype$Name <- make.names(paste(
	phenotype$Condition,
	phenotype$ID.embryo,
	phenotype$Date,
	sep='.'
))
# names(phenotype)[9] <- 'Phenotype'
phenotype$Phenotype <- ''
phenotype$migration.perturbed <- grepl('migration',phenotype$Comment.on.the.embryo)
phenotype$division.perturbed <- grepl('division',phenotype$Comment.on.the.embryo)
phenotype$orientation.perturbed <- grepl('orientation',phenotype$Comment.on.the.embryo)

phenotype$Phenotype <- sapply(
	phenotype$migration.perturbed,
	function(x) if(x) "migration" else ''
)
phenotype$Phenotype <- paste(phenotype$Phenotype,sapply(
	phenotype$division.perturbed,
	function(x) if(x) "division" else ''
),sep='/')
phenotype$Phenotype <- paste(phenotype$Phenotype,sapply(
	phenotype$orientation.perturbed,
	function(x) if(x) "orientation" else ''
),sep='/')
phenotype$Phenotype <- sub('^\\/+','',phenotype$Phenotype)
phenotype$Phenotype <- sub('\\/+$','',phenotype$Phenotype)
phenotype$Phenotype <- sub('\\/+','\\/',phenotype$Phenotype)
phenotype$Phenotype[phenotype$Phenotype!=''] <- paste(
	phenotype$Phenotype[phenotype$Phenotype!=''],
	'perturbed'
)
phenotype$Phenotype[phenotype$Phenotype==''] <- "WT"
row.names(phenotype) <- make.unique(phenotype$Name)

#phenotype <- phenotype[!duplicated(phenotype$Name),]

tmp <- setdiff(params$Row.names,row.names(phenotype))
tmp2 <- setdiff(row.names(phenotype),params$Row.names)

sel <- sapply(sub('\\.[0-9]{4}_{0-9}{2}_{0-9}{2}.*','',tmp),grep,row.names(phenotype),ignore.case=T)

row.names(phenotype)[sel] <- tmp

dat.phenotype <- merge(params,phenotype,by.x='Row.names',by.y=0,all.x=T)
# dat.phenotype$Condition <- sub('\\..*','',dat.phenotype[,1])
dat.phenotype$Date <- sub('.*\\.','',dat.phenotype[,1])

dir.csv(dat.phenotype,'dat','out',row.names=F,append.date=F)

groups <- dat.phenotype[,c(1,(length(params)+1):length(dat.phenotype))]
			   #         'Row.names', 'Condition', 'Phenotype',# 'nTVC', 'nATM',
			   #         'migration.perturbed', 'orientation.perturbed', 'division.perturbed'
	#         'TVC.contiguous', 'ATM.contiguous'
# )]

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
row.names(pheno) <- groups$Row.names
dir.csv(pheno,'phenotype', 'out', append.date=F)


pheno.sub <- groups[
	dat.phenotype$nTVC==4&dat.phenotype$nATM==2&dat.phenotype$TVC.contiguous&dat.phenotype$ATM.contiguous,
]

sapply(phenotype[,c("Threshold.nuclei","Threshold.membrane")],as.numeric)

dir.csv(groups,'groups', 'out', row.names=F, append.date=F)
dir.csv(pheno.sub,'phenoSub','out',row.names=F,append.date=F)
dir.csv(params[params[,1]%in%pheno.sub[,1],],'phenoSubParam','out',row.names=F,append.date=F)
dir.csv(z[z[,1]%in%pheno.sub[,1],],'phenoSubZ','out',row.names=F,append.date=F)

library(dirfns)
library(moreComplexHeatmap)
corHeatmap(params[,-1], buffer.h=50, buffer.w=50)
quantHeatmap(params[,c(-1,-106,-107,-116,-117)], 'params', cell.w=0.012, cell.h=0.005, conds=groups[,-1])

quantHeatmap(z[,c(-1,-106,-107,-116,-117)], 'z', cell.w=0.012, cell.h=0.005, conds=groups[,-1])

# source('dewakss.R')
#res <- seq(0.25, 2, 0.25)
# 
# runDewakss()
# runDewakss(params='out/z_dat.csv', out='out/z')
# runDewakss(params='out/phenoSubParam.csv', out='out/phenoSub', groups='out/phenoSub.csv')
# runDewakss(params='out/phenoSubZ.csv', out='out/phenoSubZ', groups='out/phenoSub.csv')
# 
#sapply(res, function(x) runDewakss(res=x))
#sapply(res, function(x) runDewakss(res=x, params='out/z_dat.csv', out='out/z'))
#sapply(res, function(x) runDewakss(res=x, params='out/phenoSubParam.csv', out='out/phenoSub', groups='out/phenoSub.csv'))
#sapply(res, function(x) runDewakss(res=x, params='out/phenoSubZ.csv', out='out/phenoSubZ', groups='out/phenoSub.csv'))

library(rgl)

cellaxis <- function(cell,c,var) do.call(rbind,
                lapply(c("A","B","C"),
                    function(axis) as.matrix(setNames(cell[,c(paste0(c,".Position.",var),
                                        paste0(c,".Ellipsoid.Axis.",axis,".",var))],1:2))))

drawcell <- function(cell){
    X <- cellaxis(cell,"Cell","X")
    Y <- cellaxis(cell,"Cell","Y")
    Z <- cellaxis(cell,"Cell","Z")
    segments3d(X,Y,Z)
}
drawcell(celldat[[1]])

# Function to generate points for an ellipsoid
ellipsoid_points <- function(center=c(0,0,0), radii=c(1,1,1), resolution = 50) {
  theta <- seq(0, pi, length.out = resolution)  
  phi <- seq(0, 2 * pi, length.out = resolution)
  Theta <- matrix(rep(theta, each = resolution), ncol = resolution)
  Phi <- matrix(rep(phi, times = resolution), ncol = resolution)
  
  x <- center[1] + radii[1] * sin(Theta) * cos(Phi)
  y <- center[2] + radii[2] * sin(Theta) * sin(Phi)
  z <- center[3] + radii[3] * cos(Theta)
  
  list(x=x, y=y, z=z)
}

# Generate ellipsoid points
ep <- ellipsoid_points(center=c(1,2,3), radii=c(2,3,1))

# Plot the ellipsoid
surface3d(ep$x, ep$y, ep$z, color="red", alpha=1)

