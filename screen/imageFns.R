readEmbryo <- function(x,y) {
  sel <- 1:(grep("Unit",names(y))-1)
  if("Channel"%in%names(y)){
    names(y)[sel] <- paste0(names(y)[sel],'Channel',as.numeric(y$Channel[1]))
  }
  if("CellID"%in%names(y)){
    y$ID <- y$CellID
    y$CellID <- NULL
  }
  #   y <- y[,c(sel,ncol(y)-c(2,1))]
  y <- y[,c('ID',names(y)[sel])]
  #   if(!"Default.Labels"%in%names(y)) y$Default.Labels <- NA
  merge(x,y,by=c("ID"),all=T)
}

readEmbryos <- function(i) {
	print(i)
	j <- lapply(
	    Filter(function(x) !grepl("Overall",x),list.files(i,'\\.csv',full.names = T)),
	    read.csv,
	    stringsAsFactors=F,
	    skip=3
	)
	init <- data.frame(ID=j[[1]]$ID,Default.Labels=j[[1]]$Default.Labels,
			   stringsAsFactors=F)
	if("CellID"%in%names(j[[1]])){
	  init$ID <- j[[1]]$CellID
	}
	Reduce(
	  readEmbryo,
	  j,
	  init = init
	)
}

abs.max <- function(x) x[which.max(abs(x))]

normvol <- function(x){
  if(all(x$Default.Labels=="ATM")) x$Default.Labels <- "TVC"
  x[,"Nucleus.Volume.frac"] <- x[,'Nucleus.Volume']/sum(x[,'Nucleus.Volume'])
  x[,"Cell.Volume.frac"] <- x[,'Cell.Volume']/sum(x[,'Cell.Volume'])
  x
}

# labelCells <- function(x) {
#         x$Default.Labels <- gsub('\\s','',x$Default.Labels)
#         x$Default.Labels <- sub("group",'',x$Default.Labels)
#         x$Default.Labels <- sub("Anterior",'',x$Default.Labels)
#         x$Default.Labels <- sub("Posterior",'',x$Default.Labels)
#         x$Default.Labels <- sub("2",'',x$Default.Labels)
#         x[is.na(x$Default.Labels),"Default.Labels"] <- "TVC"
#         x[x$Default.Labels%in%c("sTVC","FHP",''),"Default.Labels"] <- "TVC"
#         if(all(x$Default.Labels=="ATM")) x$Default.Labels <- "TVC"
#         x[x$Default.Labels%in%c("ALL","Groupcell"),"Default.Labels"] <- "TVC"
#         x
# }

embryoCol <- function(x){
  require(reshape2)
  x <- x[,c(
    # "Cell.Cytoplasm.Volume","Cell.Volume","Cell.Nucleus.to.Cytoplasm.Volume.Ratio","Cell.Number.Of.Nuclei",
    "Cell.Ellipticity..oblate.","Cell.Ellipticity..prolate.","Cell.Sphericity",
    # "Nucleus.Distance.from.Position.to.Cell.Membrane","Nucleus.Volume",
    "Nucleus.Ellipticity..oblate.","Nucleus.Ellipticity..prolate.","Nucleus.Sphericity",
    "Cell.VolumeLog2FC","Nucleus.VolumeLog2FC"
  )]
  x$ID <- row.names(x)
  x <- melt(x)
  res <- x$value
  row.names(x) <- mapply(paste,x[,1],x[,2],MoreArgs = list(sep="_"))
  return(x[,3,drop=F])
}

coord <- function(x,prefix="Nucleus",
		  suffix='Center.of.Homogeneous.Mass'){
	if(prefix!='') {
		ix <- paste(prefix,suffix,sep='.')
	} else ix <- suffix
	if(!any(grepl(ix,names(x)))) ix <- paste0(prefix,'.Position')
	ix <- paste0(ix,c('.X','.Y','.Z'))
	return(x[,ix])
}
ncoord <- function(x) coord(x,"Nucleus")
ccoord <- function(x) coord(x,"Cell")
# ncoord <- function(x) x[,c(
#   "Nucleus.Center.of.Homogeneous.Mass.X", 
#   "Nucleus.Center.of.Homogeneous.Mass.Y", 
#   "Nucleus.Center.of.Homogeneous.Mass.Z"
# )]
# 
# ccoord <- function(x) x[,c(
#   "Cell.Center.of.Homogeneous.Mass.X", 
#   "Cell.Center.of.Homogeneous.Mass.Y", 
#   "Cell.Center.of.Homogeneous.Mass.Z"
# )]

getDists <- function(x,nucleus=T,normdist=T){
	if(all(x[,2]=="TVC")){
		x <- rbind(x,NA)
		x[nrow(x),2] <- "ATM"
	}
  if(nucleus){
    ncoords <- ncoord(x)
    suffix <- "_Nucleus"
  }else {
    ncoords <- ccoord(x)
    suffix <- "_Cell"
  }
  ndists <- dist(ncoords)
  maxdist <- max(ndists,na.rm=T)

  coords <- split(ncoords,x[,2])

  TVC.ATM <- NA
  extraATM <- 0
  atmCosine <- NA
  if(is.null(coords$ATM)) coords <- append(
	coords,list(ATM=data.frame()),0
  )

  if(nrow(coords$ATM)>0){
	  TVC.ATM <- do.call(data.frame,lapply(1:nrow(coords$ATM),function(y){
	    sapply(1:nrow(coords$TVC),function(z){
	      dist(rbind(coords$ATM[y,],coords$TVC[z,]))
	    })
	  }))
	  
	  if(ncol(TVC.ATM)>2){
		  extraATM <- 1
		  sel <- which.min(apply(TVC.ATM,2,mean))
		  coords$ATM <- coords$ATM[-sel]
	  }
  }

  interdist <- append(lapply(coords,dist),list(TVC.ATM=as.matrix(TVC.ATM)))
  
  if(nrow(coords$ATM)>1){
    comb <- t(combn(labels(interdist$ATM),2))
    comb <- apply(comb,1,function(y) do.call(data.frame,append(lapply(
	labels(interdist$TVC),
	function(z) c(y[1],z,y[2])
    ),list(stringsAsFactors=F))))
    atmCosine <- unlist(lapply(comb,function(y) lapply(
	y,combCosAngle,as.matrix(ndists)
    )))
  }


  res <- sapply(interdist,function(y) c(range(y),mean(y),sd(y)))
  labs <- c('min','max','mean','sd')
  names(res) <- sapply(colnames(res),paste,labs,sep='.dist.')
  res <- c(res)
  if(!is.null(coords$ATM)) res <- res[c(-1,-2,-4)]
  dists <- res/maxdist

  tvcCosine <- NA
  if(!is.null(coords$TVC))if(nrow(coords$TVC)>2){
    comb <- t(combn(labels(dist(coords$TVC)),3))
    comb <- rbind(comb,comb[,c(1,3,2)],comb[,c(2,1,3)])
    
    tvcCosine <- apply(
      comb,1,combCosAngle,as.matrix(ndists)
    )
  }
  tvcCosine <- c(range(tvcCosine),sd(tvcCosine))
  names(tvcCosine) <- paste0('TVC.cosine.',labs[-3])

  atmCosine <- c(range(atmCosine),sd(atmCosine))
  names(atmCosine) <- paste0('ATM.cosine.',labs[-3])

  res <- c(
	   dists,tvcCosine#,atmCosine
  )
  names(res) <- paste0(names(res),suffix)
  #   if(nucleus) res <- c(res,extraATM=extraATM)
  if(nucleus){
	 if('Nucleus.Distance.from.Position.to.Cell.Membrane'%in%colnames(x)){
	       tmp <- x[,'Nucleus.Distance.from.Position.to.Cell.Membrane']
	 }else tmp <- rep(NA,nrow(x))
  	tmp <- tmp/maxdist
	tmp <- split(tmp,x[,2])
	tmp <- lapply(tmp,function(y) setNames(unlist(sapply(
		c(range,mean,sd),
		function(z) z(y,na.rm=T)
	)),paste0('Nucleus.Dist.to.Cell.Membrane.',c('min','max','mean','SD'))))
	res <- c(res,unlist(tmp))
  }
  res
}

groupStats <- function(x,cols=c(
	'Cell.Ellipticity..oblate.',
	'Cell.Ellipticity..prolate.',
	'Cell.Nucleus.to.Cytoplasm.Volume.Ratio',
	#         'Cell.Number.Of.Nuclei',
	'Cell.Sphericity',
	'Cell.Volume.frac',
	'Nucleus.Ellipticity..oblate.',
	'Nucleus.Ellipticity..prolate.',
	'Nucleus.Sphericity',
	#         'Nucleus.Distance.from.Position.to.Cell.Membrane',
	'Nucleus.Volume.frac'
)){
	if(all(x[,2]=="TVC")){
		x <- rbind(x,NA)
		x[nrow(x),2] <- "ATM"
	}
	x[,cols[!cols%in%names(x)]] <- NA
	x <- split(x[,cols],x[,2])
	res <- lapply(x,lapply,function(y) setNames(unlist(sapply(
		c(range,mean,sd),
		function(z) z(y,na.rm=T)
	)),c('min','max','mean','SD')))
	unlist(res)
}

membraneStats <- function(x,cols=c(
	'Ellipticity..oblate.',
	'Ellipticity..prolate.',
	'Sphericity'
)){
	#         x[,2] <- sub("ALLgroup","TVCgroup",x[,2]) #moved to labelCells()
	if(nrow(x)==1){
		x <- rbind(x,NA)
		x[,2] <- c("TVCgroup","ATMgroup")
	}
	x[,cols[!cols%in%names(x)]] <- NA
	x$Volume <- x$Volume/sum(x$Volume,na.rm=T)
	x <- split(x,x[,2])
	#         if(is.null(x$ATMgroup)) x$ATMgroup <- x$TVCgroup
	tmp <- lapply(x,'[',cols)
	res <- unlist(lapply(tmp,lapply,mean,na.rm=T))
	c(
		res,
		TVC.contiguous = as.numeric(nrow(tmp$TVC)<2),
		ATM.contiguous = as.numeric(nrow(tmp$ATM)<2),
		TVC.Volume = sum(x$TVC$Volume),
		ATM.Volume = sum(x$ATM$Volume)
	)
}

whichAngles <- function(x,suffix='_CellAngle') x$angle[paste0(c(
  'a.FHP_p.FHP_p.TVC','p.FHP_a.FHP_a.TVC',
  'a.FHP_a.TVC_p.TVC','p.FHP_a.TVC_p.TVC',
  'a.FHP_a.TVC_a.ATM','p.FHP_p.TVC_p.ATM'
),suffix)
]

getAsym <- function(x,log=T){
  res <- c(
    Cell.Asym_a.sTVC.FHP=x['a.TVC',"Cell.Volume"]/x['a.FHP',"Cell.Volume"],
    Cell.Asym_p.sTVC.FHP=x['p.TVC',"Cell.Volume"]/x['p.FHP',"Cell.Volume"],
    Nucleus.Asym_a.sTVC.FHP=x['a.TVC',"Nucleus.Volume"]/x['a.FHP',"Nucleus.Volume"],
    Nucleus.Asym_p.sTVC.FHP=x['p.TVC',"Nucleus.Volume"]/x['p.FHP',"Nucleus.Volume"],
    Cell.Asym_a.ATM.TVC=sum(
      x[c('a.ATM','a.ATM.2'),"Cell.Volume"],na.rm = T
    )/sum(
      x[c('a.TVC','a.FHP'),'Cell.Volume'],na.rm = T
    ),
    Cell.Asym_p.ATM.TVC=x['p.ATM',"Cell.Volume"]/sum(
      x[c('p.TVC','p.FHP'),'Cell.Volume'],na.rm = T
    ),
    Nucleus.Asym_a.ATM.TVC=sum(
      x[c('a.ATM','a.ATM.2'),"Nucleus.Volume"],na.rm = T
    )/sum(
      x[c('a.TVC','a.FHP'),'Nucleus.Volume'],na.rm = T
    ),
    Nucleus.Asym_p.ATM.TVC=x['p.ATM',"Nucleus.Volume"]/sum(
      x[c('p.TVC','p.FHP'),'Nucleus.Volume'],na.rm = T
    )
  )
  if(log) res <- log2(res)
  res
}

inEllipsoid <- function(pos,group){
  a <- group[,c(
	"Ellipsoid.Axis.A.X","Ellipsoid.Axis.A.Y","Ellipsoid.Axis.A.Z"
  )]/group[,"Ellipsoid.Axis.Length.A"]^2
  b <- group[,c(
	"Ellipsoid.Axis.B.X","Ellipsoid.Axis.B.Y","Ellipsoid.Axis.B.Z"
  )]/group[,"Ellipsoid.Axis.Length.B"]^2
  c <- group[,c(
		"Ellipsoid.Axis.C.X","Ellipsoid.Axis.C.Y","Ellipsoid.Axis.C.Z"
  )]/group[,"Ellipsoid.Axis.Length.C"]^2
  mat <- lapply(1:nrow(group), function(i) rbind(
    unlist(a[i,]),unlist(b[i,]),unlist(c[i,])
  ))
  v <- lapply(
    1:nrow(group), 
    function(i) as.matrix(pos-group[i,c(
      "Center.of.Homogeneous.Mass.X",
      "Center.of.Homogeneous.Mass.Y",
      "Center.of.Homogeneous.Mass.Z"
    )])
  )
  res <- mapply(function(x,y) y%*%x%*%t(y),mat, v)
}

combAngle <- function(names) {
  res <- combn(
    names,3,function(x) combn(
      x,2,function(y) c(y,x[!x%in%y])
    )
  )
  res <- lapply(1:dim(res)[3],function(x) res[,,x])
  t(do.call(cbind,res))
}

#gives cosine of angle opposite y
cosAngle <- function(x,y,z) (x^2+z^2-y^2)/(2*x*z)

#accepts vector of 3 point labels and calculates angle between them from a distance matrix
combCosAngle <- function(x,mat) cosAngle(
  mat[x[2],x[1]],
  mat[x[3],x[1]],
  mat[x[3],x[2]]
)

distCos <- function(mat) {
  comb <- combAngle(row.names(mat))
  res <- apply(
    comb,1,combCosAngle,mat
  )
  names(res) <- apply(comb,1,paste,collapse='_')
  res
}

groupLab <- function(group,cell){
	if(!is.data.frame(cell)) {
		cell <- as.data.frame(cell)
		row.names(cell) <- cell$Default.Labels
	}
  res <- do.call(data.frame,append(
      sapply(
        names(group),function(x) rep(NA,4),simplify = F
      ),
      list(row.names=c('aATMgroup','pATMgroup','aTVCgroup','pTVCgroup'))
    )
  )
  if(!is.null(group)){
    if(nrow(group)==1){
      res['aATMgroup',] <- group[1,]
      cell$group <- "ATM"
    }else {
      row.names(group) <- group[,"Default.Labels"]
      #cell$group <- c(rep("TVC",5),rep("ATM",2))
      if("ATM group"%in%row.names(group)){
        res['aATMgroup',] <- group["ATM group",]
      }else{
        res['aATMgroup',] <- group["Anterior ATM",]
        res['pATMgroup',] <- group["Posterior ATM",]
      }
      if("Group cell"%in%row.names(group)){
        res['pTVCgroup',] <- group["Group cell",]
      }else if("Posterior TVC"%in%row.names(group)){
        res['pTVCgroup',] <- group["Posterior TVC",]
        res['aTVCgroup',] <- group["Anterior TVC",]
      }
    }
  }
  res$group <- c("ATM","ATM","TVC","TVC")
  return(list(
    group=res,
    cell=cell,
    dat=c(
      ngroup=sum(!is.na(res[,1])),
      nATMgroup=sum(!is.na(res[1:2,1])),
      nTVCgroup=sum(!is.na(res[3:4,1])),
      nATMcell=sum(!is.na(cell[6:7,1])),
      nTVCcell=sum(!is.na(cell[1:5,1]))
    )
  ))
}

getGroup <- function(groupdat,id){
  res <- groupdat[id]
  
}

parseGroup <- function(groupdat,celldat){
  group <- groupLab(groupdat)
}

zscore <- function(x) {
	x <- as.numeric(x)
	x[!is.finite(x)] <- NaN
	x[x==0] <- NaN
	m <- mean(x,na.rm=T)
	x <- (x-m)/sd(x,na.rm=T)
	#         x[!is.finite(x)] <- m
	x[is.na(x)] <- 0
	x
}
zdf <- function(x){
	apply(x,2,zscore)
	#         row.names(y) <- row.names(x)
	#         as.data.frame(y)
}

mergeDat <- function(l){
	tmp <- Reduce(
		      function(x,y) merge(x,y,1,all=T),
		      lapply(l,function(x) data.frame(names(x),x,stringsAsFactors = F))
	)
	row.names(tmp) <- tmp[,1]
	tmp <- tmp[,-1]
	names(tmp) <- names(l)
	tmp
}

mergeRownames <- function(l){
	tmp <- Reduce(
		      function(x,y) merge(x,y,by.x="Row.names",by.y=0,all=T),
		      l,data.frame(Row.names=row.names(l[[1]]),stringsAsFactors = F)
	)
	row.names(tmp) <- tmp[,1]
	tmp <- tmp[,-1]
	names(tmp) <- names(l)
	tmp
}


#return 3D plot of one embryo 
pos3d <- function(x){
  require(lattice)
  res <- x[,1:3]
  names(res) <- c('x','y','z')
  pc <- as.data.frame(prcomp(res)$x)
  col <- cols[x$celltype]
  pch <- pchPos[x$cellpos]
  cex <- 5*x$Area/max(x$Area)
  return(list(
    cloud(x~y*z,data=res,pch = pch,cex = cex,col=col),
    cloud(y~x*z,data=res,pch = pch,cex = cex,col=col),
    cloud(z~y*x,data=res,pch = pch,cex = cex,col=col)
    # cloud(PC1~PC2*PC3,data=pc,pch = pch,cex = cex,col=col),
    # xyplot(PC1~PC2,pc,pch = pch,cex = cex,col=col),
    # xyplot(PC1~PC3,pc,pch = pch,cex = cex,col=col),
    # xyplot(PC2~PC3,pc,pch = pch,cex = cex,col=col)
  ))
}

