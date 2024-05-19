model.layers <- function(layers,activation){
	require(keras)
	if(length(activation)==1) {
		activation <- rep(activation,
				  length(layers)-2)
	}
	model <- keras_model_sequential()
	model %>% layer_dense(units = layers[2], 
			      activation = activation[1], 
			      input_shape = layers[1])
	layerfn <- function(x,i){
		layer_dense(x, units=layers[i+1],
			    activation=activation[i])
	}
	model <- Reduce(layerfn,2:length(activation),model)
	#         for (i in 2:length(activation)){
	#                 model %>% layer_dense(units=layers[i+1],
	#                                       activation=activation[i])
	#         }
	model %>% layer_dense(units=layers[length(layers)])
	return(model)
}


get.model <- function(x,y,layers,activation='tanh',
		      loss='mean_squared_error',
		      optimizer='adam',epochs=10000){
	require(keras)
	model <- model.layers(c(ncol(x),layers),activation)
	model %>% compile(
	  loss = loss, 
	  optimizer = optimizer
	)

	# fit model
	model %>% fit(
	  x = as.matrix(x), 
	  y = as.matrix(y), 
	  epochs = epochs,
	  verbose = 0
	)
	return(model)
}

denoise <- function(x,y,layers,out,...){
	require(keras)
	require(dirfns)
	out <- mkdate(out,'')
	model <- get.model(x,y,layers,'tanh',...)
	save_model_hdf5(model,paste0(out,'.model'))
	return(model)
}

autoencode <- function(dat,layers,groups,out,...){
	require(dirfns)
	require(keras)
	out <- mkdate(out,'')
	layersfull <- c(layers,
			rev(layers[-length(layers)]),
			ncol(dat))
	model <- get.model(dat,dat,layersfull,'tanh',...)
	model.eval(model,as.matrix(dat),as.matrix(dat),
		   groups,length(layers),out)
	save_model_hdf5(model,paste0(out,'.model'))
	return(model)
}

plot.clust <- function(dat,clust,out,x=V1,y=V2,labs=names(cols),title='',width=20,ncols=3,...){
	require(ggplot2)
	require(ggpubr)

	clust <- as.data.frame(clust)
	cols <- apply(clust,2,as.character)
	cols <- as.data.frame(cols)
	names(cols) <- make.names(names(clust))

	nrows <- ceiling(ncol(cols)/ncols)
	height <- width/ncols*nrows

	dot <- function(z,dat,cols) {
		dat <- cbind(as.data.frame(dat),cols)
		ggplot(dat,
		       aes_string(x=names(dat)[1],
				  y=names(dat)[2],
				  col=z)) + geom_point()
	}
	if(!is.null(dim(dat))){
		plots <- lapply(names(cols),dot,dat,cols)
	}else{
		plots <- mapply(dot,names(cols),dat,MoreArgs = list(cols=cols),SIMPLIFY = F)
	}
	g <- ggarrange(plotlist=plots,
		       labels=labs,ncol=ncols,
		       nrow=nrows)

	g <- annotate_figure(g, top = text_grob(title,
		       face = "bold", size = 14))

	ggexport(g,filename=paste0(out,'.pdf'),
		 width=width,height=height,...)
}

arrange.plots <- function(plots,out,labs=names(plots),
			  title='',width=20,ncols=3,...){
	require(ggplot2)
	require(ggpubr)

	nrows <- ceiling(ncol(cols)/ncols)
	height <- width/ncols*nrows

	g <- ggarrange(plotlist=plots,
		       labels=labs,ncol=ncols,
		       nrow=nrows)

	ggexport(g,filename=paste0(out,'.pdf'),
		 width=width,height=height,...)
}

model.eval <- function(model,x,y,groups,outix,out){
	require(keras)
	#         require(umap)
	mse <- evaluate(model, x, y)

	#         outmodel <- keras_model(inputs = model$input, 
	#                                 outputs = get_layer(model, 
	#                                                     index=outix)$output)
	#         outlayer <- predict(outmodel, x)
	outlayer <- model.out(model,x,outix)
	#         model.umap <- umap(outlayer)$layout
	#         colnames(model.umap) <- c("UMAP1","UMAP2")
	#         dir.tab(model.umap,out)
	plot.clust(outlayer[,1:2],#model.umap,
		   groups,out,
	       title=paste('MSE =',
			   as.character(signif(mse,6))),
	       width=20)
}

model.out <- function(model,x,index){
	require(keras)
	outmodel <- keras_model(inputs = model$input, 
				outputs = get_layer(model, 
						    index=index)$output)
	res <- predict(outmodel, x)
	return(res)
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

knn.adj <- function(knn,dat){
	require(igraph)
	dists <- as.matrix(dist(dat))
	knn <- as_adjacency_matrix(knn)
	return(sapply(1:ncol(knn),
		     function(i){
			     x <- dists[,i]
			     x[!as.logical(knn[,i])] <- NaN
			     return(x)
		     }))
}

get.clusts <- function(dists,k,res=0.5,...){
	require(leiden)
	require(igraph)
	
	g <- get.knn(dists,k)
	clust <- leiden(g,resolution_parameter=res,...)
	return(clust)
}

test.knn <- function(dat,k,clust,sample=0.9,reps=1000,...){
	n.train <- round(nrow(dat)*sample)
	#         traindat <- dat[trainsel,]
	#         testdat <- dat[-trainsel,]
	cv <- replicate(reps,knn.err(dat,k,clust,n.train))
	cv[!is.finite(cv)] <- -log2(1/(2+n.train))
	err <- mean(cv)
	return(err)
}

knn.err <- function(dat,k,clust,n.train){
	require(class)

	trainsel <- sample(1:n.train)
	traindat <- dat[trainsel,]
	testdat <- dat[-trainsel,]

	cv <- class::knn(traindat,testdat,clust[trainsel],k)
	err <- sum(cv!=clust[-trainsel])/n.train
	return(-log2(err))
}

encoded.leiden <- function(model,x,layer,out,#groups,
			   res=seq(0.1,1,0.1),
			   k=3:21){
	require(dirfns)
	require(umap)
	dat <- model.out(model,x,layer)
	colnames(dat) <- paste0('encoding',
				as.character(1:ncol(dat)))
	dir.tab(dat,'encoded',out)

	#         model.eval(model,as.matrix(dat),as.matrix(dat),
	#                    groups,length(model$layers),out)

	model.umap <- umap(dat)$layout
	colnames(model.umap) <- c("UMAP1","UMAP2")
	dir.tab(model.umap,'umap',out)

	dists <- as.matrix(dist(dat))
	dir.tab(dists,'dists',out)

	clusts <- lapply(res,
			 function(x){
				clust <- sapply(k,function(y) get.clusts(dists,y,x))
				clust <- apply(clust,2,as.character)
				colnames(clust) <- paste0('k',as.character(k))
				return(clust) 
			 })

	names(clusts) <- paste0('leiden',as.character(res))
	dir.apply(clusts,paste0(out,'/clusts'))

	err <- leiden.err(dat,clusts,k)
	row.names(err) <- paste0('k',as.character(k))
	dir.tab(err,'log2error',out)
	return(list(clusts=clusts,log2error=err))
}

get.umap <- function(k,dat,out='',...) {
	require(umap)
	model.umap <- umap(dat,n_neighbors=k,...)$layout
	colnames(model.umap) <- c("UMAP1","UMAP2")
	if(out!='') dir.tab(model.umap,
			    paste0('umap_k',
				   as.character(k)),
			   out, append.date=F)
	return(model.umap)
}

umap.leiden <- function(model,x,layer,out,groups,
			   res=seq(0.1,1,0.1),
			   k=3:21){
	require(dirfns)
	require(umap)
	dat <- model.out(model,x,layer)
	colnames(dat) <- paste0('encoding',
				as.character(1:ncol(dat)))
	dir.tab(dat,'encoded',out)

	model.eval(model,as.matrix(dat),as.matrix(dat),
		   groups,length(model$layers),out)

	umaps <- lapply(k,get.umap)

	dists <- as.matrix(dist(model.umap))
	dir.tab(dists,'dists',out)

	dists <- as.matrix(dist(dat))
	dir.tab(dists,'dists',out)

	clusts <- lapply(res,
			 function(x){
				clust <- sapply(k,function(y) get.clusts(dists,y,x))
				clust <- apply(clust,2,as.character)
				colnames(clust) <- paste0('k',as.character(k))
				return(clust) 
			 })

	names(clusts) <- paste0('leiden',as.character(res))
	dir.apply(clusts,paste0(out,'/clusts'))

	err <- leiden.err(dat,clusts,k)
	row.names(err) <- paste0('k',as.character(k))
	dir.tab(err,'log2error',out)
	return(list(clusts=clusts,log2error=err))
}

leiden.err <- function(dat,clusts,k){
	err <- sapply(clusts,
		      function(clust){
			      mapply(test.knn,k=k,
				     clust=as.data.frame(clust),
				     MoreArgs=list(dat=dat))
		      })
	return(err)
}

plot.leiden <- function(dat,clust,err,out,title='',...){
        cols <- as.data.frame(apply(clust,2,as.character))
        names(cols) <- colnames(clust)
        lab <- paste('k =',sub('k','',names(cols)),' -log2(err) =',as.character(round(err,2)))
        plot.clust(dat,cols,out,labs=lab,...)
}

plot.umaps <- function(dat,ks,clusts,err,out,...){
	dats <- lapply(ks,get.umap,dat,out)
	plot.leiden(dats,clusts,err,out,...)
}

read.clusts <- function(dir){
	require(dirfns)
	clusts <- lrtab(dir,pattern='\\.txt$')
	clusts$dists <- as.matrix(clusts$dists)
	clusts$clusts <- lrtab(paste0(dir,'/clusts'),
			       pattern='\\.txt')
	clusts$clusts <- do.call(function(...) { 
				mapply(cbind,...,SIMPLIFY=F) 
		      }, clusts$clusts)
	return(clusts)
}

read.model <- function(m,dat){
	require(keras)
	res <- list(model=load_model_hdf5(m))
	res$err <- evaluate(res$model,dat,dat)
	res$nlayer <- length(res$model$layers)
	res$encoded <- model.out(res$model,dat,res$nlayer/2)
	colnames(res$encoded) <- paste0('encoding',
					as.character(1:ncol(res$encoded)))
	res$bottleneck <- ncol(res$encoded)
	res$aic <- 2*res$bottleneck-2*log(1-res$err)
	res$dists <- as.matrix(dist(res$encoded))
	return(res)
}

aic.table <- function(models) 
	sapply(c('bottleneck','nlayer','err','aic'), 
	       function(x) sapply(models,'[[',x))

plot.leidens <- function(dir,dat='umap') {
	require(dirfns)
	out <- paste0(dir,'/',dat)
	dir.create(out)
	clusts <-read.clusts(dir) 
	fn <- function(x){
		plot.leiden(clusts[[dat]],clusts$clusts[[x]],
			    clusts$log2error[,x],
			    paste0(out,'/',x),
			    paste('res =',
				  sub('leiden','',x)))
	}
	f2 <- function(x){
		plot.umaps(clusts[[dat]],
			   as.numeric(sub('k','',
					  names(clusts$clusts[[x]]))),
			   clusts$clusts[[x]],
			    clusts$log2error[,x],
			    paste0(out,'/',x),
			    paste('res =',
				  sub('leiden','',x)))
	}
	sel <- names(clusts$clusts)
	sapply(sel,fn)
}

mse.plot <- function(model,x,y,model.umap,groups,out,...){
	require(keras)
	mse <- evaluate(model, x, y)

	plot.clust(model.umap,groups,out,
	       title=paste('MSE =',
			   as.character(signif(mse,6))),
	       ...)
}

leiden.sil <- function(clusts,out){
	require(cluster)

	dists <- read.delim(paste0(out,'/dists.txt'))
	sil <- lapply(clusts,function(x) lapply(lapply(as.data.frame(x),as.numeric),silhouette,dists))
}
			      
