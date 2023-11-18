<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# DeePWAK
Self-supervised model selection using Deep learning of a Partitioned Weighted Affinity Kernel. A DeePWAK model aims to cluster data based on finding the partition of the data which is best able to reconstruct the data. It relies on the isomorphism between denoising, clustering, and compression. 

Batched data is optionally passed through an encoder network to obtain a dimension-reduced embedding. A distance metric, which can optionally be learned, is used to produce a distance matrix for the batch. A partition network attempts to learn classifications for each data point from the embeddings.

# Julia Dependencies
Julia 1.9

[`igraph_jll.jl`](https://github.com/fcdimitr/igraph_jll.jl)

[`leiden_jll.jl`](https://github.com/fcdimitr/leiden_jll.jl)

[`Leiden.jl`](https://github.com/pitsianis/Leiden.jl)

# R Dependencies
R 4.0+

[`dirfns`](https://github.com/kewiechecki/dirfns)

[`moreComplexHeatmap`](https://github.com/kewiechecki/moreComplexHeatmap)

# Usage
```{bash}
#MNIST example
julia main.jl

# parse segmentation data from /segdat
Rscript readEmbryos.R

# generate graph of known target protein interactions from STINGdb
Rscript interactions.R

# generate embeddings, graph, clusters
julia cluster.jl

# generate plots and characterize optimal clusterings
Rscript plot.clust.R
```

# Rationale
DeePWAK attempts to generalize [a pipeline for automatic phenotype detection in microscopy images](https://github.com/ChristiaenLab/CrobustaScreen).
Its objective function is based on [noise2self](https://arxiv.org/abs/1901.11365), a paradigm for self-supervised denoising. In broad strokes, it can be equated to batched LOOCV.

# noise2self
Let $J \in \mathcal{J}$ be independent partitions of noisy data $X$. Let $\mathcal{F}(\theta)$ be a family of predictors of $X_J$ with parameters $\theta \in \Theta$ that depends on its complement $X_{J^C}$

$$
  \hat{X}_J=\mathcal{F}(\theta)(X\_{J^C})
$$

  The optimal $\theta$ is given by

$$
  \underset{\theta}{\overset{\Theta}{\mathrm{noise2self}}}[\mathcal{F}(\theta),X] := \underset{\theta}{\overset{\Theta}{\mathrm{argmax}}}[\sum_{J}^{\mathcal{J}}\mathbb{E}||X_J-\mathcal{F}(\theta)(X_{J^C})||^2]
$$

# Graph diffusion
Our choice of $\mathcal{F}$ is adapted from [DEWAKSS](https://nyuscholars.nyu.edu/en/publications/optimal-tuning-of-weighted-knn-and-diffusion-based-methods-for-de). The parameters we want to tune generate a graph $G$ from embeddings $E$. The adjacency matrix of any graph can be treated as a transition matrix (or weighted affinity kernel) by setting the diagonal to 0 and normalizing columns to sum to 1. We'll call this the $\mathrm{WAK}$ function. For each embedding, an estimate is calculated based on its neighbors in the graph. This can be expressed as matrix multiplication.

$$
\hat{E} := \mathrm{WAK}(G)E^\top
$$

# Overview
![overview](https://github.com/kewiechecki/DeePWAK/blob/master/tikz/optimization/optimization.png?raw=true)

# Enforcing Sparcity
We want to approximate selection of an optimal number of dimensions. Because adding embedding dimensions always improves performance, we want to impose a penalty for a larger number. We also want to limit polysemanticity.
To do this we attempt to learn a weight vector $\mathbf{w}$ from $E$. This gives us the predictor family

$$
\mathcal{F}(X) = \theta^{d \to m}(G\mathbf{w}\odot \theta^{m \to d}(X)^\top)
$$

To enforce sparsity, we can add entropy of the weights to the loss.

$$
\mathcal{L}(\mathcal{F}\theta,X) = \alpha\mathrm{H}(\mathbf{w}) + \log\mathbb{E}||X,\mathcal{F}\theta(X)||^2
$$

# Modularity
We want to enforce that clusters are well-separated. Modularity gives the difference between the number of edges between nodes in a cluster and expected number of edges given the number of nodes in the cluster and average degree of the graph. It is given by

$$
      \mathcal{H} = \frac{1}{2m}\sum_{c}( \,e_c - \gamma\frac{K_c^2}{2m})
$$

where $m$ is the average degree of the graph, $e_c$ is the number of edges in cluster $c$, and $K_c$ is the number of nodes in cluster $c$.
Expected number of edges is scaled by a hyperparameter $\gamma$. 
Intuitively, normalized edge density should be greater than $\gamma$ within a cluster and less than $\gamma$ between clusters. A higher $\gamma$ results in more clusters. Modularity can easily be adapted to weighted graphs.

Modularity can also be added to the loss function.

$$
\hat{X} = \theta_d(PG\mathbf{w}\odot\theta_e(X^\top))
$$

$$
m = \frac{1}{n}\sum_i^n\sum_j^nG_{i,j}
$$

$$
\mathcal{H}(PG) = \frac{1}{2m}\sum\_{i=1}^n\sum\_{j=1}^n(PG)\_{i,j} - \frac{(PG^\top)\_{i,j}^2}{2m}
$$

$$
\mathcal{L}(X) = \alpha\mathrm{H}(\mathbf{w}) + \beta\mathrm{H}(G) + \delta\mathcal{H}(\gamma,PG) + \log\mathbb{E}||X,\hat{X}||^2
$$
