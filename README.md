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
  \hat{X}_J=\mathcal{F}(\theta)(X
$$

  The optimal $\theta$ is given by

$$
  \underset{\theta}{\overset{\Theta}{\mathrm{noise2self}}}[\mathcal{F}(\theta),X] := \underset{\theta}{\overset{\Theta}{\mathrm{argmax}}}[\sum_{J}^{\mathcal{J}}\mathbb{E}||X_J-\mathcal{F}(\theta)(X_{J^C})||^2]
$$

# Overview
![overview](https://github.com/kewiechecki/DeePWAK/blob/master/tikz/optimization/optimization.png?raw=true)

# Enforcing Sparcity
