<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>
$$
\def\txtop#1{\mathop{\mathrm{#1}}\limits}
\def\argmin{\txtop{argmin}}
\def\MSE{\txtop{MSE}}
\def\ntos{\txtop{noise2self}}
\def\PCA{\txtop{PCA}}
\def\kNN{\txtop{kNN}}
\def\WAK{\txtop{WAK}}
\def\leiden{\txtop{leiden}}
\def\rdim{\txtop{reducedimension}}
\def\softmax{\txtop{softmax}}
\def\layernorm{\txtop{layernorm}}
\def\DEWAK{\txtop{DEWAK}}
\def\DEPWAK{\txtop{DEPWAK}}
\def\DeePWAK{\txtop{DeePWAK}}
$$

# DeePWAK
Self-supervised model selection using Deep learning of a Partitioned Weighted Affinity Kernel.

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
