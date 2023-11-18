<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

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

# Rationale
DeePWAK attempts to generalize [a pipeline for automatic phenotype detection in microscopy images](https://github.com/ChristiaenLab/CrobustaScreen). 

# Overview
![overview](https://github.com/kewiechecki/DeePWAK/blob/master/tikz/optimization/optimization.png?raw=true)
