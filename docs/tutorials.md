# Tutorials

Ten Jupyter notebooks, committed with their outputs so they can be read
directly on GitHub. Open them in order: notebook 0 covers the GNN background,
notebook 1 introduces sheaves and the library's training pipeline on a real
dataset, and notebooks 2-9 each study one model family on small synthetic
graphs. Each notebook has theory with equations, executable experiments, an
interpretability section that opens up the trained model, and exercises.

| # | Notebook | Focus |
|---|---|---|
| 0 | [GNN basics](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/0_GNNs_Basics.ipynb) | Message passing, spectral view, GCN, GraphSAGE, GAT, GIN and their limits |
| 1 | [Sheaf Neural Networks](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/1_Sheaf_Neural_Networks_Introduction.ipynb) | Cellular sheaves, the sheaf Laplacian, Neural Sheaf Diffusion, training on Texas |
| 2 | [Heterogeneous sheaves](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/2_Heterogeneous_Neural_Networks.ipynb) | Node and relation types in the restriction maps |
| 3 | [Directed sheaves](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/3_Directed_Sheaf_Neural_Networks.ipynb) | Directed NSD and the complex directed sheaf Laplacian |
| 4 | [Polynomial NSD](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/4_Polynomial_Neural_Sheaf_Diffusion.ipynb) | Chebyshev spectral filters on the sheaf Laplacian |
| 5 | [Bayesian sheaves](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/5_Bayesian_Sheaf_Neural_Sheaf_Diffusion.ipynb) | Distributions over restriction maps, ELBO, uncertainty |
| 6 | [Bundle networks](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/6_Bundle_Neural_networks.ipynb) | Per-node frames and heat-kernel message diffusion |
| 7 | [Cooperative sheaves](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/7_Cooperative_Sheaf_Neural_Networks.ipynb) | Conformal outgoing and incoming maps |
| 8 | [Hypergraph sheaves](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/8_Hypergraph_Sheaf_Neural_Networks.ipynb) | Design notebook: implement the model yourself, checked against reference computations |
| 9 | [Directed hypergraph sheaves](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/9_Directed_Hypergraph_Sheaf_Neural_Networks.ipynb) | Design notebook: Hermitian directed hypergraph Laplacian |

To run them locally, install the `notebooks` extra (or run `uv sync`)
and launch `jupyter notebook` from the repository root. Notebook 1 downloads
Texas on first use; the others need no download and run on CPU in under a
minute each.
