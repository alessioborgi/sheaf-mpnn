# Sheaf neural network hands-on notebooks

Open these notebooks in numerical order. Notebook 0 covers the graph neural network background, notebook 1 introduces sheaves and the repository pipeline on a real dataset, and notebooks 2–9 each study one model family on small synthetic graphs. Every notebook follows the same structure: theory with equations, figures from the original paper, executable experiments, an interpretability section that opens up the trained model, and exercises.

| Notebook | Focus | Interpretability analyses |
|---|---|---|
| 0 | GNN basics: graphs as tensors, message passing, spectral view, GCN, GraphSAGE, GAT, GIN, limitations | attention maps, gradient-based edge importance, oversmoothing, heterophily, oversquashing, 1-WL limits |
| 1 | Cellular sheaves, the sheaf Laplacian, Neural Sheaf Diffusion, repository data and Lightning training (Texas) | restriction-map heatmaps, clustering, exemplar edges, transport alignment on same-label vs cross-label edges |
| 2 | Heterogeneous sheaves (HetSheaf): node and relation types in the restriction maps | maps per relation type, counterfactual relation types, per-relation generators |
| 3 | Directed sheaves: the repository's Directed NSD and the complex directed sheaf Laplacian | role of the direction weight α, learnable α, edge-reversal test, spectrum vs charge q |
| 4 | Polynomial NSD: Chebyshev spectral filters on the sheaf Laplacian | learned frequency responses with the learned spectrum, Jacobian influence vs hop distance |
| 5 | Bayesian sheaves: distributions over restriction maps, ELBO, uncertainty | entropy vs epistemic variance, selective prediction, corrupted inputs, posterior over maps, Monte Carlo error |
| 6 | Bundle networks: per-node frames and heat-kernel message diffusion | learned frames, relative transports, Taylor vs spectral solver |
| 7 | Cooperative sheaves: conformal outgoing and incoming maps | who broadcasts and who listens, per layer |
| 8 | Hypergraph sheaves (design notebook, TODO model) | clique-expansion information loss, linear sheaf hypergraph Laplacian checks, normalization pitfall, oversmoothing vs global sections |
| 9 | Directed hypergraph sheaves (design notebook, TODO model) | Hermitian directed Laplacian checks, reduction to the magnetic Laplacian, net flow |

Use the repository's Python 3.13 environment (`uv sync`), launch `uv run jupyter notebook` from the repository root, and select that environment as the kernel. Run each notebook from a fresh kernel, top to bottom. Notebook 1 downloads Texas and its splits on first use and runs a 10-fold cross-validation; the other notebooks need no download and run on CPU in one to two minutes each. The notebooks are committed with their outputs, so they can be read on GitHub without running anything; re-run them from a fresh kernel to regenerate the outputs. Numbers quoted in markdown cells describe typical behaviour on the toy graphs, not benchmark results.

The current torch 2.8.0 pin has no Intel macOS wheel. Use a supported Linux/Windows machine or Apple Silicon macOS for the locked environment.

For notebooks 8 and 9, leave `RUN_STUDENT_IMPLEMENTATION = False` until the TODO methods are implemented. The intentional `NotImplementedError` methods are not called during the default walkthrough, and the reference computations in each notebook double as a test suite for your implementation.

## Validation status

All ten notebooks were executed top to bottom without errors in the locked environment (Python 3.13, PyTorch 2.8.0, PyTorch Geometric 2.7.0, Lightning 2.6.0, CPU, Apple Silicon macOS) on 2026-11-03, and the committed outputs come from that run. They pass `ruff check` and `ruff format --check` with the repository configuration.

## Figure credits

Figures from the original papers are embedded as notebook attachments for teaching purposes, each with a caption that cites the source. Several papers are distributed under arXiv's non-exclusive distribution licence, which does not grant reuse rights by itself; check with the authors before redistributing those figures outside this repository.

| Notebook | Paper | arXiv | Figures used | Licence |
|---|---|---|---|---|
| 0 | Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks* | [1609.02907](https://arxiv.org/abs/1609.02907) | Fig. 1 (left), Fig. 3 | arXiv non-exclusive |
| 0 | Hamilton, Ying & Leskovec (2017), *Inductive Representation Learning on Large Graphs* | [1706.02216](https://arxiv.org/abs/1706.02216) | Fig. 1 | arXiv non-exclusive |
| 0 | Veličković et al. (2018), *Graph Attention Networks* | [1710.10903](https://arxiv.org/abs/1710.10903) | Fig. 1 (recompiled from the TikZ source), Fig. 2 | arXiv non-exclusive |
| 0 | Xu et al. (2019), *How Powerful are Graph Neural Networks?* | [1810.00826](https://arxiv.org/abs/1810.00826) | Figs. 1, 2, 3 | arXiv non-exclusive |
| 1 | Bodnar et al. (2022), *Neural Sheaf Diffusion* | [2202.04579](https://arxiv.org/abs/2202.04579) | Figs. 1, 2, 3, 4, 5 | arXiv non-exclusive |
| 2 | Braithwaite et al. (2024), *Heterogeneous Sheaf Neural Networks* | [2409.08036](https://arxiv.org/abs/2409.08036) | Fig. 1; Figs. 2 and 3 recompiled from the TikZ sources | CC BY 4.0 |
| 3 | Fiorini et al. (2025), *Sheaves Reloaded: A Directional Awakening* | [2506.02842](https://arxiv.org/abs/2506.02842) | Fig. 1 | arXiv non-exclusive |
| 4 | Borgi, Silvestri & Liò (2025), *Polynomial Neural Sheaf Diffusion* | [2512.00242](https://arxiv.org/abs/2512.00242) | Fig. 1, two appendix figures | CC BY-NC-SA 4.0 |
| 5 | Gillespie et al. (2025), *Bayesian Sheaf Neural Networks* | [2410.09590](https://arxiv.org/abs/2410.09590) | Fig. 2 (recompiled from the TikZ source), Fig. 3 | CC BY-NC-SA 4.0 |
| 6 | Bamberger et al. (2025), *Bundle Neural Networks for message diffusion on graphs* | [2405.15540](https://arxiv.org/abs/2405.15540) | Figs. 2, 3 | CC BY 4.0 |
| 7 | Ribeiro et al. (2025), *Cooperative Sheaf Neural Networks* | [2507.00647](https://arxiv.org/abs/2507.00647) | Figs. 1, 4 | arXiv non-exclusive |
| 8 | Duta et al. (2023), *Sheaf Hypergraph Networks* | [2309.17116](https://arxiv.org/abs/2309.17116) | Figs. 1, 2, one appendix figure | arXiv non-exclusive |
| 9 | Mule et al. (2025), *Directional Sheaf Hypergraph Networks* | [2510.04727](https://arxiv.org/abs/2510.04727) | Fig. 1, two appendix figures | CC BY 4.0 |

The large figure at the top of notebook 1 is the repository's own (`img/MPSNN.pdf`).
