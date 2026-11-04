# Models

Every family follows the same `encoder -> L sheaf conv layers -> decoder`
structure. Node `v` carries a stalk $X_v \in \mathbb{R}^{d \times f}$
(`stalk_dim=d`, `hidden_dim=f`) and each node-edge incidence a learned
restriction map. Families differ in how the maps are parameterised and how
messages are combined. Every convolution subclasses
{py:class}`~sheaf_mpnn.base_conv.BaseSheafConv`, itself a
`torch_geometric.nn.MessagePassing`.

| `--model.type` | Family | Variants | Package |
|---|---|---|---|
| `NSD` | Neural Sheaf Diffusion (Bodnar et al., 2022) | `diagonal`, `general`, `orthogonal`, `low_rank`, `general_attention`, `orthogonal_attention` | {py:mod}`sheaf_mpnn.nsd` |
| `POLY_NSD` | NSD with a Chebyshev polynomial filter (Borgi et al., 2025) | as NSD | {py:mod}`sheaf_mpnn.nsd` |
| `CONN_NSD` | NSD with connection-Laplacian maps from local PCA (no learned maps) | none | {py:mod}`sheaf_mpnn.nsd` |
| `DNSD` | Deep NSD (Bourgerie et al., 2026): tanh/gated layers with post-norm | as NSD | {py:mod}`sheaf_mpnn.dnsd` |
| `NSP` | Neural Sheaf Propagation: second-order (wave) recurrence | as NSD | {py:mod}`sheaf_mpnn.nsp` |
| `SHEAF_ADJ` | Adjacency-normalised NSD | as NSD | {py:mod}`sheaf_mpnn.sheaf_adj` |
| `SHEAF_ATTN` | Sheaf Attention Networks | `sheafan`, `res_sheafan` | {py:mod}`sheaf_mpnn.sheaf_attn` |
| `BSNN` | Bayesian Sheaf NN (Gillespie et al., 2025): variational maps, ELBO loss | `diagonal`, `general`, `orthogonal`, `low_rank` | {py:mod}`sheaf_mpnn.bsnn` |
| `NLSD` | Nonlinear Sheaf Diffusion (Zaghen et al., 2024) | `diagonal`, `orthogonal`, `general`, `low_rank` | {py:mod}`sheaf_mpnn.nlsd` |
| `BUNN` | Bundle Neural Networks (Bamberger et al., 2025): heat-kernel diffusion on a flat bundle | none | {py:mod}`sheaf_mpnn.bunn` |
| `DIRECTED_NSD` | Directed sheaf diffusion (DSNN-style); use with `--directed` | as NSD | {py:mod}`sheaf_mpnn.directed_nsd` |
| `CSNN` | Cooperative Sheaf NN with a GNN map learner (Ribeiro et al., 2025) | `conformal`, `non_conformal` | {py:mod}`sheaf_mpnn.csnn` |
| `ID_SHEAF`, `JOINT_DIFF`, `ROT_INV` | Identity-sheaf, joint-diffusion and rotation-invariant sheaves (Hernandez Caralt et al., 2024) | none | {py:mod}`sheaf_mpnn.id_sheaf`, {py:mod}`sheaf_mpnn.jointdiff` |
| `HET_*` | Heterogeneous mirrors of every family above: restriction maps conditioned on node and edge types (Braithwaite et al., 2024) | as the homogeneous family | {py:mod}`hetsheaf_mpnn` |

For graph-level tasks, {py:class}`~sheaf_mpnn.sheaf_pool.SheafPool`
is a gauge-invariant readout over sheaf stalks.

## Restriction-map variants

Shared by the diffusion families:

| Variant | Map per edge | Parameters per edge | Notes |
|---|---|---|---|
| `diagonal` | diagonal $d \times d$ | $d$ | Cheapest |
| `general` | dense $d \times d$ | $d^2$ | Most expressive |
| `orthogonal` | $O(d)$ matrix | $d(d-1)/2$ | `orth_strategy` in `{cayley, fasth, fasthpp, householder}` |
| `low_rank` | rank-$r$ factor | $2dr$ | `rank=r` |
| `general_attention`, `orthogonal_attention` | as above, attention-initialised | | NSD-style families only |

Several families ship a longer write-up next to their code:
[NSD](https://github.com/alessioborgi/sheaf-mpnn/blob/main/src/sheaf_mpnn/nsd/README.md),
[NSP](https://github.com/alessioborgi/sheaf-mpnn/blob/main/src/sheaf_mpnn/nsp/README.md),
[BSNN](https://github.com/alessioborgi/sheaf-mpnn/blob/main/src/sheaf_mpnn/bsnn/README.md) and
[CSNN](https://github.com/alessioborgi/sheaf-mpnn/blob/main/src/sheaf_mpnn/csnn/README.md).
