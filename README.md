<div align="center">

# Sheaf-MPNN: A Library for Sheaf Neural Networks as Message Passing

Alessio Borgi<sup>1,2,∗</sup>,
Mario Severino<sup>1</sup>,
Luke Braithwaite<sup>1</sup>,
Steve Azzolin<sup>3</sup>,
Rémi Bourgerie<sup>4</sup>,
Olga Zaghen<sup>5</sup>,
Emanuele Mule<sup>6</sup>,
Riccardo Ali<sup>1</sup>,
Leo Minh Kustermann<sup>7</sup>,
Edoardo Coppola<sup>1</sup>,
Antonio Purificato<sup>2</sup>,
Stefano Fiorini<sup>8</sup>,
Christopher Irwin<sup>1</sup>,
Filippo Spiga<sup>9</sup>,
Fabrizio Silvestri<sup>2</sup>,
Michael Bronstein<sup>10,11</sup>,
Pietro Liò<sup>1</sup>

<sup>1</sup>University of Cambridge ·
<sup>2</sup>Sapienza University of Rome ·
<sup>3</sup>University of Trento ·
<sup>4</sup>KTH Royal Institute of Technology ·
<sup>5</sup>University of Amsterdam ·
<sup>6</sup>Télécom Paris, Institut Polytechnique de Paris ·
<sup>7</sup>ETH Zürich ·
<sup>8</sup>Independent researcher ·
<sup>9</sup>NVIDIA ·
<sup>10</sup>University of Oxford ·
<sup>11</sup>AITHYRA

<sup>∗</sup>Corresponding author

[![arXiv](https://img.shields.io/badge/arXiv-2611.04670-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.04670)
[![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/alessioborgi/sheaf-mpnn)
[![Docs](https://img.shields.io/badge/Docs-online-2ea44f?logo=readthedocs&logoColor=white)](https://alessioborgi.github.io/sheaf-mpnn/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.6-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-%E2%89%A52.6-3c2179)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

## Overview 🚀

> **TL;DR.** Sheaf-MPNN is a PyTorch Geometric library that reformulates sheaf
> diffusion as local incidence-wise transport and aggregation, eliminating the
> need to construct the global sheaf Laplacian. It implements fifteen families of
> Sheaf Neural Networks under a unified message-passing framework, each supporting
> homogeneous and heterogeneous, directed and undirected graphs, with a benchmark
> suite (unified `sheaf` CLI, 1,300+ ready-made presets, 900+ shipped
> hyperparameter sweeps) covering 20+ node-, graph- and link-level datasets.

📄 **Paper:** [arXiv:2611.04670](https://arxiv.org/abs/2611.04670) · 📚 **Documentation:** <https://alessioborgi.github.io/sheaf-mpnn/>

![Sheaf-MPNN](img/MPSNN-1.png)

## Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset access](#dataset-access)
4. [Quick start](#quick-start)
5. [Repository layout](#repository-layout)
6. [The `sheaf` command line](#the-sheaf-command-line)
7. [Models](#models)
8. [Datasets](#datasets)
9. [Using the library from Python](#using-the-library-from-python)
10. [Hyperparameter sweeps](#hyperparameter-sweeps)
11. [Weights & Biases](#weights--biases)
12. [Efficiency benchmark vs. the original NSD code](#efficiency-benchmark-vs-the-original-nsd-code)
13. [Development](#development)
14. [Citation](#citation)
15. [Contributing](#contributing)
16. [License](#license)

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11, 3.12 or 3.13 | The locked environment uses 3.13 (`.python-version`); `uv` downloads it if missing. |
| Linux, macOS or Windows | Tested in CI on all three. |
| NVIDIA GPU (optional) | Everything falls back to CPU automatically. In the locked environment the Linux/Windows wheels are PyTorch 2.8 + CUDA 12.8 and macOS gets CPU wheels. |
| Network access | Datasets and canonical splits are downloaded on first use. |

## Installation

### With pip

The library itself (`sheaf_mpnn`, `hetsheaf_mpnn`) only needs PyTorch, PyTorch
Geometric, NumPy and SciPy:

```bash
pip install "sheaf_mpnn @ git+https://github.com/alessioborgi/sheaf-mpnn"                # the library
pip install "sheaf_mpnn[experiments] @ git+https://github.com/alessioborgi/sheaf-mpnn"   # plus the `sheaf` CLI, datasets and sweeps
pip install "sheaf_mpnn[all] @ git+https://github.com/alessioborgi/sheaf-mpnn"           # everything
```

| Extra | Adds |
|---|---|
| `experiments` | The `sheaf` command line: training loop, dataset loaders, presets, Optuna sweeps |
| `wandb` | Weights & Biases logging |
| `benchmark` | The efficiency benchmark against the original NSD code |
| `notebooks` | Jupyter and matplotlib for the tutorial notebooks |
| `householder` | The compiled `torch-householder` kernel (a numerically identical native fallback is used without it) |

### From source with uv (reproducible environment)

[`uv.lock`](uv.lock) pins a fully reproducible environment, including the
optional compiled PyG extensions:

```bash
git clone https://github.com/alessioborgi/sheaf-mpnn.git
cd sheaf-mpnn
uv sync                       # creates .venv with every extra and the dev tools
source .venv/bin/activate     # or prefix every command with `uv run`
sheaf --help                  # should print the four sub-commands
```

The efficiency benchmark against the original NSD code additionally needs the
git submodule under `external/`; see
[that section](#efficiency-benchmark-vs-the-original-nsd-code). Nothing else
requires it.

## Dataset access

All datasets are downloaded on first use and cached under `exp/data/`
(gitignored). No account or token is needed.

The citation and Platonov graphs we benchmark are the *directed* DSNN variants,
whose edge lists, labels and masks differ from the public originals, so they are
served from this repository's `dataset-assets-v1` release:

| Source | Datasets |
|---|---|
| `dataset-assets-v1` release | `cora`, `citeseer`, `pubmed`; `amazon_ratings`, `chameleon_filtered`, `minesweeper`, `questions`, `roman_empire`, `squirrel_filtered`, `tolokers`; the synthetic `dataset_nodes500_*` graphs |
| Release mirror first, upstream as fallback | `chameleon`, `squirrel`, `cornell`, `texas`, `wisconsin`, `film`, `telegram` |
| Upstream | `ogbn_arxiv`, `ogbn_products`, `ogbn_proteins` (OGB), `pokec` (LINKX, via Google Drive), HGB (`imdb`, `dblp`, `acm`, `freebase`), TU graph classification sets, `lastfm`, `movielens` |

Set `SHEAF_DATASET_REPO=<owner>/<name>` (and optionally `SHEAF_DATASET_TAG`) to
download from a fork or mirror. If `GITHUB_TOKEN` or `GH_TOKEN` is set, or a git
credential helper provides a token, the authenticated GitHub API is used instead
of the public download URL.

The canonical Geom-GCN 10-fold splits for the nine NPZ-split datasets are
fetched automatically into `src/exp/splits/` on first run. You can pre-fetch or
regenerate them:

```bash
sheaf splits                          # all nine NPZ-split datasets
sheaf splits --datasets cora texas    # a subset
sheaf splits --source generate        # stratified 48/32/20 splits made locally
```

## Quick start

Train Neural Sheaf Diffusion on Cora with its built-in best-known configuration
(10-fold cross-validation, early stopping on validation loss):

```bash
sheaf run --preset cora
```

A sanity check that finishes in a few seconds on GPU or CPU: cap the epochs
and folds. Every preset field can be overridden this way.

```bash
sheaf run --preset cora --optim.epochs 3 --cv.folds 2
```

The run prints a panel with the dataset statistics, the split sizes and a
parameter summary, then one row per fold and the mean +/- std of the test
metric:

```
 nsd-general on cora [10 folds]
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃        Fold ┃       Test acc ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│           0 │       ...      │
├─────────────┼────────────────┤
│ Mean +- Std │   xx.xx +- x.x%│
└─────────────┴────────────────┘
```

Checkpoints are written to a temporary directory and deleted after each fold;
the printed table (or W&B, see below) is the record of the run.

## Repository layout

```
src/sheaf_mpnn/        The library: one sub-package per model family, each with
                       <family>_layers.py (message-passing convs) and
                       <family>_model.py (encoder -> convs -> decoder).
src/hetsheaf_mpnn/     Heterogeneous-graph mirrors of the families (type-aware
                       restriction maps for HGB datasets).
src/exp/               Experiment runner.
  cli.py               The `sheaf` entry point (run / splits / sweep / benchmark).
  run.py               10-fold CV on node classification (sheaf run).
  het_run.py           Same loop for HGB heterogeneous datasets.
  gc_run.py            Same loop for TU graph-classification datasets.
  config.py            Typed config dataclasses = every CLI flag.
  registries/          dataset_registry, model_registry, preset_registry.
  data.py, datasets/   Dataset loaders and Lightning DataModules.
  sweeps/              Optuna sweep runner and the YAML schema.
  benchmark/           Efficiency benchmark vs. the original NSD repo.
configs/<dataset>/<family>/<variant>.yaml   Shipped sweep definitions.
scripts/               Sweep launchers (bash), dataset mirror/verify helpers.
tests/                 pytest suite; tests/integration holds the slow ones.
notebooks/             Tutorial series, 0 (GNN basics) to 9 (directed hypergraph sheaves).
docs/                  Documentation site (Sphinx; API reference generated from docstrings).
results/bestconf/      Best hyperparameters found by the authors' sweeps.
exp/data/              Dataset cache (gitignored, created on first run).
external/              Submodule with the original NSD implementation.
```

## The `sheaf` command line

```
sheaf run        10-fold cross-validation on a node-classification dataset
sheaf splits     download or generate the Geom-GCN train/val/test splits
sheaf sweep      YAML-driven Optuna hyperparameter sweep
sheaf benchmark  submit / collect / report the efficiency benchmark
```

Add `--help` after any sub-command for its flags. `python -m exp.run` is the
equivalent of `sheaf run` and still works.

### Presets

A preset is a complete, named configuration (dataset, model family,
restriction-map variant, training budget). There are over 1,300 of them, all
generated from two tables in `src/exp/registries/presets.py`. Keys follow one
pattern:

| Key | Meaning | Example |
|---|---|---|
| `<dataset>` | NSD with that dataset's default variant | `cora`, `texas`, `roman_empire` |
| `<dataset>_<family>_<variant>` | One family/variant on that dataset | `cora_nsd_orthogonal`, `texas_nsp_diagonal`, `questions_bsnn_general` |
| `<dataset>_<family>` | Families with no variant grid | `cora_conn_nsd`, `cora_id_sheaf`, `cora_bunn`, `cora_csnn`, `cora_sheafan` |
| `<dataset>_directed_nsd_<variant>` | Directed NSD; also sets `--directed` | `cora_directed_nsd_general` |

The five Pei et al. datasets (`texas`, `wisconsin`, `cornell`, `chameleon`,
`squirrel`) use the exact Bodnar et al. hyperparameters for their bare preset.

List the presets available for a dataset:

```bash
python -c "from exp.registries import preset_registry as r; print(sorted(k for k in r.list_keys() if k.startswith('cora')))"
```

Passing an unknown preset also prints the full list of known keys.

### Overriding fields

Flags mirror the nested dataclasses in `src/exp/config.py`: `--<group>.<field>`.
Choices are lower-case; `--model.type` and `--model.pooling` also accept upper-case enum names.

```bash
sheaf run --preset texas --model.variant orthogonal --model.stalk-dim 5
sheaf run --preset cora --optim.lr 0.005 --optim.epochs 500 --cv.folds 5 --hardware.cuda 1
sheaf run --preset cora_directed_nsd_general          # trains on the directed graph
sheaf run --preset cora --directed                    # keep edges directed for any model
```

| Group | Flags (abridged) |
|---|---|
| `--dataset.*` | `name`, `root` (default `exp/data`); top-level `--directed` |
| `--model.*` | `type`, `variant`, `stalk-dim`, `hidden-dim`, `num-layers`, `alpha`, `learn-alpha`, `rank`, `orth-strategy`, plus per-family knobs (`bunn-*`, `nlsd-*`, `kl-*`, `gnn-*`, ...) |
| `--reg.*` | `input-dropout`, `dropout` |
| `--optim.*` | `lr`, `weight-decay`, `sheaf-decay`, `epochs`, `early-stopping`, `stop-strategy {loss,acc}` |
| `--cv.*` | `folds`, `seed`, `min-acc` |
| `--hardware.*` | `cuda`, `num-workers`, `pin-memory` |
| `--wandb.*` | `enabled`, `entity`, `project` |

### Heterogeneous and graph-level tasks

HGB heterogeneous datasets and TU graph-classification datasets have their own
runners (they need different DataModules). They accept the same `--preset` and
override flags:

```bash
# HGB node classification (imdb, dblp, acm, freebase); het_* model types
python -m exp.het_run --preset dblp --model.type het_nsd
python -m exp.het_run --preset dblp_het_nlsd_general
python -m exp.het_run --preset dblp_het_dnsd_general
python -m exp.het_run --preset dblp_het_bunn

# TU graph classification (mutag, proteins, enzymes, nci1, nci109, ptcmr, dd,
# collab, imdb_b, imdb_m, reddit_b, reddit_5k); pooling via --model.pooling
python -m exp.gc_run --preset mutag
```

## Models

Every family follows the same `encoder -> L sheaf conv layers -> decoder`
structure; node `v` carries a stalk `X_v in R^{d x f}` (`--model.stalk-dim d`,
`--model.hidden-dim f`) and each edge a learned restriction map. Families differ
in how the maps are parameterised and how messages are combined. All 30 values
of `--model.type` (shown as the enum names; the flag accepts either case):

| `--model.type` | Family | Variants (`--model.variant`) | Where |
|---|---|---|---|
| `NSD` | Neural Sheaf Diffusion (Bodnar et al., 2022) | `diagonal`, `general`, `orthogonal`, `low_rank`, `general_attention`, `orthogonal_attention` | [`nsd/`](src/sheaf_mpnn/nsd/README.md) |
| `POLY_NSD` | NSD with a Chebyshev polynomial filter | as NSD | `nsd/poly_nsd_*.py` |
| `CONN_NSD` | NSD with connection-Laplacian maps from local PCA (no learned maps) | none | `nsd/conn_nsd_model.py` |
| `DNSD` | Deep NSD (Bourgerie et al., 2026): tanh/gated layers with post-norm | as NSD | `dnsd/` |
| `NSP` | Neural Sheaf Propagation: second-order (wave) recurrence | as NSD | [`nsp/`](src/sheaf_mpnn/nsp/README.md) |
| `SHEAF_ADJ` | Adjacency-normalised NSD | as NSD | `sheaf_adj/` |
| `SHEAF_ATTN` | Sheaf Attention Networks | `sheafan`, `res_sheafan` | `sheaf_attn/` |
| `BSNN` | Bayesian Sheaf NN (Gillespie et al., 2025): variational maps, ELBO loss | `diagonal`, `general`, `orthogonal`, `low_rank` | [`bsnn/`](src/sheaf_mpnn/bsnn/README.md) |
| `NLSD` | Nonlinear Sheaf Diffusion (Zaghen et al., 2024) | `diagonal`, `orthogonal`, `general`, `low_rank` | `nlsd/` |
| `BUNN` | Bundle Neural Networks (Bamberger et al., 2025): heat-kernel diffusion on a flat bundle | none (`bunn-*` flags) | `bunn/` |
| `DIRECTED_NSD` | Directed sheaf diffusion (DSNN-style); use with `--directed` | as NSD | `directed_nsd/` |
| `CSNN` | Cooperative Sheaf NN with a GNN map learner | `conformal`, `non_conformal` | [`csnn/`](src/sheaf_mpnn/csnn/README.md) |
| `ID_SHEAF`, `JOINT_DIFF`, `ROT_INV` | Identity-sheaf, joint-diffusion and rotation-invariant sheaves (Hernandez Caralt et al., 2024) | none | `id_sheaf/`, `jointdiff/` |
| `HET_NSD`, `HET_NSP`, `HET_SHEAF_ATTN`, `POLY_HET_NSD`, `HET_NLSD`, `HET_BSNN`, `HET_CSNN`, `HET_DIRECTED_NSD`, `HET_ID_SHEAF`, `HET_JOINT_DIFF`, `HET_ROT_INV`, `HET_SHEAF_ADJ`, `HET_DNSD`, `HET_CONN_NSD`, `HET_BUNN` | Heterogeneous mirrors: maps conditioned on node/edge types (`HET_ID_SHEAF` and `HET_CONN_NSD` have fixed or feature-derived maps and ignore the types; `HET_BUNN` feeds the types to its per-node bundle learner, edge types only with `--model.bunn-phi sum_gnn`, and keeps the diffusion type-blind) | as the homogeneous family | [`src/hetsheaf_mpnn/`](src/hetsheaf_mpnn/) via `exp.het_run` |

Restriction-map variants shared by the diffusion families:

| Variant | Map per edge | Parameters per edge | Notes |
|---|---|---|---|
| `diagonal` | diagonal `d x d` | `d` | Cheapest |
| `general` | dense `d x d` | `d^2` | Most expressive |
| `orthogonal` | `O(d)` matrix | `d(d-1)/2` | `--model.orth-strategy {cayley,fasth,fasthpp,householder}` |
| `low_rank` | rank-`r` factor | `2dr` | `--model.rank r` |
| `general_attention`, `orthogonal_attention` | as above, attention-initialised | | NSD-style families only |

## Datasets

Node-classification datasets available to `sheaf run` (`--dataset.name`).
Edges are counted once per undirected pair after symmetrisation, which is the
default; `--directed` keeps the original direction.

| Name | Nodes | Edges | Features | Classes | Metric | Folds | Source |
|---|---:|---:|---:|---:|:---:|:---:|---|
| `cora` | 2,708 | 5,278 | 1,433 | 7 | Acc | 10 | directed citation (DSNN) + Geom-GCN splits |
| `citeseer` | 3,312 | 4,536 | 3,703 | 6 | Acc | 10 | directed citation (DSNN); generated splits |
| `pubmed` | 19,717 | 44,324 | 500 | 3 | Acc | 10 | directed citation (DSNN) + Geom-GCN splits |
| `chameleon` | 2,277 | 31,371 | 2,325 | 5 | Acc | 10 | Geom-GCN |
| `chameleon_filtered` | 890 | 8,854 | 2,325 | 5 | Acc | 10 | Platonov et al. (directed) |
| `squirrel` | 5,201 | 198,353 | 2,089 | 5 | Acc | 10 | Geom-GCN |
| `squirrel_filtered` | 2,223 | 46,998 | 2,089 | 5 | Acc | 10 | Platonov et al. (directed) |
| `cornell` | 183 | 277 | 1,703 | 5 | Acc | 10 | Geom-GCN (WebKB) |
| `texas` | 183 | 279 | 1,703 | 5 | Acc | 10 | Geom-GCN (WebKB) |
| `wisconsin` | 251 | 450 | 1,703 | 5 | Acc | 10 | Geom-GCN (WebKB) |
| `film` | 7,600 | 26,659 | 932 | 5 | Acc | 10 | Geom-GCN (Actor) |
| `amazon_ratings` | 24,492 | 93,050 | 300 | 5 | Acc | 10 | Platonov et al. (directed) |
| `minesweeper` | 10,000 | 39,402 | 7 | 2 | ROC-AUC | 10 | Platonov et al. (directed) |
| `questions` | 48,921 | 153,540 | 301 | 2 | ROC-AUC | 10 | Platonov et al. (directed) |
| `roman_empire` | 22,662 | 32,920 | 300 | 18 | Acc | 10 | Platonov et al. (directed) |
| `tolokers` * | 11,758 | 519,000 | 10 | 2 | ROC-AUC | 10 | Platonov et al. (directed) |
| `ogbn_arxiv` | 169,343 | 1,157,799 | 128 | 40 | Acc | 1 | OGB official split |
| `ogbn_products` * | 2,449,029 | 61,859,140 | 100 | 47 | Acc | 1 | OGB official split |
| `ogbn_proteins` * | 132,534 | 39,561,252 | 8 | 112 (multilabel) | ROC-AUC | 1 | OGB official split |
| `pokec` * | 1,632,803 | 30,622,564 | 65 | 2 | Acc | 5 | LINKX official splits |
| `telegram` | 245 | 8,126 | 1 | 4 | Acc | 10 | directed, via torch_geometric_signed_directed |

Rows marked * quote the counts published with the source release; all other
rows were measured with this repository's loaders. Because `cora`, `citeseer`,
`pubmed` and the Platonov graphs are the directed DSNN files, their node and
edge counts (and therefore results) are not directly comparable with papers
that used the public undirected versions.

Datasets for the other runners: HGB `imdb`, `dblp`, `acm`, `freebase`
(`exp.het_run`); TU `mutag`, `proteins`, `enzymes`, `nci1`, `nci109`, `ptcmr`,
`dd`, `collab`, `imdb_b`, `imdb_m`, `reddit_b`, `reddit_5k` (`exp.gc_run`);
link-prediction DataModules for `lastfm` and `movielens`
(`exp.datasets.link_pred`, used programmatically and in the integration tests).

The dataset loader doubles as a quick check that your data access works:

```bash
python scripts/verify_datasets.py cora texas      # loads directed + undirected, checks masks
```

## Using the library from Python

The packages are importable once the environment is active (`src/` is the
package root). A node-level model:

```python
from sheaf_mpnn.nsd import NSDModel, NSDVariant

model = NSDModel(
    in_channels=1433, out_channels=7,
    stalk_dim=4, hidden_dim=16, num_layers=2,
    variant=NSDVariant.GENERAL, alpha=1.0,
)
logits = model(x, edge_index)  # x: [N, 1433], edge_index: [2, E] -> [N, 7]
```

A single layer, for building your own architecture. Convs take the node state
twice: flat (`x_feat`, used to generate the restriction maps) and as stalks
(`x_stalk`, the signal being diffused):

```python
from torch import nn
from sheaf_mpnn.nsd import GeneralNSDConv

d, hidden_dim = 4, 16
encoder = nn.Linear(1433, d * hidden_dim)
layer = GeneralNSDConv(
    stalk_dim=d, in_channels=hidden_dim, hidden_dim=hidden_dim,
    context_dim=d * hidden_dim, alpha=1.0,
)

x_stalk = encoder(x).view(-1, d, hidden_dim)       # [N, d, hidden_dim]
x_feat = x_stalk.reshape(x_stalk.size(0), -1)      # [N, d * hidden_dim]
x_stalk = layer(x_feat, x_stalk, edge_index)       # [N, d, hidden_dim]
```

Loading a benchmark dataset the same way the runner does:

```python
from exp.data import load_dataset

data, info = load_dataset("cora", root="exp/data")   # PyG Data + metadata
print(data.num_nodes, info.num_classes, info.metric)  # 2708 7 acc
```

`sheaf_mpnn.__all__` lists every public model, conv and variant enum. For a
guided walk through the maths and the API, open
`notebooks/1_Sheaf_Neural_Networks_Introduction.ipynb` (`uv run jupyter notebook`).
`notebooks/0_GNNs_Basics.ipynb` covers the GNN background (message passing, GCN,
GraphSAGE, GAT, GIN), and notebooks 2-9 each cover one model family; see
[`notebooks/README.md`](notebooks/README.md).

## Hyperparameter sweeps

Sweeps are Optuna studies described in YAML. One is shipped for every
(dataset, family, variant) under `configs/<dataset>/<family>/<variant>.yaml`,
and each file names the preset it must be run with in its header comment.
Run from the repository root (the dataset cache path is relative):

```bash
sheaf sweep --yaml-path configs/cora/nsd/general.yaml --preset cora_nsd_general
```

> [!IMPORTANT]
> **Always pass the `--preset` documented in the YAML header.** The YAML only
> defines the search space and the Optuna settings; the dataset, model variant
> and training budget come from the preset. Without it the sweep silently falls
> back to the default config (`cora`, `general`), trains the wrong thing and
> logs a wrong identity to W&B. The only exception is a YAML that carries its
> own `dataset:` block and intends the default variant.

What a sweep does, in order:

1. Samples `n_trials` configurations (TPE sampler, median pruner). Each trial
   trains `n_seeds_per_trial` folds and reports mean - `std_weight` * std of
   the validation metric. Duplicate configurations are pruned automatically.
2. Prints a progress line per trial and a summary (history, top trials,
   parameter importances) every 10 trials.
3. Retrains the best configuration on all 10 folds and prints test metrics.
4. Writes the best hyperparameters to `<dataset>_<model>_bestconf.yaml` in
   the current directory. Curated results are kept under `results/bestconf/`.

Minimal YAML (any field of `ModelConfig`, `RegConfig` or `OptimConfig` can be
swept):

```yaml
model: nsd
search_space:
  variant:
    type: categorical
    choices: [diagonal, general, orthogonal]
  stalk_dim:
    type: int
    low: 2
    high: 8
  lr:
    type: float
    low: 0.0001
    high: 0.1
    log: true
config:
  n_trials: 100
  n_seeds_per_trial: 10          # Bodnar protocol: every trial trains all folds
  study_name: nsd-cora
  storage: sqlite:///nsd_cora.db  # optional: resumable / multi-worker study
  # wandb_project: my-project    # optional: one W&B run per trial
  # wandb_entity: my-entity
```

Before running a shipped config, check its `config:` block: they set
`storage: sqlite:///<study>.db` (a resumable database in the current directory)
and `wandb_project` / `wandb_entity` pointing at the authors' W&B team. See
[Weights & Biases](#weights--biases) to log to your own entity or skip W&B.

With `storage` set, launching the same command from several machines or GPUs
runs one distributed study; `n_trials` caps the total across all workers.
The launcher scripts under `scripts/<dataset>/` and `scripts/sweep_*.sh` show
how the shipped sweeps are batched.

## Weights & Biases

W&B logging is optional and off by default. `wandb` is installed by `uv sync`
(or the `wandb` extra); you only need an account and an API key.

### 1. Log in once per machine

Copy your API key from <https://wandb.ai/authorize>, then:

```bash
wandb login                  # paste the key; it is stored in ~/.netrc
wandb login --verify         # confirms a key is configured
```

On clusters or in CI, export the key instead of logging in interactively:

```bash
export WANDB_API_KEY=<your key>
```

If the key belongs to a team **service account** rather than to you, also set
your username, or every run is attributed to the service account:

```bash
export WANDB_USERNAME=<your wandb username>
```

### 2. Choose where runs go

An **entity** is your user or team name; a **project** groups runs inside it.
If you omit the entity, W&B uses the default entity of your key. If you omit
the project, each runner picks one from the dataset name:

| Entry point | Default project |
|---|---|
| `sheaf run` | `nsd-<dataset>` |
| `python -m exp.het_run` | `het-nsd-<dataset>` |
| `python -m exp.gc_run` | `gc-nsd-<dataset>` |
| `sheaf sweep` | none: logging only happens when the YAML sets `wandb_project` |

### 3. Single runs

Add `--wandb.enabled` to any `sheaf run`, `exp.het_run` or `exp.gc_run`
command:

```bash
sheaf run --preset cora --wandb.enabled --wandb.entity <entity> --wandb.project <project>
```

Each cross-validation fold becomes its own W&B run, named
`<model>-<variant>-d<stalk>-h<hidden>-L<layers>-fold<k>` and grouped under the
same name without the fold suffix, so the W&B "Group" view aggregates the
folds. Every run stores the full configuration as its config. It logs training
loss and validation metrics every epoch, then test metrics once at the best
checkpoint: loss, accuracy, ROC-AUC, PR-AUC, macro/micro F1, precision and
recall. The terminal prints a `View run at` link for each fold.

### 4. Sweeps

Sweeps read their W&B settings from the YAML `config:` block, not from CLI
flags:

```yaml
config:
  study_name: nsd-cora-general
  wandb_project: <project>
  wandb_entity: <entity>     # optional
```

A sweep then creates:

- one run per Optuna trial, named `trial/<n>/<random-name>` and grouped by
  `study_name`. Its config holds the sampled hyperparameters plus `model`,
  `dataset` and `transport_type` (the restriction-map variant). Its summary
  holds `mean_*` and `std_*` of every metric across the trial's folds.
- one `<study_name>_final_test` run with the best configuration's 10-fold
  test metrics.

Shipped configs point at the authors' team (`wandb_entity: adavit`,
`wandb_project: sheaf-mpnn-runs`). Edit those two keys before launching, or use
`WANDB_MODE=disabled` below. When filtering or grouping sweep runs in the W&B
table, use the **config** columns (`dataset`, `transport_type`). Pruned trials
never write a summary, so summary-based grouping shows them as `null`.

### Offline, disabled, and local files

```bash
WANDB_MODE=offline sheaf run --preset cora --wandb.enabled   # log locally, no network
wandb sync wandb/offline-run-<timestamp>-<id>                # upload later
WANDB_MODE=disabled sheaf sweep --yaml-path <file> --preset <preset>   # ignore the YAML's W&B keys
```

`wandb sync` also accepts `--entity` and `--project` to choose the destination.
Local run data is written to `./wandb/`, which is gitignored.

## Efficiency benchmark vs. the original NSD code

`sheaf benchmark` compares this repository's `diagonal` / `orthogonal` /
`general` NSD variants against
[twitter-research/neural-sheaf-diffusion](https://github.com/twitter-research/neural-sheaf-diffusion)
on wall-clock time, peak GPU memory, parameter count and inference latency under
matched architectures. It is designed for a SLURM cluster (jobs are submitted
with `submitit`).

One-time setup (an isolated Python 3.9 / PyTorch 1.11 environment for the
original code):

```bash
git submodule update --init external/neural-sheaf-diffusion
CUDA_TAG=cu113 bash scripts/benchmark/setup_original_env.sh   # CUDA_TAG=cpu without a GPU
```

Fill in the matched hyperparameters under
`configs/benchmark/bestconf/{ours,original}/<dataset>/<variant>.yaml` and the
`slurm:` block of the manifest, then:

```bash
sheaf benchmark submit  --manifest configs/benchmark/nsd_vs_original.yaml
sheaf benchmark collect --manifest configs/benchmark/nsd_vs_original.yaml
sheaf benchmark report  --manifest configs/benchmark/nsd_vs_original.yaml
```

`submit` is idempotent: re-running it only submits jobs without a result. Add
`--modes efficiency` or `--modes convergence` to work on one axis at a time.

## Development

Formatting, linting, type checking and tests run through `uv`:

```bash
uv run prek run --all-files     # ruff check/format, ty type check, hygiene hooks
uv run pytest                   # unit tests (about 4,600 tests, with coverage)
uv run pytest -m integration    # slow tests: real datasets, short training runs
uv run --only-group docs sphinx-build -W docs docs/_build/html   # documentation
```

The individual tools, if you prefer them one at a time:

```bash
uv run ty check                          # static type checking
uv run ruff check --fix --unsafe-fixes   # lint and auto-fix
uv run ruff format .                     # format
```

Install the git hooks with `uv run prek install` so the same checks run on every
commit. CI (`.github/workflows/ci.yml`) runs lint, type check and the unit
suite on every push and pull request, repeats the unit suite on an unlocked
install across Python 3.11-3.13 and Linux, macOS and Windows, and builds the
documentation; the benchmark integration tests run weekly.

## Citation

If you use this library in your research, please cite the
[paper](https://arxiv.org/abs/2611.04670) (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{borgi2026sheafmpnn,
  title         = {Sheaf-MPNN: A Library for Sheaf Neural Networks as Message Passing},
  author        = {Borgi, Alessio and Severino, Mario and Braithwaite, Luke and
                   Azzolin, Steve and Bourgerie, R{\'e}mi and Zaghen, Olga and
                   Mule, Emanuele and Ali, Riccardo and Kustermann, Leo Minh and
                   Coppola, Edoardo and Purificato, Antonio and
                   Fiorini, Stefano and Irwin, Christopher and
                   Spiga, Filippo and Silvestri, Fabrizio and
                   Bronstein, Michael and Li{\`o}, Pietro},
  year          = {2026},
  eprint        = {2611.04670},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.04670}
}
```

## Contributing

Bug reports and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md)
for the workflow and code conventions, and the
[Code of Conduct](CODE_OF_CONDUCT.md). Changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

## License

Released under the [MIT License](LICENSE). The figures embedded in the tutorial
notebooks come from the cited papers and keep their own licences; see
[`notebooks/README.md`](notebooks/README.md#figure-credits).
