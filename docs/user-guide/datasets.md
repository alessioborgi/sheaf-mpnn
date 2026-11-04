# Datasets

All datasets are downloaded on first use and cached under `exp/data/`
(relative to the working directory). No credentials are needed.

## Sources

| Datasets | Source |
|---|---|
| `cora`, `citeseer`, `pubmed` (directed citation graphs) | Repository release `dataset-assets-v1` |
| `amazon_ratings`, `chameleon_filtered`, `minesweeper`, `questions`, `roman_empire`, `squirrel_filtered`, `tolokers` (directed Platonov graphs) | Repository release `dataset-assets-v1` |
| synthetic `dataset_nodes500_*` directed graphs | Repository release `dataset-assets-v1` |
| `chameleon`, `squirrel`, `cornell`, `texas`, `wisconsin`, `film` | Repository mirror first, PyG download as fallback |
| `telegram` | Repository mirror first, upstream download as fallback |
| `ogbn_arxiv`, `ogbn_products`, `ogbn_proteins` | OGB |
| `pokec` | LINKX (Google Drive) |
| HGB `imdb`, `dblp`, `acm`, `freebase`; TU graph-classification sets; `lastfm`, `movielens` | Upstream sources |

The directed citation and Platonov graphs are the DSNN variants, whose edge
lists, labels and masks differ from the public undirected originals, which is
why they are served from the repository's own release. Set
`SHEAF_DATASET_REPO=<owner>/<name>` (and optionally `SHEAF_DATASET_TAG`) to
download them from a fork or mirror. If `GITHUB_TOKEN` or `GH_TOKEN` is set,
the authenticated GitHub API is used instead of the public download URL.

## Node classification

Datasets available to `sheaf run` (`--dataset.name`). Edges are counted once per
undirected pair after symmetrisation, which is the default; `--directed` keeps
the original direction.

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

## Other tasks

- Heterogeneous node classification (`exp.het_run`): HGB `imdb`, `dblp`, `acm`,
  `freebase`.
- Graph classification (`exp.gc_run`): TU `mutag`, `proteins`, `enzymes`,
  `nci1`, `nci109`, `ptcmr`, `dd`, `collab`, `imdb_b`, `imdb_m`, `reddit_b`,
  `reddit_5k`.
- Link prediction: DataModules for `lastfm` and `movielens` in
  {py:mod}`exp.datasets.link_pred`.
