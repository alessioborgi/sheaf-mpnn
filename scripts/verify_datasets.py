#!/usr/bin/env python
"""Verify that datasets load correctly, both directed and undirected.

For every dataset this loads it two ways -- ``--directed`` (the original
directed graph) and the undirected default -- and checks:

* ``x`` / ``edge_index`` / ``y`` are present with consistent shapes;
* the directed load is actually directed and the undirected load symmetric;
* the two disagree on edge count (i.e. ``--directed`` really changes the graph);
* fold 0 has non-empty, pairwise-disjoint train / val / test masks.

Usage
-----
    .venv/bin/python scripts/verify_datasets.py                 # all datasets
    .venv/bin/python scripts/verify_datasets.py cora telegram   # a subset
    .venv/bin/python scripts/verify_datasets.py --hosted        # DSNN sets only

``--hosted`` restricts to the DSNN-sourced sets (synthetic, directed Platonov
.npz, directed citation), which download from this repository's
``dataset-assets-v1`` release on first use and are cached under ``exp/data``.
The citation sets additionally fetch the canonical geom-gcn split files on
first use (cached under ``src/exp/splits``).  The geom-gcn sets and telegram
download their graphs from their own upstream sources on first use.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_geometric.utils import is_undirected  # noqa: E402

from exp.data import load_dataset  # noqa: E402
from exp.datasets.synthetic import available_synthetic  # noqa: E402
from exp.splits import apply_split  # noqa: E402

DATA_ROOT = os.path.join(ROOT, "exp", "data")

# DSNN-sourced sets, downloaded from the dataset-assets release on first use.
HOSTED = (
    available_synthetic(DATA_ROOT)
    + [
        "roman_empire",
        "amazon_ratings",
        "questions",
        "chameleon_filtered",
        "squirrel_filtered",
        "minesweeper",
        "tolokers",
    ]
    + ["cora", "citeseer", "pubmed"]
)
# Downloaded from their own upstream sources on first use.
DOWNLOAD = [
    "telegram",
    "cornell",
    "texas",
    "wisconsin",
    "chameleon",
    "squirrel",
    "film",
]


def check(name: str) -> tuple[bool, str]:
    """Return (ok, detail-line) for one dataset."""
    dd, info = load_dataset(name, root=DATA_ROOT, to_undirected=False)  # --directed
    du, _ = load_dataset(name, root=DATA_ROOT, to_undirected=True)  # default
    s = apply_split(dd, info, fold=0)

    if dd.x is None or dd.y is None or dd.edge_index is None or du.edge_index is None:
        return False, "<<< missing x/y/edge_index"
    num_nodes = dd.num_nodes
    assert num_nodes is not None

    problems = []
    if int(dd.x.size(0)) != int(num_nodes):
        problems.append("x/num_nodes mismatch")

    directed = not is_undirected(dd.edge_index)
    undirected = is_undirected(du.edge_index)
    e_dir, e_undir = dd.edge_index.size(1), du.edge_index.size(1)
    if not directed:
        problems.append("--directed graph is not directed")
    if not undirected:
        problems.append("default graph is not undirected")

    tr, va, te = (
        int(s.train_mask.sum()),
        int(s.val_mask.sum()),
        int(s.test_mask.sum()),
    )
    if min(tr, va, te) == 0:
        problems.append("empty train/val/test mask")
    if (
        (s.train_mask & s.val_mask).any()
        or (s.train_mask & s.test_mask).any()
        or (s.val_mask & s.test_mask).any()
    ):
        problems.append("overlapping masks")

    detail = (
        f"N={num_nodes:>6}  E_dir={e_dir:>8}  E_undir={e_undir:>8}  "
        f"F={info.num_features:>5}  C={info.num_classes:>2}  {info.metric:<7} "
        f"folds={info.num_splits:>2}  tr/va/te={tr}/{va}/{te}"
    )
    if problems:
        return False, detail + "  <<< " + "; ".join(problems)
    return True, detail


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    hosted_only = "--hosted" in sys.argv

    if args:
        names = args
    else:
        names = HOSTED if hosted_only else HOSTED + DOWNLOAD

    print(f"Verifying {len(names)} dataset(s) from {DATA_ROOT}\n")
    n_ok = 0
    for name in names:
        try:
            ok, detail = check(name)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"ERROR {type(exc).__name__}: {exc}"
        n_ok += ok
        print(f"[{'OK  ' if ok else 'FAIL'}] {name:34s} {detail}")

    print(f"\n{n_ok}/{len(names)} datasets loaded correctly.")
    return 0 if n_ok == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
