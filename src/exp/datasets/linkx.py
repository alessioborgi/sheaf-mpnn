# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""LINKX large-scale non-homophilous datasets (Lim et al., 2021).

Currently only ``pokec`` (1.6M profiles of the Slovak social network, gender
labels, accuracy).  The raw ``pokec.mat`` and the authors' fixed
``pokec-splits.npy`` (five 50/25/25 splits drawn over labeled nodes) are
fetched from this repository's dataset-assets release first, falling back to
the LINKX authors' Google Drive files (CUAI/Non-Homophily-Large-Scale).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch_geometric.data import Data

LINKX_DATASETS: frozenset[str] = frozenset({"pokec"})

# Google Drive file ids from the LINKX repository's data_utils.py.
_DRIVE_IDS: dict[str, str] = {
    "pokec.mat": "1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y",
    "pokec-splits.npy": "1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_",
}


def _ensure_file(filename: str, root: str) -> str:
    """Fetch *filename* into ``root/raw``, trying the assets release first."""
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    dst = os.path.join(raw_dir, filename)
    if os.path.isfile(dst):
        return dst
    try:
        from exp.datasets.assets import ensure_asset  # noqa: PLC0415

        return ensure_asset(filename, dst)
    except Exception:
        pass
    import gdown  # noqa: PLC0415

    print(f"Downloading {filename} from the LINKX Google Drive release")
    out = gdown.download(id=_DRIVE_IDS[filename], output=dst, quiet=False)
    if out is None:
        raise RuntimeError(
            f"Could not download {filename!r}: neither the dataset-assets "
            "release nor the LINKX Google Drive file was reachable."
        )
    return dst


def load_linkx(name: str, root: str) -> Data:
    """Load a LINKX dataset with the official splits as ``(N, folds)`` masks.

    Labels use -1 for unlabeled nodes; the official splits index labeled
    nodes only, so every mask column excludes the unlabeled ones.
    """
    assert name in LINKX_DATASETS, f"Unknown LINKX dataset: {name!r}"
    processed = os.path.join(root, "processed", f"{name}.pt")
    if os.path.isfile(processed):
        return torch.load(processed, weights_only=False)

    import scipy.io  # noqa: PLC0415

    mat = scipy.io.loadmat(_ensure_file(f"{name}.mat", root))
    edge_index = torch.as_tensor(
        np.ascontiguousarray(mat["edge_index"]), dtype=torch.long
    )
    x = torch.as_tensor(mat["node_feat"]).float()
    y = torch.as_tensor(np.asarray(mat["label"]).flatten(), dtype=torch.long)

    splits = np.load(_ensure_file(f"{name}-splits.npy", root), allow_pickle=True)
    n = int(x.size(0))
    num_splits = len(splits)
    masks = {
        key: torch.zeros(n, num_splits, dtype=torch.bool)
        for key in ("train", "valid", "test")
    }
    for fold, split in enumerate(splits):
        for key, mask in masks.items():
            idx = torch.as_tensor(np.asarray(split[key]).flatten(), dtype=torch.long)
            mask[idx, fold] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=masks["train"],
        val_mask=masks["valid"],
        test_mask=masks["test"],
    )
    os.makedirs(os.path.dirname(processed), exist_ok=True)
    torch.save(data, processed)
    return data
