# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Lio

"""Build the ICLR 2026 DSHN Cora topology, projected to undirected q=0.

The official loader builds edges from cora.cites, NOT hypergraph.pickle.
Content rows agree exactly with the official feature/label pickle ordering.
Only checksum-pinned text files are parsed; no pickle is deserialized here.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

REFERENCE_COMMIT = "d94f81bc7d86bd8d41e416083063f3446c1ad796"
DATASET_ID = "dshn-cora-q0-v1"
SOURCE = "https://github.com/EmaMule/DirectionalSheafHypergraphs"
RAW_BASE = (
    "https://raw.githubusercontent.com/EmaMule/DirectionalSheafHypergraphs/"
    f"{REFERENCE_COMMIT}/data/original_data/cocitation/cora/"
)
RAW_SHA256 = {
    "cora.content": "b8ebed0fd45e53995abafd1b70cf6a9fcc8f86abc5826d69b97fe6ca46e9c8ee",
    "cora.cites": "ec1a372391b7f0f60a6aff0084e8abd8f19f0faa7e1f2441a41c492042d5945e",
}
# Audited against the official labels.pickle, not alphabetical class order.
LABEL_IDS = {
    "Genetic_Algorithms": 0,
    "Theory": 1,
    "Rule_Learning": 2,
    "Neural_Networks": 3,
    "Case_Based": 4,
    "Probabilistic_Methods": 5,
    "Reinforcement_Learning": 6,
}


def build_incidence(
    node_ids: list[str], citation_lines: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the reference tail/head convention and first-seen edge numbering.

    The first citation column is the tail key in the reference loader. Each
    hyperedge contains that tail PLUS all second-column nodes linked to it.
    Roles (-1 tail, +1 head) are retained for audit; HSNN uses their q=0 support.
    """
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("duplicate paper IDs")
    lookup = {key: i for i, key in enumerate(node_ids)}
    heads: dict[int, set[int]] = {}
    for line in citation_lines:
        if not line.strip():
            continue
        first, second = line.split()
        if first not in lookup or second not in lookup:
            continue  # The official loader ignores unknown endpoints.
        tail, head = lookup[first], lookup[second]
        heads.setdefault(tail, set())
        if tail != head:
            heads[tail].add(head)
    triples = []
    for edge, (tail, members) in enumerate(heads.items()):
        triples.append((tail, edge, -1))
        triples.extend((head, edge, 1) for head in sorted(members))
    if not triples:
        raise ValueError("empty citation hypergraph")
    triples.sort()  # Canonical incidence ordering; edge IDs remain first-seen.
    values = np.asarray(triples, dtype=np.int64)
    return (
        values[:, :2].T.copy(),
        values[:, 2].astype(np.int8),
        np.array(list(heads), dtype=np.int64),
    )


def make_splits(num_nodes: int, runs: int, seed: int) -> dict[str, np.ndarray]:
    """Replay the reference's sequential CPU randperm stream before model setup."""
    if num_nodes < 4 or runs < 1 or seed < 0:
        raise ValueError("need >=4 nodes, positive runs and a nonnegative seed")
    generator = torch.Generator().manual_seed(seed)
    permutations = torch.stack(
        [torch.randperm(num_nodes, generator=generator) for _ in range(runs)]
    ).numpy()
    train_end = int(num_nodes * 0.5)
    val_end = train_end + int(num_nodes * 0.25)
    return {
        "train": permutations[:, :train_end],
        "val": permutations[:, train_end:val_end],
        "test": permutations[:, val_end:],
    }


def prepare(raw_dir: Path, output_dir: Path, download: bool = False) -> Path:
    """Write an audited data file, shared split bank and provenance sidecar."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in RAW_SHA256.items():
        path = raw_dir / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; use --download to fetch it")
            with urllib.request.urlopen(RAW_BASE + name, timeout=60) as response:
                blob = response.read()
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError(f"Download checksum mismatch: {name}")
            path.write_bytes(blob)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Raw file checksum mismatch: {name}")

    content = np.genfromtxt(raw_dir / "cora.content", dtype=str)
    x = content[:, 1:-1].astype(np.float32)
    y = np.array([LABEL_IDS[label] for label in content[:, -1]], dtype=np.int64)
    node_ids = content[:, 0]
    index, roles, tails = build_incidence(
        node_ids.tolist(), (raw_dir / "cora.cites").read_text().splitlines()
    )
    sizes = np.bincount(index[1], minlength=len(tails))
    stats = {
        "nodes": len(y),
        "features": x.shape[1],
        "classes": len(np.unique(y)),
        "hyperedges": len(tails),
        "incidences": index.shape[1],
        "max_edge_size": int(sizes.max()),
    }
    expected = dict(
        nodes=2708,
        features=1433,
        classes=7,
        hyperedges=1565,
        incidences=6994,
        max_edge_size=167,
    )
    if stats != expected:
        raise ValueError(f"Cora reference statistics mismatch: {stats}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "cora-dshn-q0.npz"
    sidecar = target.with_suffix(".provenance.json")
    split_path = target.with_suffix(".splits.npz")
    if any(path.exists() for path in (target, sidecar, split_path)):
        raise FileExistsError(
            "Use a fresh output directory to preserve prior artifacts"
        )
    np.savez_compressed(
        target,
        x=x,
        y=y,
        hyperedge_index=index,
        num_hyperedges=np.int64(len(tails)),
        incidence_roles=roles,
        hyperedge_tails=tails,
        node_ids=node_ids,
        dataset_id=np.array(DATASET_ID),
        reference_commit=np.array(REFERENCE_COMMIT),
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    splits = make_splits(len(y), runs=10, seed=0)
    np.savez_compressed(
        split_path,
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        dataset_sha256=np.array(digest),
        split_seed=np.int64(0),
    )
    provenance = {
        "dataset_id": DATASET_ID,
        "reference_commit": REFERENCE_COMMIT,
        "source": SOURCE,
        "raw_base": RAW_BASE,
        "raw_sha256": RAW_SHA256,
        "dataset_sha256": digest,
        "splits_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "statistics": stats,
        "mean_hyperedge_size": float(sizes.mean()),
        "mean_node_degree": float(index.shape[1] / len(y)),
        "label_ids": LABEL_IDS,
        "reference_loader": (
            "dataset_loaders/data_utils_direction.py:"
            "load_citation_dataset_direction, processing"
        ),
        "topology": "tail plus heads; q=0 support; no added singleton edges",
        "features": "raw bag-of-words; official content row order; no normalization",
        "splits": "10 sequential CPU torch.randperm draws from seed 0; 50/25/25",
        "split_torch_version": str(torch.__version__),
        "scope": "paper topology/split protocol; not full DSHN reproduction",
    }
    sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    return target


def main() -> None:
    """Prepare once; share the resulting files unchanged across all models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/cora-raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cora-dshn"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir, args.download)


if __name__ == "__main__":
    main()
