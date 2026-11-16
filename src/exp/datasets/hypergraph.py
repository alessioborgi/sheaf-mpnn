"""Names and artifact locations for the directional hypergraph benchmarks.

Graph and hypergraph names deliberately occupy separate namespaces. Dataset
registration is not a claim that raw data have been downloaded or validated.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REFERENCE_COMMIT = "d94f81bc7d86bd8d41e416083063f3446c1ad796"


@dataclass(frozen=True)
class HypergraphSpec:
    reference_name: str
    source_directory: str
    task: str = "node"
    split: str = "predefined"


HYPERGRAPHS = {
    "roman-empire-hypergraph": HypergraphSpec("roman", "additional/roman"),
    "squirrel-hypergraph": HypergraphSpec("squirrel", "additional/squirrel"),
    "email-eu-hypergraph": HypergraphSpec("EU", "mail/EU", split="random"),
    "telegram-hypergraph": HypergraphSpec("telegram", "other"),
    "chameleon-hypergraph": HypergraphSpec("chameleon", "additional/chameleon"),
    "email-enron-hypergraph": HypergraphSpec("enroll", "mail/enroll", split="random"),
    "cornell-hypergraph": HypergraphSpec("cornell", "other"),
    "wisconsin-hypergraph": HypergraphSpec("wisconsin", "other"),
    "amazon-ratings-hypergraph": HypergraphSpec("amazon", "additional/amazon"),
    "texas-hypergraph": HypergraphSpec("texas", "other"),
    "citeseer-hypergraph": HypergraphSpec(
        "citeseer", "cocitation/citeseer", split="random"
    ),
    "cora-hypergraph": HypergraphSpec("cora", "cocitation/cora", split="random"),
    "molecular-1-hypergraph": HypergraphSpec(
        "Dataset-1", "protein/Dataset-1", "hyperedge", "random"
    ),
    "molecular-2-hypergraph": HypergraphSpec(
        "Dataset-2", "protein/Dataset-2", "hyperedge", "random"
    ),
}


def canonical_name(name: str) -> str:
    return "cora-hypergraph" if name == "cora_hypergraph" else name


def is_hypergraph(name: str) -> bool:
    return canonical_name(name) in HYPERGRAPHS


def artifact_path(name: str, root: str) -> Path:
    name = canonical_name(name)
    if name not in HYPERGRAPHS:
        raise ValueError(f"Unknown hypergraph dataset: {name}")
    filename = "cora-dshn-q0.npz" if name == "cora-hypergraph" else f"{name}.npz"
    return Path(root) / filename


def ensure_hypergraph(name: str, root: str) -> Path:
    """Download missing data on first use; preserve and verify cached artifacts."""
    path = artifact_path(name, root)
    artifacts = (
        path,
        path.with_suffix(".splits.npz"),
        path.with_suffix(".provenance.json"),
    )
    present = [file.exists() for file in artifacts]
    if all(present):
        validate_reference_artifact(name, path)
        return path
    if any(present):
        raise FileNotFoundError(
            f"Incomplete hypergraph cache at {path}; use a fresh dataset root"
        )
    from exp.datasets.hypergraph_prepare import prepare

    result = prepare(
        canonical_name(name),
        Path(root),
        Path(__file__).with_name("hypergraph_sources.json"),
        download=True,
    )
    validate_reference_artifact(name, result)
    return result


def artifact_fingerprint(path: Path) -> dict[str, str]:
    """Hash actual numeric inputs and target indices, independently of ZIP metadata."""
    result = {}
    for filename, keys in (
        (path, ("x", "y", "hyperedge_index", "incidence_roles", "num_hyperedges")),
        (path.with_suffix(".splits.npz"), None),
    ):
        with np.load(filename, allow_pickle=False) as data:
            selected = (
                keys
                if keys is not None
                else sorted(
                    k
                    for k in data.files
                    if k not in ("dataset_sha256", "split_seed", "num_splits")
                )
            )
            for key in selected:
                array = np.ascontiguousarray(data[key])
                header = json.dumps([array.dtype.str, array.shape]).encode()
                result[f"{filename.suffixes[-2:]}/{key}"] = hashlib.sha256(
                    header + array.tobytes()
                ).hexdigest()
    return result


def validate_reference_artifact(name: str, path: Path) -> None:
    """Reject data or split changes against the audited upstream reference."""
    manifest = json.loads(
        Path(__file__)
        .with_name("hypergraph_fingerprints.json")
        .read_text(encoding="utf-8")
    )
    expected = manifest[canonical_name(name)]
    actual = artifact_fingerprint(path)
    if actual != expected:
        changed = sorted(
            k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)
        )
        raise ValueError(
            f"Hypergraph data/splits differ from the pinned reference: "
            f"{name}: {changed}"
        )


def load_artifact(path: Path):
    """Validate a versioned numeric artifact, including its prediction domain."""
    with np.load(path, allow_pickle=False) as archive:
        name = str(archive["dataset_name"].item())
        if (
            name not in HYPERGRAPHS
            or archive["dataset_id"].item() != "dshn-hypergraph-v1"
        ):
            raise ValueError("Unknown hypergraph artifact schema or dataset")
        task = str(archive["task"].item())
        if task != HYPERGRAPHS[name].task:
            raise ValueError("Prediction domain differs from the dataset manifest")
        x, y, index, roles = (
            torch.from_numpy(archive[k].copy())
            for k in ("x", "y", "hyperedge_index", "incidence_roles")
        )
        edges = int(archive["num_hyperedges"].item())
    from sheaf_mpnn.directed_hypergraph_sheaf_conv import (
        BaseDirectedHypergraphSheafConv,
    )

    BaseDirectedHypergraphSheafConv._validate_incidence(
        index, x.size(0), edges, x.device
    )
    BaseDirectedHypergraphSheafConv._validate_roles(roles, index)
    if x.ndim != 2 or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError("Expected finite float32 node features")
    target_count = edges if task == "hyperedge" else x.size(0)
    if y.dtype != torch.long or y.shape != (target_count,) or y.numel() == 0:
        raise ValueError("Labels must match the prediction domain")
    if not torch.equal(y.unique(), torch.arange(int(y.max()) + 1)):
        raise ValueError("Labels must be contiguous nonnegative integers")
    return x, y, index, edges


def load_split_bank(path: Path, count: int, runs: int, checksum: str):
    """Load variable-size predefined splits without imposing Cora proportions."""
    with np.load(path, allow_pickle=False) as archive:
        if archive["dataset_sha256"].item() != checksum:
            raise ValueError("Split bank belongs to another artifact")
        total = int(archive["num_splits"].item())
        if not 1 <= runs <= total:
            raise ValueError("Requested folds exceed frozen split bank")
        result = []
        for fold in range(total):
            parts = [
                archive[f"{key}_{fold}"].copy() for key in ("train", "val", "test")
            ]
            if any(p.dtype != np.int64 or p.ndim != 1 or not p.size for p in parts):
                raise ValueError("Invalid split indices")
            joined = np.concatenate(parts)
            if not np.array_equal(np.sort(joined), np.arange(count)):
                raise ValueError("Split overlap, omission or out-of-range target")
            result.append(tuple(torch.from_numpy(p) for p in parts))
    return result[:runs]
