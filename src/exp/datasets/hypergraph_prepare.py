"""Prepare pinned DSHN data as numeric artifacts; never use graph presets.

The reference tree must come from the pinned GitHub commit. Pickles are read
only after their Git blob SHA has been verified against that tree.
"""

import argparse
import hashlib
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import torch

from exp.datasets.hypergraph import (
    HYPERGRAPHS,
    REFERENCE_COMMIT,
    load_artifact,
)


def grouped_edges(pairs):
    groups = {}
    for source, target in pairs:
        source, target = int(source), int(target)
        groups.setdefault(source, set())
        if source != target:
            groups[source].add(target)
    return [([source], sorted(targets)) for source, targets in groups.items()]


def incidence(groups):
    triples = []
    for edge, (tail, head) in enumerate(groups):
        if set(tail) & set(head):
            raise ValueError(
                "Overlapping tail/head requires an explicit preprocessing policy"
            )
        triples.extend((int(node), edge, -1) for node in sorted(set(tail)))
        triples.extend((int(node), edge, 1) for node in sorted(set(head)))
    values = np.asarray(sorted(triples), dtype=np.int64)
    return values[:, :2].T.copy(), values[:, 2].copy()


def prepare(name, root, tree_path, download=False):
    if name == "cora-hypergraph":
        from exp.datasets.hypergraph_cora import prepare as prepare_cora

        return prepare_cora(root / "raw" / "cocitation" / "cora", root, download)
    spec = HYPERGRAPHS[name]
    tree = json.loads(tree_path.read_text(encoding="utf-8-sig"))
    if tree["sha"] != REFERENCE_COMMIT:
        raise ValueError("Reference tree commit mismatch")
    entries = {entry["path"]: entry for entry in tree["tree"]}
    sources = {}

    def raw(filename):
        relative = f"data/original_data/{spec.source_directory}/{filename}"
        entry = entries[relative]
        path = root / "raw" / spec.source_directory / filename
        if not path.exists() and download:
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/EmaMule/DirectionalSheafHypergraphs/{REFERENCE_COMMIT}/{relative}"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            actual = hashlib.sha1(
                f"blob {len(content)}\0".encode() + content
            ).hexdigest()
            if actual != entry["sha"]:
                raise ValueError(f"Git blob checksum mismatch: {relative}")
            path.write_bytes(content)
        content = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if actual != entry["sha"]:
            raise ValueError(f"Git blob checksum mismatch: {relative}")
        sources[relative] = hashlib.sha256(content).hexdigest()
        return path

    def unpickle(filename):
        return pickle.loads(raw(filename).read_bytes())

    masks = None
    if spec.source_directory.startswith("additional/"):
        with np.load(
            raw(f"{spec.reference_name}_filtered.npz"), allow_pickle=False
        ) as a:
            x, y = a["node_features"], a["node_labels"]
            pairs = a["edges"]
            if pairs.shape[1] != 2:
                pairs = pairs.T
            groups = grouped_edges(pairs)
            masks = [a[f"{key}_masks"].copy() for key in ("train", "val", "test")]
    elif spec.source_directory.startswith("cocitation/"):
        rows = [
            line.split()
            for line in raw(f"{spec.reference_name}.content").read_text().splitlines()
        ]
        ids = {row[0]: i for i, row in enumerate(rows)}
        pairs = [
            line.split()
            for line in raw(f"{spec.reference_name}.cites").read_text().splitlines()
        ]
        groups = grouped_edges(
            (ids[a], ids[b]) for a, b in pairs if a in ids and b in ids
        )
        x = unpickle("features.pickle")
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        y = np.asarray(unpickle("labels.pickle"))
    elif spec.source_directory.startswith("mail/"):
        groups = grouped_edges(unpickle("weight_graph.pickle"))
        y = np.asarray(unpickle("weight_label.pickle"))
        x = None
    elif spec.task == "hyperedge":
        x = np.asarray(unpickle("features_more_data.pickle"))
        groups, labels = [], []
        for edge in unpickle("hypergraph_more_class_more_data.pickle").values():
            tail = sorted({int(n) for n in edge["reagente"] if 0 <= n < len(x)})
            head = sorted({int(n) for n in edge["prodotto"] if 0 <= n < len(x)})
            if not tail or not head or set(tail) & set(head):
                continue
            groups.append((tail, head))
            labels.append(edge["class"])
        y = np.asarray(labels)
        if y.min() == 1:
            y = y - 1
    else:
        from torch_geometric_signed_directed.data import load_directed_real_data

        dataset = "telegram" if spec.reference_name == "telegram" else "webkb"
        data = load_directed_real_data(
            dataset=dataset, root=str(root / "raw" / "other"), name=spec.reference_name
        )
        x, y = data.x.numpy(), data.y.numpy()
        groups = grouped_edges(data.edge_index.T.tolist())
        masks = [
            getattr(data, f"{key}_mask").numpy().T for key in ("train", "val", "test")
        ]
        sources["external_loader"] = "torch_geometric_signed_directed"
    index, roles = incidence(groups)
    if x is None:
        x = np.bincount(index[0], minlength=len(y)).reshape(-1, 1)
    # Cora keeps its separately audited legacy preparation path.
    path = root / f"{name}.npz"
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dataset_id="dshn-hypergraph-v1",
        dataset_name=name,
        task=spec.task,
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        hyperedge_index=index,
        incidence_roles=roles,
        num_hyperedges=len(groups),
    )
    load_artifact(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    splits = {"dataset_sha256": checksum, "num_splits": 10}
    if masks is None:
        generator = torch.Generator().manual_seed(0)
        for fold in range(10):
            order = torch.randperm(len(y), generator=generator).numpy()
            a, b = int(len(y) * 0.5), int(len(y) * 0.25)
            for key, values in zip(
                ("train", "val", "test"),
                (order[:a], order[a : a + b], order[a + b :]),
                strict=True,
            ):
                splits[f"{key}_{fold}"] = values
    else:
        splits["num_splits"] = masks[0].shape[0]
        for fold in range(masks[0].shape[0]):
            for key, mask in zip(("train", "val", "test"), masks, strict=True):
                splits[f"{key}_{fold}"] = np.flatnonzero(mask[fold]).astype(np.int64)
    split_path = path.with_suffix(".splits.npz")
    np.savez_compressed(split_path, **splits)
    metadata = dict(
        dataset_id="dshn-hypergraph-v1",
        dataset_name=name,
        task=spec.task,
        reference_commit=REFERENCE_COMMIT,
        dataset_sha256=checksum,
        splits_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        sources=sources,
    )
    path.with_suffix(".provenance.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    from exp.datasets.hypergraph import load_split_bank

    load_split_bank(split_path, len(y), int(splits["num_splits"]), checksum)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=HYPERGRAPHS, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/hypergraphs"))
    parser.add_argument(
        "--tree",
        type=Path,
        default=Path(__file__).with_name("hypergraph_sources.json"),
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(prepare(args.dataset, args.root, args.tree, args.download))
