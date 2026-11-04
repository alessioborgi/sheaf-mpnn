# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Utility transforms for dataset preprocessing and splitting."""

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class RemoveSelfLoops(BaseTransform):
    """Removes self-loops from graph data while maintaining attribute alignment.

    Unlike the standard PyG `RemoveSelfLoops`, this version dynamically identifies
    and filters ALL edge-level attributes (including `edge_weight`, `edge_attr`, etc.)
    to ensure they remain synchronized with the pruned `edge_index`.

    It supports both homogeneous :class:`~torch_geometric.data.Data` and
    heterogeneous :class:`~torch_geometric.data.HeteroData` objects.
    """

    def forward(
        self,
        data: Data | HeteroData,
    ) -> Data | HeteroData:
        for store in data.edge_stores:
            # Skip bipartite edge sets (no self-loops possible) or missing edges
            if store.is_bipartite() or "edge_index" not in store:
                continue

            edge_index = store.edge_index
            mask = edge_index[0] != edge_index[1]

            if not mask.all():
                # Filter all edge-level attributes that match the original edge count
                for key, value in store.items():
                    if key == "edge_index":
                        continue

                    # Check if the attribute is actually an edge attribute
                    if isinstance(data, Data):
                        is_edge_attr = data.is_edge_attr(key)
                    else:
                        is_edge_attr = store.is_edge_attr(key)

                    if (
                        is_edge_attr
                        and torch.is_tensor(value)
                        and value.size(0) == mask.size(0)
                    ):
                        store[key] = value[mask]

                # Finally update the edge index
                store.edge_index = edge_index[:, mask]

        return data


class SubsetTrainSplit(BaseTransform):
    """Splits an existing training mask into training and validation subsets.

    This transform is useful for datasets that only provide a train/test split
    or when a custom validation ratio is needed from the training data.

    Args:
        val_ratio: Fraction of training nodes to move to the validation set.
        seed: Random seed for reproducibility.
        use_index: If True, stores results in `train_idx`/`val_idx` instead of masks.
    """

    def __init__(
        self, val_ratio: float = 0.1, seed: int | None = None, use_index: bool = False
    ):
        self.val_ratio = val_ratio
        self.seed = seed
        self.use_index = use_index

    def forward(self, data: Data | HeteroData) -> Data | HeteroData:
        # Handle both HeteroData and regular Data objects
        node_types = data.node_types if isinstance(data, HeteroData) else [None]

        g = torch.Generator()
        if self.seed is not None:
            g.manual_seed(self.seed)

        # Flag to ensure we actually found a mask to split
        found_train_mask = False

        for node_type in node_types:
            # Access the store (either data[node_type] or the data object itself)
            store = data[node_type] if node_type is not None else data

            if hasattr(store, "train_mask"):
                found_train_mask = True
                # 1. Convert mask to indices if necessary
                if store.train_mask.dtype == torch.bool:
                    train_indices = torch.where(store.train_mask)[0]
                else:
                    train_indices = store.train_mask

                # 2. Shuffle and Split
                perm = torch.randperm(train_indices.size(0), generator=g)
                train_indices = train_indices[perm]

                val_size = int(train_indices.size(0) * self.val_ratio)
                val_idx = train_indices[:val_size]
                train_idx = train_indices[val_size:]

                # 3. Save as indices or masks
                if self.use_index:
                    store.train_idx = train_idx
                    store.val_idx = val_idx
                else:
                    assert store.num_nodes is not None
                    mask_device = train_indices.device

                    # Create new masks from indices
                    new_val_mask = torch.zeros(
                        store.num_nodes, dtype=torch.bool, device=mask_device
                    )
                    new_val_mask[val_idx] = True

                    # Derive the new train mask by removing validation nodes
                    if store.train_mask.dtype == torch.bool:
                        base_train_mask = store.train_mask
                    else:
                        base_train_mask = torch.zeros(
                            store.num_nodes, dtype=torch.bool, device=mask_device
                        )
                        base_train_mask[store.train_mask] = True

                    new_train_mask = base_train_mask & ~new_val_mask

                    store.train_mask = new_train_mask
                    store.val_mask = new_val_mask

        # Raise error if no node type had a training mask to avoid silent failures
        if not found_train_mask:
            raise ValueError("Data object does not have any 'train_mask' to split.")

        return data
