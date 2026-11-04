# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Heterogeneous link-prediction DataModules.

Each DataModule downloads a PyG dataset, applies standard transforms
(``Constant``, ``ToUndirected``, ``NormalizeFeatures``, ``RemoveSelfLoops``),
splits edges with :class:`~torch_geometric.transforms.RandomLinkSplit`, and
exposes full-graph dataloaders via
:class:`~torch_geometric.data.lightning.LightningLinkData`.

Supported datasets
------------------
* :class:`LastFMDataModule`   - user->artist listening graph (PyG ``LastFM``).
* :class:`MovieLensDataModule` - user->movie rating graph (PyG ``MovieLens``).
"""

from __future__ import annotations

from typing import Any, Literal, cast

import lightning as L
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.lightning import LightningLinkData
from torch_geometric.datasets import LastFM
from torch_geometric.datasets import MovieLens as _MovieLensBase

from exp.datasets.info import LPDatasetInfo
from exp.datasets.utils import RemoveSelfLoops

DATA_DIR = "data"


# ---------------------------------------------------------------------------
# Compatibility shim: sentence-transformers 5.x rejects pandas StringArray
# (df['title'].values)  convert to plain list before encoding.
# ---------------------------------------------------------------------------


class _MovieLens(_MovieLensBase):
    """PyG MovieLens with a sentence-transformers 5.x compatibility fix."""

    def process(self) -> None:  # pragma: no cover
        import pandas as pd
        import torch
        from sentence_transformers import SentenceTransformer
        from torch_geometric.data import HeteroData

        data = HeteroData()

        df = pd.read_csv(self.raw_paths[0], index_col="movieId")
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = df["genres"].str.get_dummies("|").values
        genres = torch.from_numpy(genres).to(torch.float)

        model = SentenceTransformer(self.model_name)
        with torch.no_grad():
            # .tolist() converts pandas StringArray  plain Python strings,
            # which sentence-transformers 5.0 requires.
            emb = model.encode(
                df["title"].tolist(),
                show_progress_bar=True,
                convert_to_tensor=True,
            ).cpu()

        data["movie"].x = torch.cat([emb, genres], dim=-1)

        df = pd.read_csv(self.raw_paths[1])
        user_mapping = {idx: i for i, idx in enumerate(df["userId"].unique())}
        data["user"].num_nodes = len(user_mapping)

        src = [user_mapping[idx] for idx in df["userId"]]
        dst = [movie_mapping[idx] for idx in df["movieId"]]
        edge_index = torch.tensor([src, dst])

        rating = torch.from_numpy(df["rating"].values).to(torch.long)
        time = torch.from_numpy(df["timestamp"].values).to(torch.long)

        data["user", "rates", "movie"].edge_index = edge_index
        data["user", "rates", "movie"].edge_label = rating
        data["user", "rates", "movie"].time = time

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class LinkPredBase(L.LightningDataModule):
    """Base class for heterogeneous link-prediction DataModules.

    Subclasses implement :meth:`download_data` to return a prepared
    ``HeteroData`` object; this class handles the edge split, attribute
    population, and dataloaders.

    Args:
        target: Edge-type triple ``(src, rel, dst)`` to predict.
        rev_target: Reverse edge-type triple for undirected split.
        data_dir: Root directory for dataset caching.
        is_homogeneous: If ``True``, convert to homogeneous before splitting.
        split_ratio: ``(train, val, test)`` fractions (must sum to <= 1).
        node_feature_type: ``"constant"`` (default) - adds a scalar 1 feature
            to featureless nodes; ``"degree"`` - uses log-normalised in/out
            degree as node features, giving the model a structural signal even
            when no raw attributes exist (e.g. LastFM, MovieLens).
    """

    def __init__(
        self,
        target: tuple[str, str, str],
        rev_target: tuple[str, str, str],
        data_dir: str = DATA_DIR,
        is_homogeneous: bool = False,
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        node_feature_type: Literal["constant", "degree"] = "constant",
    ) -> None:
        super().__init__()
        self.target = target
        self.rev_target = rev_target
        self.data_dir = data_dir
        self.is_homogeneous = is_homogeneous
        self.split_ratio = split_ratio
        self.node_feature_type = node_feature_type

        # Always use T.Constant() in the HeteroData transform pipeline:
        # LocalDegreeProfile only works on homogeneous Data, not HeteroData.
        # When node_feature_type="degree", we overwrite x with degree features
        # in prepare_data() after to_homogeneous() (see below).
        self.transform = T.Compose(
            [
                T.Constant(),
                T.ToUndirected(),
                T.NormalizeFeatures(),
                RemoveSelfLoops(),
            ]
        )

        self.info: LPDatasetInfo | None = None
        self.train_data: Data | HeteroData | None = None
        self.val_data: Data | HeteroData | None = None
        self.test_data: Data | HeteroData | None = None

    def download_data(self) -> HeteroData:
        """Download and return the raw ``HeteroData``.  Implement in subclasses."""
        raise NotImplementedError

    def prepare_data(self) -> None:
        """Download -> transform -> split.  Called once before ``setup()``."""
        if not self.is_homogeneous:
            raise ValueError(
                "LP DataModules require is_homogeneous=True; "
                "pass homogeneous=True when constructing the DataModule."
            )
        data: HeteroData | Data = self.download_data()

        num_node_types = len(data.node_types)
        num_edge_types = len(data.edge_types)
        in_channels: Any = {nt: data[nt].num_features for nt in data.node_types}

        if self.is_homogeneous:
            data = data.to_homogeneous()

            if self.node_feature_type == "degree":
                # Replace the constant placeholder with log(out_deg + 1) and
                # log(in_deg + 1)  two informative structural features that work
                # on featureless graphs (LastFM, MovieLens).
                import torch
                from torch_geometric.utils import degree as _deg

                n = data.num_nodes
                assert n is not None
                assert data.edge_index is not None
                out_d = _deg(data.edge_index[0], n, dtype=torch.float)
                in_d = _deg(data.edge_index[1], n, dtype=torch.float)
                data.x = torch.stack([(out_d + 1).log(), (in_d + 1).log()], dim=1)

            node_type_names: tuple[str, ...] = tuple(data._node_type_names)
            edge_type_names: tuple[str, ...] = tuple(
                str(e) for e in data._edge_type_names
            )
            in_channels = data.num_features
            num_node_types = data.num_node_types
            num_edge_types = data.num_edge_types
        else:  # pragma: no cover
            node_type_names = tuple(str(t) for t in data.node_types)
            edge_type_names = tuple(str(t) for t in data.edge_types)

        split = T.RandomLinkSplit(
            num_val=self.split_ratio[1],
            num_test=self.split_ratio[2],
            edge_types=None if self.is_homogeneous else self.target,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=0.6,
            rev_edge_types=None if self.is_homogeneous else self.rev_target,
        )
        self.train_data, self.val_data, self.test_data = split(data)

        self.info = LPDatasetInfo(
            name=repr(self),
            num_features=in_channels if isinstance(in_channels, int) else 0,
            num_classes=1,
            num_splits=1,
            metric="roc_auc",
            split_type="lp_builtin",
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            node_type_names=node_type_names,
            edge_type_names=edge_type_names,
            target=self.target,
        )

    def train_dataloader(self) -> Any:
        """Full-graph training dataloader."""
        assert self.train_data is not None, "Call prepare_data() first"
        return LightningLinkData(self.train_data, loader="full").full_dataloader()

    def val_dataloader(self) -> Any:
        """Full-graph validation dataloader."""
        assert self.val_data is not None, "Call prepare_data() first"
        return LightningLinkData(self.val_data, loader="full").full_dataloader()

    def test_dataloader(self) -> Any:
        """Full-graph test dataloader."""
        assert self.test_data is not None, "Call prepare_data() first"
        return LightningLinkData(self.test_data, loader="full").full_dataloader()


# ---------------------------------------------------------------------------
# Concrete DataModules
# ---------------------------------------------------------------------------


class LastFMDataModule(LinkPredBase):
    """LastFM user->artist listening-history link-prediction dataset.

    Uses PyG's :class:`~torch_geometric.datasets.LastFM` dataset.
    Target edge type: ``("user", "to", "artist")``.
    """

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = True,
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        node_feature_type: Literal["constant", "degree"] = "constant",
    ) -> None:
        super().__init__(
            data_dir=f"{data_dir}/lastfm",
            target=("user", "to", "artist"),
            rev_target=("artist", "to", "user"),
            is_homogeneous=homogeneous,
            split_ratio=split_ratio,
            node_feature_type=node_feature_type,
        )

    def download_data(self) -> HeteroData:
        """Load LastFM and strip pre-computed split edge tensors."""
        data: HeteroData = cast(
            HeteroData, LastFM(self.data_dir, transform=self.transform)[0]
        )
        # PyG's LastFM ships pre-split edge tensors; remove them so that
        # RandomLinkSplit can create consistent train/val/test splits.
        for key in (
            "train_neg_edge_index",
            "val_pos_edge_index",
            "val_neg_edge_index",
            "test_pos_edge_index",
            "test_neg_edge_index",
        ):
            if key in data[self.target]:
                del data[self.target][key]
        return data

    def __repr__(self) -> str:
        """Return dataset name."""
        return "LastFM"


class MovieLensDataModule(LinkPredBase):
    """MovieLens user->movie rating link-prediction dataset.

    Uses PyG's :class:`~torch_geometric.datasets.MovieLens` dataset.
    Target edge type: ``("user", "rates", "movie")``.
    """

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = True,
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        node_feature_type: Literal["constant", "degree"] = "constant",
    ) -> None:
        super().__init__(
            data_dir=f"{data_dir}/movielens",
            target=("user", "rates", "movie"),
            rev_target=("movie", "rev_rates", "user"),
            is_homogeneous=homogeneous,
            split_ratio=split_ratio,
            node_feature_type=node_feature_type,
        )

    def download_data(self) -> HeteroData:
        """Load MovieLens and remove pre-attached edge labels."""
        data: HeteroData = cast(
            HeteroData, _MovieLens(self.data_dir, transform=self.transform)[0]
        )
        # PyG's MovieLens attaches floating-point rating values as edge_label;
        # remove them so RandomLinkSplit creates clean binary existence labels.
        for edge_type in (self.target, self.rev_target):
            if "edge_label" in data[edge_type]:
                del data[edge_type]["edge_label"]
        return data

    def __repr__(self) -> str:
        """Return dataset name."""
        return "MovieLens"
