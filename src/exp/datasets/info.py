# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Info dataclasses for heterogeneous and graph-classification datasets.

Both extend :class:`~exp.data.DatasetInfo` so that the existing training
infrastructure (``module.py``, ``run.py``) can access ``.metric``,
``.num_features``, and ``.num_classes`` without modification.
"""

from __future__ import annotations

from dataclasses import dataclass

from exp.data import DatasetInfo


@dataclass(frozen=True)
class HetDatasetInfo(DatasetInfo):
    """DatasetInfo for HGB datasets converted to homogeneous form.

    Adds type-count fields that sheaf models on heterogeneous graphs need
    (``num_node_types``, ``num_edge_types``) while preserving full backward
    compatibility with the base ``DatasetInfo`` interface.
    """

    num_node_types: int = 0
    num_edge_types: int = 0
    node_type_names: tuple[str, ...] = ()
    edge_type_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class GCDatasetInfo(DatasetInfo):
    """DatasetInfo for TUDataset graph-classification tasks.

    Adds type-count fields and the task label.  ``split_type`` is always
    ``"seed"`` for TU datasets (reproducible random splits keyed by fold).
    """

    num_node_types: int = 1
    num_edge_types: int = 1


@dataclass(frozen=True)
class LPDatasetInfo(DatasetInfo):
    """DatasetInfo for heterogeneous link-prediction datasets.

    ``metric`` is always ``"roc_auc"`` and ``task`` is ``"binary"``
    (edge existence prediction).
    """

    num_node_types: int = 0
    num_edge_types: int = 0
    node_type_names: tuple[str, ...] = ()
    edge_type_names: tuple[str, ...] = ()
    target: tuple[str, str, str] = ("", "", "")
    task: str = "binary"
