# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Heterogeneous and graph-classification dataset modules for sheaf models."""

from exp.datasets.graph_classification import (
    # Shims  homogeneous
    COLLABDataModule,
    DDDataModule,
    ENZYMESDataModule,
    # Transforms
    ExtractTypes,
    # Registries
    GCDatasetEntry,
    GCHetDatasetEntry,
    GCHetSheafDataModule,
    # DataModules
    GCSheafDataModule,
    # Legacy
    GraphClassificationDataModule,
    IMDBBinaryDataModule,
    IMDBMultiDataModule,
    # Shims  heterogeneous
    MUTAGDataModule,
    NCI1DataModule,
    NCI109DataModule,
    PROTEINSDataModule,
    PTCMRDataModule,
    REDDIT5KDataModule,
    REDDITBinaryDataModule,
    SetHomogeneous,
    gc_het_registry,
    gc_homo_registry,
)
from exp.datasets.hgb import (
    ACMDataModule,
    DBLPDataModule,
    FreebaseDataModule,
    # Registry
    HetDatasetEntry,
    # DataModules
    HetSheafDataModule,
    HGBAmazonLPDataModule,
    # Legacy
    HGBBaseDataModule,
    # Link prediction
    HGBLinkPredBase,
    HGBPubMedLPDataModule,
    HGBYouTubeLPDataModule,
    IMDBDataModule,
    # Loader
    _load_hgb_dataset,
    het_dataset_registry,
)
from exp.datasets.info import GCDatasetInfo, HetDatasetInfo, LPDatasetInfo
from exp.datasets.link_pred import (
    LastFMDataModule,
    LinkPredBase,
    MovieLensDataModule,
)
from exp.datasets.utils import RemoveSelfLoops, SubsetTrainSplit

__all__ = [
    # Info
    "HetDatasetInfo",
    "GCDatasetInfo",
    "LPDatasetInfo",
    # Link prediction
    "LinkPredBase",
    "LastFMDataModule",
    "MovieLensDataModule",
    # Utils
    "RemoveSelfLoops",
    "SubsetTrainSplit",
    # HGB registry / loader
    "HetDatasetEntry",
    "het_dataset_registry",
    "_load_hgb_dataset",
    # HGB DataModules
    "HetSheafDataModule",
    "IMDBDataModule",
    "DBLPDataModule",
    "ACMDataModule",
    "FreebaseDataModule",
    "HGBBaseDataModule",
    "HGBLinkPredBase",
    "HGBAmazonLPDataModule",
    "HGBPubMedLPDataModule",
    "HGBYouTubeLPDataModule",
    # GC transforms
    "ExtractTypes",
    "SetHomogeneous",
    # GC registries
    "GCDatasetEntry",
    "GCHetDatasetEntry",
    "gc_homo_registry",
    "gc_het_registry",
    # GC DataModules
    "GCSheafDataModule",
    "GCHetSheafDataModule",
    "GraphClassificationDataModule",
    "MUTAGDataModule",
    "PROTEINSDataModule",
    "ENZYMESDataModule",
    "NCI1DataModule",
    "NCI109DataModule",
    "PTCMRDataModule",
    "DDDataModule",
    "COLLABDataModule",
    "IMDBBinaryDataModule",
    "IMDBMultiDataModule",
    "REDDITBinaryDataModule",
    "REDDIT5KDataModule",
]
