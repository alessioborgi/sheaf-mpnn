# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Sheaf message-passing neural network layers and models."""

from sheaf_mpnn.base_conv import BaseSheafConv
from sheaf_mpnn.bsnn import (
    BSNNModel,
    BSNNVariant,
    DiagonalBSNNConv,
    GeneralBSNNConv,
    LowRankBSNNConv,
    OrthogonalBSNNConv,
)
from sheaf_mpnn.bunn import (
    BundleMapLearner,
    BuNNConv,
    BuNNModel,
)
from sheaf_mpnn.csnn import (
    ConformalSheafLearner,
    CSNNConv,
    CSNNModel,
    CSNNVariant,
)
from sheaf_mpnn.id_sheaf.id_sheaf_layers import IdentitySheafConv
from sheaf_mpnn.id_sheaf.id_sheaf_model import IdentitySheafModel
from sheaf_mpnn.jointdiff.joint_layers import JointDiffusionSheafConv
from sheaf_mpnn.jointdiff.joint_model import JointDiffusionModel, RotationInvariantModel
from sheaf_mpnn.jointdiff.rot_inv_layers import RotationInvariantSheafConv
from sheaf_mpnn.nlsd import (
    BaseNLSDConv,
    DiagonalNLSDConv,
    GeneralNLSDConv,
    LowRankNLSDConv,
    NLSDModel,
    NLSDNonlinearity,
    NLSDVariant,
    OrthogonalNLSDConv,
)
from sheaf_mpnn.nsd import (
    BasePolyNSDConv,
    DiagonalNSDConv,
    DiagonalPolyNSDConv,
    GeneralNSDConv,
    GeneralPolyNSDConv,
    LowRankPolyNSDConv,
    NSDModel,
    NSDVariant,
    OrthogonalNSDConv,
    OrthogonalPolyNSDConv,
    PolyNSDModel,
    PolyNSDVariant,
)
from sheaf_mpnn.nsp import (
    DiagonalNSPConv,
    GeneralNSPConv,
    NSPModel,
    NSPVariant,
    OrthogonalNSPConv,
)
from sheaf_mpnn.sheaf_adj import (
    DiagonalSheafAdjConv,
    GeneralSheafAdjConv,
    LowRankSheafAdjConv,
    OrthogonalSheafAdjConv,
    SheafAdjModel,
    SheafAdjVariant,
)
from sheaf_mpnn.sheaf_attn import (
    SheafAttnConv,
    SheafAttnModel,
    SheafAttnVariant,
)
from sheaf_mpnn.sheaf_pool import SheafPool

__all__ = [
    "BaseSheafConv",
    "SheafPool",
    "BSNNModel",
    "BSNNVariant",
    "DiagonalBSNNConv",
    "GeneralBSNNConv",
    "LowRankBSNNConv",
    "OrthogonalBSNNConv",
    # NSD
    "DiagonalNSDConv",
    "GeneralNSDConv",
    "OrthogonalNSDConv",
    "NSDModel",
    "NSDVariant",
    # NLSD
    "BaseNLSDConv",
    "DiagonalNLSDConv",
    "OrthogonalNLSDConv",
    "GeneralNLSDConv",
    "LowRankNLSDConv",
    "NLSDModel",
    "NLSDVariant",
    "NLSDNonlinearity",
    # BuNN
    "BuNNConv",
    "BuNNModel",
    "BundleMapLearner",
    # PolyNSD
    "BasePolyNSDConv",
    "DiagonalPolyNSDConv",
    "GeneralPolyNSDConv",
    "OrthogonalPolyNSDConv",
    "LowRankPolyNSDConv",
    "PolyNSDModel",
    "PolyNSDVariant",
    # NSP
    "DiagonalNSPConv",
    "GeneralNSPConv",
    "OrthogonalNSPConv",
    "NSPModel",
    "NSPVariant",
    # SheafAdj
    "DiagonalSheafAdjConv",
    "GeneralSheafAdjConv",
    "LowRankSheafAdjConv",
    "OrthogonalSheafAdjConv",
    "SheafAdjModel",
    "SheafAdjVariant",
    # SheafAttn
    "SheafAttnConv",
    "SheafAttnModel",
    "SheafAttnVariant",
    "IdentitySheafConv",
    "IdentitySheafModel",
    "JointDiffusionSheafConv",
    "RotationInvariantSheafConv",
    "JointDiffusionModel",
    "RotationInvariantModel",
    "CSNNConv",
    "CSNNModel",
    "CSNNVariant",
    "ConformalSheafLearner",
]
