# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Bayesian Sheaf Neural Networks (Gillespie et al., 2025, arXiv:2410.09590).

The cellular sheaf is a latent random variable: a variational sheaf learner
parameterizes a distribution over restriction maps, one i.i.d. sheaf sample
drives each diffusion layer, and training maximizes the ELBO (cross-entropy
plus an annealed KL regularizer). Test-time predictions ensemble multiple
stochastic forward passes, enabling uncertainty quantification.
"""

from sheaf_mpnn.bsnn.bsnn_layers import (
    BaseBSNNConv,
    DiagonalBSNNConv,
    GeneralBSNNConv,
    LowRankBSNNConv,
    OrthogonalBSNNConv,
)
from sheaf_mpnn.bsnn.bsnn_model import (
    BSNNModel,
    BSNNVariant,
    reverse_edge_permutation,
)
from sheaf_mpnn.bsnn.uncertainty import (
    epistemic_variance,
    expected_calibration_error,
    mutual_information,
    predictive_entropy,
    uncertainty_summary,
)
from sheaf_mpnn.bsnn.variational import (
    BaseVariationalSheafLearner,
    DiagonalVariationalSheafLearner,
    GeneralVariationalSheafLearner,
    LowRankVariationalSheafLearner,
    OrthogonalVariationalSheafLearner,
    cayley_involution,
    cayley_kl,
    cayley_log_density,
    gaussian_kl,
    sample_cayley_rotations,
    sample_uniform_so,
)

__all__ = [
    "BaseBSNNConv",
    "DiagonalBSNNConv",
    "GeneralBSNNConv",
    "LowRankBSNNConv",
    "OrthogonalBSNNConv",
    "BSNNModel",
    "BSNNVariant",
    "reverse_edge_permutation",
    "BaseVariationalSheafLearner",
    "DiagonalVariationalSheafLearner",
    "GeneralVariationalSheafLearner",
    "LowRankVariationalSheafLearner",
    "OrthogonalVariationalSheafLearner",
    "cayley_involution",
    "cayley_kl",
    "cayley_log_density",
    "gaussian_kl",
    "sample_cayley_rotations",
    "sample_uniform_so",
    "epistemic_variance",
    "expected_calibration_error",
    "mutual_information",
    "predictive_entropy",
    "uncertainty_summary",
]
