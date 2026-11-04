# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""Uncertainty-quantification metrics for Bayesian sheaf models.

Implements the metrics of Gillespie et al. (2025, Section 2.5) on an ensemble
of ``T`` stochastic forward passes stacked as probabilities ``[T, N, C]``:

* **Predictive entropy** -- total (aleatoric + epistemic) uncertainty of the
  ensemble mean prediction (Eq. 12).
* **Epistemic variance** -- class-averaged variance of the per-pass
  probabilities (Eq. 13).
* **Mutual information** -- entropy of the mean minus mean of the entropies,
  isolating the epistemic component (Eq. 14).
* **Expected calibration error** -- confidence/accuracy gap over equally
  spaced confidence bins (Eq. 15).

All functions are pure torch and operate on the full node axis; mask before
calling to restrict to a data split.
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def predictive_entropy(mean_probs: torch.Tensor) -> torch.Tensor:
    r"""Entropy :math:`H[\bar p(y|x)]` of the ensemble mean prediction.

    Args:
        mean_probs: Ensemble-averaged probabilities ``[N, C]``.

    Returns:
        Per-node predictive entropy ``[N]`` (natural log).
    """
    return -(mean_probs * (mean_probs + _EPS).log()).sum(dim=-1)


def epistemic_variance(ensemble_probs: torch.Tensor) -> torch.Tensor:
    """Class-averaged variance of the per-pass probabilities.

    Args:
        ensemble_probs: Stacked per-pass probabilities ``[T, N, C]``.

    Returns:
        Per-node epistemic variance ``[N]``; zero when ``T == 1``.
    """
    if ensemble_probs.size(0) < 2:
        return ensemble_probs.new_zeros(ensemble_probs.shape[1])
    return ensemble_probs.var(dim=0, unbiased=True).mean(dim=-1)


def mutual_information(ensemble_probs: torch.Tensor) -> torch.Tensor:
    r"""Epistemic uncertainty :math:`H[\bar p] - \tfrac{1}{T}\sum_t H[p_t]`.

    Args:
        ensemble_probs: Stacked per-pass probabilities ``[T, N, C]``.

    Returns:
        Per-node mutual information ``[N]`` (clamped at zero: the analytic
        quantity is nonnegative, finite ensembles can dip below by rounding).
    """
    mean_probs = ensemble_probs.mean(dim=0)
    entropy_of_mean = predictive_entropy(mean_probs)
    mean_of_entropies = (
        -(ensemble_probs * (ensemble_probs + _EPS).log()).sum(dim=-1).mean(dim=0)
    )
    return (entropy_of_mean - mean_of_entropies).clamp(min=0.0)


def expected_calibration_error(
    mean_probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> torch.Tensor:
    r"""Expected Calibration Error over equally spaced confidence bins.

    .. math::

        \mathrm{ECE} = \sum_m \frac{|B_m|}{N}
        \,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|.

    Args:
        mean_probs: Ensemble-averaged probabilities ``[N, C]``.
        labels: Ground-truth class indices ``[N]``.
        n_bins: Number of equally spaced confidence intervals.

    Returns:
        Scalar ECE in :math:`[0, 1]`.
    """
    if mean_probs.numel() == 0:
        return mean_probs.new_zeros(())
    confidence, prediction = mean_probs.max(dim=-1)
    correct = prediction.eq(labels).float()
    # Bin by confidence; bucketize on the upper edges keeps bins half-open
    # (m-1)/M < p <= m/M as in the reference definition.
    edges = torch.linspace(
        0.0, 1.0, n_bins + 1, device=mean_probs.device, dtype=confidence.dtype
    )
    bin_idx = torch.bucketize(confidence, edges[1:-1], right=False)
    ece = mean_probs.new_zeros(())
    total = confidence.numel()
    for m in range(n_bins):
        in_bin = bin_idx == m
        count = int(in_bin.sum())
        if count == 0:
            continue
        acc = correct[in_bin].mean()
        conf = confidence[in_bin].mean()
        ece = ece + (count / total) * (acc - conf).abs()
    return ece


def uncertainty_summary(
    ensemble_probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> dict[str, float]:
    """Aggregate UQ metrics of an ensemble over one node set.

    Args:
        ensemble_probs: Stacked per-pass probabilities ``[T, N, C]``.
        labels: Ground-truth class indices ``[N]``.
        n_bins: Confidence bins for the ECE.

    Returns:
        Mapping with mean predictive entropy, mean epistemic variance, mean
        mutual information, and ECE (all floats; zeros for empty input).
    """
    if ensemble_probs.numel() == 0:
        return {
            "entropy": 0.0,
            "epistemic_variance": 0.0,
            "mutual_information": 0.0,
            "ece": 0.0,
        }
    mean_probs = ensemble_probs.mean(dim=0)
    return {
        "entropy": float(predictive_entropy(mean_probs).mean()),
        "epistemic_variance": float(epistemic_variance(ensemble_probs).mean()),
        "mutual_information": float(mutual_information(ensemble_probs).mean()),
        "ece": float(expected_calibration_error(mean_probs, labels, n_bins)),
    }


__all__ = [
    "epistemic_variance",
    "expected_calibration_error",
    "mutual_information",
    "predictive_entropy",
    "uncertainty_summary",
]
