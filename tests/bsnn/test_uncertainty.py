# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for the BSNN uncertainty-quantification metrics."""

import math

import pytest
import torch

from sheaf_mpnn.bsnn.uncertainty import (
    epistemic_variance,
    expected_calibration_error,
    mutual_information,
    predictive_entropy,
    uncertainty_summary,
)


class TestPredictiveEntropy:
    def test_uniform_is_log_c(self):
        probs = torch.full((5, 4), 0.25)
        torch.testing.assert_close(
            predictive_entropy(probs),
            torch.full((5,), math.log(4)),
            atol=1e-5,
            rtol=0.0,
        )

    def test_one_hot_is_zero(self):
        probs = torch.eye(4)
        torch.testing.assert_close(
            predictive_entropy(probs), torch.zeros(4), atol=1e-5, rtol=0.0
        )


class TestEpistemicVariance:
    def test_identical_passes_have_zero_variance(self):
        probs = torch.softmax(torch.randn(1, 6, 3), dim=-1).expand(4, 6, 3)
        torch.testing.assert_close(
            epistemic_variance(probs), torch.zeros(6), atol=1e-6, rtol=0.0
        )

    def test_single_pass_is_zero(self):
        probs = torch.softmax(torch.randn(1, 6, 3), dim=-1)
        torch.testing.assert_close(
            epistemic_variance(probs), torch.zeros(6), atol=0.0, rtol=0.0
        )

    def test_disagreeing_passes_are_positive(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(4, 6, 3), dim=-1)
        assert (epistemic_variance(probs) > 0).all()


class TestMutualInformation:
    def test_identical_passes_have_zero_mi(self):
        probs = torch.softmax(torch.randn(1, 6, 3), dim=-1).expand(4, 6, 3)
        torch.testing.assert_close(
            mutual_information(probs), torch.zeros(6), atol=1e-5, rtol=0.0
        )

    def test_disagreement_yields_positive_mi(self):
        # Two passes fully confident in different classes: MI = H(mean) > 0.
        p1 = torch.tensor([[1.0, 0.0]]).unsqueeze(0)
        p2 = torch.tensor([[0.0, 1.0]]).unsqueeze(0)
        probs = torch.cat([p1, p2], dim=0)
        assert mutual_information(probs).item() == pytest.approx(math.log(2), rel=1e-4)


class TestECE:
    def test_confident_and_correct_is_zero(self):
        probs = torch.eye(3).repeat(4, 1)
        labels = torch.arange(3).repeat(4)
        assert expected_calibration_error(probs, labels).item() == pytest.approx(
            0.0, abs=1e-6
        )

    def test_confident_and_wrong_is_one(self):
        probs = torch.eye(3).repeat(4, 1)
        labels = (torch.arange(3) + 1).remainder(3).repeat(4)
        assert expected_calibration_error(probs, labels).item() == pytest.approx(
            1.0, abs=1e-6
        )

    def test_empty_input_is_zero(self):
        probs = torch.empty(0, 3)
        labels = torch.empty(0, dtype=torch.long)
        assert expected_calibration_error(probs, labels).item() == 0.0


class TestSummary:
    def test_keys_and_finiteness(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(3, 10, 4), dim=-1)
        labels = torch.randint(0, 4, (10,))
        summary = uncertainty_summary(probs, labels)
        assert set(summary) == {
            "entropy",
            "epistemic_variance",
            "mutual_information",
            "ece",
        }
        assert all(math.isfinite(v) for v in summary.values())
        assert summary["epistemic_variance"] >= 0
        assert summary["mutual_information"] >= 0

    def test_empty_ensemble(self):
        summary = uncertainty_summary(
            torch.empty(3, 0, 4), torch.empty(0, dtype=torch.long)
        )
        assert all(v == 0.0 for v in summary.values())
