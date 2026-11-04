# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Unit tests for the NLSD edge-space nonlinearities."""

import random

import pytest
import torch

from sheaf_mpnn.nlsd.nonlinearity import (
    EdgeThresholdNonlinearity,
    IdentityNonlinearity,
    MLPNonlinearity,
    ScalarThresholdNonlinearity,
    psi_derivative,
    straight_through_indicator,
)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


NUM_EDGES, STALK_DIM, CHANNELS, CONTEXT_DIM = 6, 3, 4, 12


@pytest.fixture
def edge_batch():
    set_seed(42)
    edge_stalk = torch.randn(NUM_EDGES, STALK_DIM, CHANNELS)
    x_dst = torch.randn(NUM_EDGES, CONTEXT_DIM)
    x_src = torch.randn(NUM_EDGES, CONTEXT_DIM)
    return edge_stalk, x_dst, x_src


class TestStraightThroughIndicator:
    def test_forward_is_the_hard_indicator(self):
        sq_norm = torch.tensor([0.1, 0.9, 1.0, 2.5])
        threshold = torch.tensor(1.0)
        out = straight_through_indicator(sq_norm, threshold)
        torch.testing.assert_close(out, torch.tensor([1.0, 1.0, 0.0, 0.0]))

    def test_gradient_reaches_the_threshold(self):
        """The hard step has zero gradient a.e.; the surrogate must not."""
        sq_norm = torch.tensor([0.5, 1.2])
        threshold = torch.tensor(1.0, requires_grad=True)
        straight_through_indicator(sq_norm, threshold).sum().backward()
        assert threshold.grad is not None
        assert threshold.grad.abs() > 0


class TestPsiDerivative:
    def test_gates_out_high_disagreement_edges(self):
        """Bounded confidence: stalks above the threshold stop interacting."""
        edge_stalk = torch.zeros(2, STALK_DIM, 1)
        edge_stalk[0, 0, 0] = 0.5  # squared norm 0.25 -> below threshold
        edge_stalk[1, 0, 0] = 3.0  # squared norm 9.00 -> above threshold
        out = psi_derivative(edge_stalk, "linear", torch.tensor(1.0))
        torch.testing.assert_close(out[0], edge_stalk[0])
        torch.testing.assert_close(out[1], torch.zeros_like(edge_stalk[1]))

    def test_preserves_shape(self, edge_batch):
        edge_stalk, _, _ = edge_batch
        out = psi_derivative(edge_stalk, "linear", torch.tensor(5.0))
        assert out.shape == edge_stalk.shape

    @pytest.mark.parametrize("psi", ["linear", "cos"])
    def test_is_odd(self, edge_batch, psi):
        """Oddness is what lets both edge orientations share one Phi output."""
        edge_stalk, _, _ = edge_batch
        threshold = torch.tensor(4.0)
        pos = psi_derivative(edge_stalk, psi, threshold)
        neg = psi_derivative(-edge_stalk, psi, threshold)
        torch.testing.assert_close(pos, -neg)

    def test_threshold_may_be_per_edge(self, edge_batch):
        edge_stalk, _, _ = edge_batch
        threshold = torch.rand(NUM_EDGES, 1, 1) + 0.5
        out = psi_derivative(edge_stalk, "linear", threshold)
        assert out.shape == edge_stalk.shape

    def test_unknown_psi_raises(self, edge_batch):
        edge_stalk, _, _ = edge_batch
        with pytest.raises(ValueError, match="psi must be one of"):
            psi_derivative(
                edge_stalk,
                "quadratic",  # ty: ignore[invalid-argument-type]
                torch.tensor(1.0),
            )

    def test_cos_stays_finite_at_zero_threshold(self, edge_batch):
        """pi/threshold must not turn a fully closed gate into NaNs.

        A zero threshold is reachable in training: |tau| crossing zero for the
        scalar variant, or the edge-predicted threshold on two identical
        endpoint contexts.  The gate is closed there, so the output must be a
        finite zero, not 0 * inf.
        """
        edge_stalk, _, _ = edge_batch
        out = psi_derivative(edge_stalk, "cos", torch.tensor(0.0))
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, torch.zeros_like(out))


class TestIdentityNonlinearity:
    def test_returns_input_unchanged(self, edge_batch):
        edge_stalk, x_dst, x_src = edge_batch
        out = IdentityNonlinearity()(edge_stalk, x_dst, x_src)
        assert out is edge_stalk


class TestScalarThresholdNonlinearity:
    def test_output_shape_and_finiteness(self, edge_batch):
        edge_stalk, x_dst, x_src = edge_batch
        out = ScalarThresholdNonlinearity()(edge_stalk, x_dst, x_src)
        assert out.shape == edge_stalk.shape
        assert torch.isfinite(out).all()

    def test_threshold_is_learnable(self, edge_batch):
        edge_stalk, x_dst, x_src = edge_batch
        phi = ScalarThresholdNonlinearity(init_threshold=1.0)
        phi(edge_stalk, x_dst, x_src).pow(2).sum().backward()
        assert phi.threshold.grad is not None
        assert phi.threshold.grad.abs() > 0

    def test_negative_threshold_is_used_by_magnitude(self, edge_batch):
        """|tau| keeps the gate valid if training drives tau below zero."""
        edge_stalk, x_dst, x_src = edge_batch
        pos = ScalarThresholdNonlinearity(init_threshold=2.0)
        neg = ScalarThresholdNonlinearity(init_threshold=-2.0)
        torch.testing.assert_close(
            pos(edge_stalk, x_dst, x_src), neg(edge_stalk, x_dst, x_src)
        )

    def test_reset_parameters_restores_init(self, edge_batch):
        phi = ScalarThresholdNonlinearity(init_threshold=1.5)
        with torch.no_grad():
            phi.threshold.fill_(9.0)
        phi.reset_parameters()
        torch.testing.assert_close(phi.threshold, torch.tensor(1.5))

    def test_unknown_psi_raises(self):
        with pytest.raises(ValueError, match="psi must be one of"):
            ScalarThresholdNonlinearity(psi="quadratic")  # ty: ignore[invalid-argument-type]


class TestEdgeThresholdNonlinearity:
    def test_output_shape(self, edge_batch):
        edge_stalk, x_dst, x_src = edge_batch
        phi = EdgeThresholdNonlinearity(CONTEXT_DIM)
        assert phi(edge_stalk, x_dst, x_src).shape == edge_stalk.shape

    def test_endpoint_order_does_not_matter(self, edge_batch):
        """Both directed copies of an edge must predict the same threshold."""
        edge_stalk, x_dst, x_src = edge_batch
        phi = EdgeThresholdNonlinearity(CONTEXT_DIM)
        torch.testing.assert_close(
            phi(edge_stalk, x_dst, x_src), phi(edge_stalk, x_src, x_dst)
        )

    def test_threshold_head_is_learnable(self, edge_batch):
        edge_stalk, x_dst, x_src = edge_batch
        phi = EdgeThresholdNonlinearity(CONTEXT_DIM)
        phi(edge_stalk, x_dst, x_src).pow(2).sum().backward()
        assert phi.threshold.weight.grad is not None
        assert phi.threshold.weight.grad.abs().sum() > 0

    def test_threshold_head_has_no_bias(self):
        assert EdgeThresholdNonlinearity(CONTEXT_DIM).threshold.bias is None


class TestMLPNonlinearity:
    def test_output_shape(self, edge_batch):
        edge_stalk, x_dst, x_src = edge_batch
        phi = MLPNonlinearity(STALK_DIM, CHANNELS)
        assert phi(edge_stalk, x_dst, x_src).shape == edge_stalk.shape

    def test_maps_zero_to_zero(self, edge_batch):
        """Bias-free layers keep Phi(0) = 0, so inert edges stay inert."""
        _, x_dst, x_src = edge_batch
        phi = MLPNonlinearity(STALK_DIM, CHANNELS)
        zero = torch.zeros(NUM_EDGES, STALK_DIM, CHANNELS)
        torch.testing.assert_close(phi(zero, x_dst, x_src), zero)

    def test_is_not_odd(self, edge_batch):
        """ReLU makes Phi genuinely nonlinear -- the point of the method."""
        edge_stalk, x_dst, x_src = edge_batch
        phi = MLPNonlinearity(STALK_DIM, CHANNELS)
        assert not torch.allclose(
            phi(edge_stalk, x_dst, x_src), -phi(-edge_stalk, x_dst, x_src)
        )

    def test_mixes_stalk_dimensions(self, edge_batch):
        """Unlike the psi gate, the MLP is not a per-stalk rescaling."""
        set_seed(0)
        _, x_dst, x_src = edge_batch
        phi = MLPNonlinearity(STALK_DIM, CHANNELS)
        one_hot = torch.zeros(1, STALK_DIM, CHANNELS)
        one_hot[0, 0, 0] = 1.0
        out = phi(one_hot, x_dst[:1], x_src[:1])
        assert (out[0, 1:].abs().sum()) > 0

    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_depth_is_configurable(self, edge_batch, num_layers):
        edge_stalk, x_dst, x_src = edge_batch
        phi = MLPNonlinearity(STALK_DIM, CHANNELS, num_layers=num_layers)
        assert sum(1 for m in phi.mlp if isinstance(m, torch.nn.Linear)) == num_layers
        assert phi(edge_stalk, x_dst, x_src).shape == edge_stalk.shape

    def test_non_positive_depth_raises(self):
        with pytest.raises(ValueError, match="num_layers must be positive"):
            MLPNonlinearity(STALK_DIM, CHANNELS, num_layers=0)

    def test_reset_parameters_changes_weights(self, edge_batch):
        set_seed(1)
        phi = MLPNonlinearity(STALK_DIM, CHANNELS)
        before = phi.mlp[0].weight.detach().clone()  # ty: ignore[call-non-callable]
        phi.reset_parameters()
        assert not torch.allclose(before, phi.mlp[0].weight)  # ty: ignore[invalid-argument-type]
