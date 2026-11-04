# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for the BSNN variational machinery: Cayley distributions on SO(n),
KL divergences, and the variational sheaf learners.
"""

import math

import pytest
import torch

from sheaf_mpnn.bsnn.variational import (
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


def _assert_special_orthogonal(mats: torch.Tensor, atol: float = 1e-3) -> None:
    # float32 Cayley-transform orthogonality drifts with the BLAS backend
    # (observed 2.1e-4 on Linux CI vs <1e-4 on macOS); 1e-3 keeps a real
    # margin while still failing loudly for anything that is not a rotation.
    d = mats.size(-1)
    eye = torch.eye(d).expand_as(mats)
    torch.testing.assert_close(mats.transpose(-2, -1) @ mats, eye, atol=atol, rtol=0.0)
    torch.testing.assert_close(
        torch.linalg.det(mats), torch.ones(mats.shape[:-2]), atol=atol, rtol=0.0
    )


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------


class TestSamplingPrimitives:
    @pytest.mark.parametrize("d", [2, 3, 4, 5])
    def test_uniform_so_samples_are_rotations(self, d):
        torch.manual_seed(0)
        q = sample_uniform_so(64, d)
        assert q.shape == (64, d, d)
        _assert_special_orthogonal(q)

    def test_uniform_so_is_not_degenerate(self):
        torch.manual_seed(0)
        q = sample_uniform_so(256, 2)
        # Rotation angles should cover the circle, not collapse to a point.
        angles = torch.atan2(q[:, 1, 0], q[:, 0, 0])
        assert angles.std() > 1.0

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_cayley_involution_maps_skew_to_so(self, d):
        torch.manual_seed(0)
        raw = torch.randn(16, d, d)
        skew = raw - raw.transpose(-2, -1)
        rot = cayley_involution(skew)
        _assert_special_orthogonal(rot)

    def test_cayley_involution_is_involutive(self):
        torch.manual_seed(0)
        raw = torch.randn(8, 3, 3)
        skew = raw - raw.transpose(-2, -1)
        torch.testing.assert_close(
            cayley_involution(cayley_involution(skew)), skew, atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_cayley_samples_are_rotations(self, d):
        torch.manual_seed(0)
        mean = sample_uniform_so(32, d)
        scale = torch.full((32, 1), 0.5)
        samples = sample_cayley_rotations(mean, scale)
        _assert_special_orthogonal(samples)

    def test_small_scale_concentrates_on_mean(self):
        torch.manual_seed(0)
        mean = sample_uniform_so(64, 3)
        near = sample_cayley_rotations(mean, torch.full((64, 1), 0.01))
        far = sample_cayley_rotations(mean, torch.full((64, 1), 0.99))
        # The Cayley family is heavy-tailed (wrapped-Cauchy-like), so compare
        # medians: the mean is dominated by the occasional large rotation.
        near_dist = (near - mean).norm(dim=(-2, -1)).median()
        far_dist = (far - mean).norm(dim=(-2, -1)).median()
        assert near_dist < 0.1
        assert far_dist > near_dist

    def test_sampling_is_reparameterized(self):
        torch.manual_seed(0)
        raw = torch.randn(8, 3, 3, requires_grad=True)
        skew = raw - raw.transpose(-2, -1)
        mean = cayley_involution(skew)
        scale = torch.full((8, 1), 0.5, requires_grad=True)
        samples = sample_cayley_rotations(mean, scale)
        samples.sum().backward()
        assert raw.grad is not None and raw.grad.abs().sum() > 0
        assert scale.grad is not None and scale.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# KL divergences
# ---------------------------------------------------------------------------


class TestGaussianKL:
    def test_zero_at_prior(self):
        mean = torch.zeros(10, 4)
        log_var = torch.zeros(10, 4)
        assert gaussian_kl(mean, log_var).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_away_from_prior(self):
        torch.manual_seed(0)
        mean = torch.randn(10, 4)
        log_var = torch.randn(10, 4)
        assert gaussian_kl(mean, log_var).item() > 0

    def test_matches_torch_distributions(self):
        torch.manual_seed(0)
        mean = torch.randn(6, 3)
        log_var = torch.randn(6, 3)
        expected = (
            torch.distributions.kl_divergence(
                torch.distributions.Normal(mean, (0.5 * log_var).exp()),
                torch.distributions.Normal(
                    torch.zeros_like(mean), torch.ones_like(mean)
                ),
            )
            .sum(dim=-1)
            .mean()
        )
        torch.testing.assert_close(gaussian_kl(mean, log_var), expected)


class TestCayleyKL:
    @pytest.mark.parametrize("d", [2, 3])
    def test_zero_at_uniform(self, d):
        # scale = 1 <=> kappa = 0 <=> the posterior IS the uniform prior.
        kl = cayley_kl(torch.ones(10, 1), d)
        assert kl.item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("d", [2, 3])
    def test_monotone_in_concentration(self, d):
        scales = [0.9, 0.5, 0.1]
        kls = [cayley_kl(torch.full((10, 1), s), d).item() for s in scales]
        assert kls[0] < kls[1] < kls[2]
        assert all(k >= 0 for k in kls)

    def test_closed_forms(self):
        scale = torch.full((1, 1), 0.5)
        kappa = (1 - 0.5) / (1 + 0.5)
        expected2 = -math.log(1 - kappa**2)
        expected3 = expected2 - 2 * math.log(1 - kappa) - 2 * kappa
        assert cayley_kl(scale, 2).item() == pytest.approx(expected2, rel=1e-5)
        assert cayley_kl(scale, 3).item() == pytest.approx(expected3, rel=1e-5)

    def test_monte_carlo_estimate_for_large_d(self):
        torch.manual_seed(0)
        # d >= 4 has no closed form; the estimate must be finite and, for a
        # concentrated posterior, clearly positive.
        kl = cayley_kl(torch.full((512, 1), 0.1), 4)
        assert torch.isfinite(kl)
        assert kl.item() > 0

    def test_gradient_flows_through_scale(self):
        for d in (2, 3, 4):
            scale = torch.full((16, 1), 0.5, requires_grad=True)
            cayley_kl(scale, d).backward()
            assert scale.grad is not None and scale.grad.abs().sum() > 0

    def test_log_density_normalizes_at_uniform(self):
        torch.manual_seed(0)
        # kappa = 0 makes the Cayley distribution uniform: density == 1.
        rotations = sample_uniform_so(32, 3)
        log_density = cayley_log_density(rotations, torch.zeros(32), 3)
        torch.testing.assert_close(log_density, torch.zeros(32), atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# Variational sheaf learners
# ---------------------------------------------------------------------------


def _edge_setup(num_nodes=8, context_dim=12):
    torch.manual_seed(0)
    x_feat = torch.randn(num_nodes, context_dim)
    src = torch.tensor([0, 1, 2, 3, 4, 5])
    dst = torch.tensor([1, 0, 3, 2, 5, 4])
    return x_feat, torch.stack([src, dst])


class TestVariationalLearners:
    def test_diagonal_shapes(self):
        d, hidden, ctx = 3, 4, 12
        x_feat, edge_index = _edge_setup(context_dim=ctx)
        learner = DiagonalVariationalSheafLearner(d, hidden, ctx)
        params = learner(x_feat, edge_index)
        assert params[0].shape == (6, d) and params[1].shape == (6, d)
        sample = learner.sample(params)
        assert sample.shape == (6, d)
        kl = learner.kl(params)
        assert kl.dim() == 0 and kl.item() >= 0

    def test_general_shapes(self):
        d, hidden, ctx = 3, 4, 12
        x_feat, edge_index = _edge_setup(context_dim=ctx)
        learner = GeneralVariationalSheafLearner(d, hidden, ctx)
        params = learner(x_feat, edge_index)
        assert params[0].shape == (6, d * d)
        sample = learner.sample(params)
        assert sample.shape == (6, d, d)
        assert learner.kl(params).item() >= 0

    def test_low_rank_shapes_and_rank_bound(self):
        d, hidden, ctx, r = 3, 4, 12, 2
        x_feat, edge_index = _edge_setup(context_dim=ctx)
        learner = LowRankVariationalSheafLearner(d, hidden, ctx, rank=r)
        params = learner(x_feat, edge_index)
        assert params[0].shape == (6, 2 * d * r)
        sample = learner.sample(params)
        assert sample.shape == (6, d, 2 * r)
        # Noise lives in the factors, so every composed map is rank <= r.
        left, right = sample.chunk(2, dim=-1)
        composed = left @ right.transpose(-2, -1)
        assert (torch.linalg.matrix_rank(composed) <= r).all()
        assert learner.kl(params).item() >= 0

    def test_low_rank_rejects_non_positive_rank(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            LowRankVariationalSheafLearner(3, 4, 12, rank=0)

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_orthogonal_samples_are_rotations(self, d):
        hidden, ctx = 4, 12
        x_feat, edge_index = _edge_setup(context_dim=ctx)
        learner = OrthogonalVariationalSheafLearner(d, hidden, ctx)
        params = learner(x_feat, edge_index)
        assert params[1].shape == (6, 1)
        sample = learner.sample(params)
        assert sample.shape == (6, d, d)
        _assert_special_orthogonal(sample)
        kl = learner.kl(params)
        assert torch.isfinite(kl)

    def test_sparse_learner_input_reduction(self):
        d, hidden = 3, 4
        ctx = d * hidden  # sparse reduction requires context_dim == d * hidden
        x_feat, edge_index = _edge_setup(context_dim=ctx)
        learner = DiagonalVariationalSheafLearner(d, hidden, ctx, sparse=True)
        assert learner.lin_mean.in_features == 2 * hidden
        sample = learner.sample(learner(x_feat, edge_index))
        assert sample.shape == (6, d)

    @pytest.mark.parametrize(
        "learner_cls",
        [
            DiagonalVariationalSheafLearner,
            GeneralVariationalSheafLearner,
            LowRankVariationalSheafLearner,
            OrthogonalVariationalSheafLearner,
        ],
    )
    def test_gradients_reach_both_heads(self, learner_cls):
        d, hidden, ctx = 3, 4, 12
        x_feat, edge_index = _edge_setup(context_dim=ctx)
        learner = learner_cls(d, hidden, ctx)
        params = learner(x_feat, edge_index)
        loss = learner.sample(params).pow(2).sum() + learner.kl(params)
        loss.backward()
        assert learner.lin_mean.weight.grad is not None
        assert learner.lin_mean.weight.grad.abs().sum() > 0
        assert learner.lin_scale.weight.grad is not None
        assert learner.lin_scale.weight.grad.abs().sum() > 0
