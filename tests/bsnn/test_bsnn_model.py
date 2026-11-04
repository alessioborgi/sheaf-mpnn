# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for the end-to-end Bayesian Sheaf Neural Network model."""

import random

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.bsnn.bsnn_model import (
    BSNNModel,
    BSNNVariant,
    reverse_edge_permutation,
)


def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Reverse-edge permutation
# ---------------------------------------------------------------------------


class TestReverseEdgePermutation:
    def test_symmetric_graph_pairs_reverse_edges(self):
        set_seed()
        num_nodes = 12
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)  # symmetric
        perm = reverse_edge_permutation(edge_index, num_nodes)
        torch.testing.assert_close(
            edge_index[:, perm], edge_index.flip(0), atol=0, rtol=0
        )

    def test_self_loops_map_to_themselves(self):
        edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
        perm = reverse_edge_permutation(edge_index, 3)
        torch.testing.assert_close(perm, torch.arange(3), atol=0, rtol=0)

    def test_directed_leftover_falls_back_to_own_index(self):
        # Edge (0, 1) has a reverse; edge (2, 3) does not.
        edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]])
        perm = reverse_edge_permutation(edge_index, 4)
        assert perm[0].item() == 1
        assert perm[1].item() == 0
        assert perm[2].item() == 2  # fallback


# ---------------------------------------------------------------------------
# BSNN model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", list(BSNNVariant))
class TestBSNNModel:
    @pytest.fixture
    def setup(self, variant):
        set_seed(42)
        num_nodes, in_channels, out_channels = 15, 10, 3
        d, hidden_dim, num_layers = 2, 4, 2

        x = torch.randn(num_nodes, in_channels)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        model = BSNNModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=d,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            variant=variant,
        )
        return model, x, edge_index

    def test_forward_shape(self, setup):
        model, x, edge_index = setup
        out = model(x, edge_index)
        assert out.shape == (x.size(0), model.out_channels)
        assert torch.isfinite(out).all()

    def test_kl_is_computed_during_forward(self, setup):
        model, x, edge_index = setup
        assert model.kl_loss().item() == 0.0  # before any forward
        model(x, edge_index)
        kl = model.kl_loss()
        assert kl.dim() == 0
        assert torch.isfinite(kl)
        assert kl.item() >= 0.0

    def test_gradient_flow_through_sampling_and_kl(self, setup):
        model, x, edge_index = setup
        y = torch.randint(0, model.out_channels, (x.size(0),))
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits, y) + model.kl_loss()
        loss.backward()

        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0
        assert model.sheaf_learner.lin_mean.weight.grad is not None
        assert model.sheaf_learner.lin_mean.weight.grad.abs().sum() > 0
        assert model.sheaf_learner.lin_scale.weight.grad is not None
        assert model.sheaf_learner.lin_scale.weight.grad.abs().sum() > 0
        assert model.layers[0].W1.grad is not None
        assert model.layers[0].W1.grad.abs().sum() > 0

    def test_forward_is_stochastic(self, setup):
        model, x, edge_index = setup
        model.eval()
        with torch.no_grad():
            out1 = model(x, edge_index)
            out2 = model(x, edge_index)
        # Different sheaf samples => different outputs (Bayesian by design).
        assert not torch.allclose(out1, out2)

    def test_predict_proba(self, setup):
        model, x, edge_index = setup
        model.eval()
        probs = model.predict_proba(x, edge_index, num_samples=4)
        assert probs.shape == (4, x.size(0), model.out_channels)
        torch.testing.assert_close(
            probs.sum(dim=-1), torch.ones(4, x.size(0)), atol=1e-5, rtol=0.0
        )

    def test_isolated_graph_stability(self, setup):
        model, x, _ = setup
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        out = model(x, empty_edge_index)
        assert out.shape == (x.size(0), model.out_channels)
        assert torch.isfinite(out).all()
        assert model.kl_loss().item() == 0.0

    def test_reset_parameters(self, setup):
        model, x, edge_index = setup
        model.reset_parameters()
        out = model(x, edge_index)
        assert torch.isfinite(out).all()

    def test_option_flags_smoke(self, variant):
        set_seed(0)
        num_nodes, in_channels, out_channels = 12, 8, 3
        x = torch.randn(num_nodes, in_channels)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        model = BSNNModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=2,
            variant=variant,
            add_lp=True,
            add_hp=True,
            second_linear=True,
            jknet=True,
            normalize_output=True,
            learn_alpha=False,
        )
        assert not model.layers[0].alpha.requires_grad
        out = model(x, edge_index)
        assert out.shape == (num_nodes, out_channels)
        assert torch.isfinite(out).all()

    def test_sparse_learner_smoke(self, variant):
        set_seed(0)
        num_nodes, in_channels, out_channels = 12, 8, 3
        x = torch.randn(num_nodes, in_channels)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        model = BSNNModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=1,
            variant=variant,
            sparse_learner=True,
        )
        out = model(x, edge_index)
        assert out.shape == (num_nodes, out_channels)
        assert torch.isfinite(out).all()


class TestBSNNValidation:
    def test_rejects_bad_dims(self):
        with pytest.raises(ValueError, match="stalk_dim"):
            BSNNModel(4, 2, stalk_dim=0)
        with pytest.raises(ValueError, match="hidden_dim"):
            BSNNModel(4, 2, hidden_dim=0)
        with pytest.raises(ValueError, match="layer"):
            BSNNModel(4, 2, num_layers=0)
        with pytest.raises(ValueError, match="ensemble_size"):
            BSNNModel(4, 2, ensemble_size=0)
        with pytest.raises(ValueError, match="stalk_dim >= 2"):
            BSNNModel(4, 2, stalk_dim=1, variant=BSNNVariant.ORTHOGONAL)

    def test_rejects_bad_input_shapes(self):
        model = BSNNModel(4, 2)
        with pytest.raises(ValueError, match="x must have shape"):
            model(torch.randn(5, 3), torch.zeros(2, 0, dtype=torch.long))
        with pytest.raises(ValueError, match="edge_index"):
            model(torch.randn(5, 4), torch.zeros(3, 1, dtype=torch.long))

    def test_low_rank_rank_reaches_learner_and_layers(self):
        set_seed(0)
        model = BSNNModel(
            8, 3, stalk_dim=4, hidden_dim=4, variant=BSNNVariant.LOW_RANK, rank=2
        )
        assert model.sheaf_learner.rank == 2
        assert all(layer.rank == 2 for layer in model.layers)
        x = torch.randn(10, 8)
        edge_index = erdos_renyi_graph(10, edge_prob=0.5)
        out = model(x, edge_index)
        assert out.shape == (10, 3)
        assert torch.isfinite(out).all()
        assert torch.isfinite(model.kl_loss())

    def test_low_rank_rejects_non_positive_rank(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            BSNNModel(4, 2, variant=BSNNVariant.LOW_RANK, rank=0)

    def test_monte_carlo_kl_path_for_large_stalk_dim(self):
        # d >= 4 exercises the MC estimate of the SO(d) KL divergence.
        set_seed(0)
        model = BSNNModel(
            8, 3, stalk_dim=4, hidden_dim=4, variant=BSNNVariant.ORTHOGONAL
        )
        x = torch.randn(10, 8)
        edge_index = erdos_renyi_graph(10, edge_prob=0.5)
        out = model(x, edge_index)
        assert torch.isfinite(out).all()
        assert torch.isfinite(model.kl_loss())
