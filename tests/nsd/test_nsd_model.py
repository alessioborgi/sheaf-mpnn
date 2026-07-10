# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import random

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.nsd.nsd_model import NSDModel, NSDVariant


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@pytest.mark.parametrize("variant", list(NSDVariant))
class TestNSDModel:
    @pytest.fixture
    def setup(self, variant):
        set_seed(42)
        num_nodes, in_channels, out_channels = 15, 10, 3
        d, hidden_dim, num_layers = 2, 4, 2

        # Model uses:
        # x_stalk features (f) = hidden_dim = 4
        # x_feat context (df) = d * hidden_dim = 2 * 4 = 8
        # Layer map_generator expects 2 * 8 = 16

        x = torch.randn(num_nodes, in_channels)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        model = NSDModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=d,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            variant=variant,
        )

        return model, x, edge_index

    def test_model_forward_shape(self, setup):
        model, x, edge_index = setup
        out = model(x, edge_index)
        assert out.shape == (x.size(0), model.out_channels)
        assert not torch.isnan(out).any()

    def test_model_gradient_flow(self, setup):
        model, x, edge_index = setup
        out = model(x, edge_index)
        loss = out.pow(2).mean()
        loss.backward()

        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0
        assert model.layers[0].W1.grad is not None
        assert model.layers[0].W1.grad.abs().sum() > 0

    def test_model_determinism(self, setup):
        model, x, edge_index = setup
        model.eval()
        with torch.no_grad():
            out1 = model(x, edge_index)
            out2 = model(x, edge_index)
        torch.testing.assert_close(out1, out2)

    def test_isolated_graph_stability(self, setup):
        model, x, _ = setup
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        out = model(x, empty_edge_index)
        assert out.shape == (x.size(0), model.out_channels)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# JKNet
# ---------------------------------------------------------------------------


_JKNET_EDGE_INDEX = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)


class TestNSDModelJKNet:
    def _model(self, normalize_output: bool = False) -> NSDModel:
        set_seed(42)
        return NSDModel(
            in_channels=8,
            out_channels=3,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=3,
            jknet=True,
            normalize_output=normalize_output,
        )

    def test_jk_attribute_created(self):
        model = self._model()
        assert hasattr(model, "jk")

    def test_decoder_expanded_for_jknet(self):
        model = self._model()
        context_dim = model.stalk_dim * model.hidden_dim
        assert model.decoder.in_features == 3 * context_dim

    def test_forward_shape(self):
        model = self._model()
        x = torch.randn(10, 8)
        out = model(x, _JKNET_EDGE_INDEX)
        assert out.shape == (10, model.out_channels)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        model = self._model()
        x = torch.randn(10, 8)
        out = model(x, _JKNET_EDGE_INDEX)
        out.pow(2).mean().backward()
        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0

    def test_normalize_output_with_jknet(self):
        model = self._model(normalize_output=True)
        x = torch.randn(10, 8)
        out = model(x, _JKNET_EDGE_INDEX)
        assert out.shape == (10, model.out_channels)
        assert torch.isfinite(out).all()


def test_nsd_model_stores_hyperparameters():
    """Verify NSDModel stores explicitly passed hyperparameters as attributes."""
    model = NSDModel(in_channels=10, out_channels=2, hidden_dim=32)
    assert model.hidden_dim == 32


def test_nsd_model_stores_rank():
    """Verify NSDModel stores rank for the LOW_RANK variant."""
    model = NSDModel(
        in_channels=10, out_channels=2, variant=NSDVariant.LOW_RANK, rank=3
    )
    assert model.rank == 3


def test_nsd_model_validation():
    """Verify NSDModel enforces valid hyperparameter ranges."""
    with pytest.raises(ValueError, match="stalk_dim must be positive"):
        NSDModel(in_channels=10, out_channels=2, stalk_dim=0)

    with pytest.raises(ValueError, match="must have at least one NSD layer"):
        NSDModel(in_channels=10, out_channels=2, num_layers=0)

    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        NSDModel(in_channels=10, out_channels=2, hidden_dim=0)

    with pytest.raises(ValueError, match="rank must be positive"):
        NSDModel(in_channels=10, out_channels=2, variant=NSDVariant.LOW_RANK, rank=0)


def test_enum_to_layer_mapping():
    """Verify the NSDVariant Enum correctly maps to implementation classes."""
    from sheaf_mpnn.nsd.nsd_layers import (
        DiagonalNSDConv,
        LowRankNSDConv,
        OrthogonalNSDConv,
    )

    assert NSDVariant.DIAGONAL.layer_class == DiagonalNSDConv
    assert NSDVariant.ORTHOGONAL.layer_class == OrthogonalNSDConv
    assert NSDVariant.LOW_RANK.layer_class == LowRankNSDConv


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

_DROPOUT_EDGE_INDEX = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)


class TestNSDModelDropout:
    def test_dropout_probabilities_stored(self):
        model = NSDModel(in_channels=10, out_channels=3, input_dropout=0.3, dropout=0.5)
        assert model.input_dropout_layer.p == pytest.approx(0.3)
        assert model.dropout_layer.p == pytest.approx(0.5)

    def test_eval_suppresses_dropout(self):
        """eval() must make consecutive forward passes identical with high dropout."""
        torch.manual_seed(0)
        x = torch.randn(4, 10)
        model = NSDModel(in_channels=10, out_channels=3, input_dropout=0.9, dropout=0.9)
        model.eval()
        with torch.no_grad():
            out1 = model(x, _DROPOUT_EDGE_INDEX)
            out2 = model(x, _DROPOUT_EDGE_INDEX)
        torch.testing.assert_close(out1, out2)

    def test_train_mode_is_stochastic(self):
        """train() with high dropout must produce different outputs each call."""
        torch.manual_seed(0)
        x = torch.randn(4, 10)
        model = NSDModel(in_channels=10, out_channels=3, input_dropout=0.9, dropout=0.9)
        model.train()
        with torch.no_grad():
            out1 = model(x, _DROPOUT_EDGE_INDEX)
            out2 = model(x, _DROPOUT_EDGE_INDEX)
        assert not torch.allclose(out1, out2)
