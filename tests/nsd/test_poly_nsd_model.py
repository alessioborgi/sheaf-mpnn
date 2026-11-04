# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.nsd.poly_nsd_model import PolyNSDModel, PolyNSDVariant
from tests.testing import set_seed

# ---------------------------------------------------------------------------
# Parametrised forward / gradient tests for all variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", list(PolyNSDVariant))
class TestPolyNSDModel:
    """Shared forward and gradient tests for every PolyNSD variant."""

    @pytest.fixture
    def setup(self, variant):
        set_seed(42)
        num_nodes, in_channels, out_channels = 15, 10, 3
        d, hidden_dim, num_layers = 2, 4, 2

        x = torch.randn(num_nodes, in_channels)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        model = PolyNSDModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=d,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            variant=variant,
            degree=3,
        )
        return model, x, edge_index

    def test_model_forward_shape(self, setup):
        """Forward must return [N, out_channels] with no NaN."""
        model, x, edge_index = setup
        out = model(x, edge_index)
        assert out.shape == (x.size(0), model.out_channels)
        assert not torch.isnan(out).any()

    def test_model_gradient_flow(self, setup):
        """Gradients must reach the encoder and the first poly layer.

        theta_raw is perturbed before the backward pass so the polynomial
        weights are unequal; at the zero initialisation the Chebyshev recurrence
        can cancel the direct z contribution for even degrees, leaving W1/W2
        gradients numerically zero despite the graph being connected.
        """
        model, x, edge_index = setup
        torch.manual_seed(1)
        for layer in model.layers:
            layer.theta_raw.data = torch.randn_like(layer.theta_raw)
        out = model(x, edge_index)
        out.pow(2).mean().backward()
        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0
        assert model.layers[0].W1.grad is not None
        assert model.layers[0].W1.grad.abs().sum() > 0
        assert model.layers[0].theta_raw.grad is not None
        assert model.layers[0].theta_raw.grad.abs().sum() > 0

    def test_model_determinism(self, setup):
        """eval() must produce identical outputs on consecutive forward passes."""
        model, x, edge_index = setup
        model.eval()
        with torch.no_grad():
            out1 = model(x, edge_index)
            out2 = model(x, edge_index)
        torch.testing.assert_close(out1, out2)

    def test_isolated_graph_stability(self, setup):
        """Empty edge_index must not crash and must return finite outputs."""
        model, x, _ = setup
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        out = model(x, empty_edge_index)
        assert out.shape == (x.size(0), model.out_channels)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_model_validation_stalk_dim():
    with pytest.raises(ValueError, match="stalk_dim must be positive"):
        PolyNSDModel(in_channels=10, out_channels=2, stalk_dim=0)


def test_model_validation_hidden_dim():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        PolyNSDModel(in_channels=10, out_channels=2, hidden_dim=0)


def test_model_validation_num_layers():
    with pytest.raises(ValueError, match="must have at least one PolyNSD layer"):
        PolyNSDModel(in_channels=10, out_channels=2, num_layers=0)


def test_model_validation_degree():
    with pytest.raises(ValueError, match="degree must be at least 1"):
        PolyNSDModel(in_channels=10, out_channels=2, degree=0)


def test_model_validation_low_rank():
    with pytest.raises(ValueError, match="rank must be positive"):
        PolyNSDModel(
            in_channels=10, out_channels=2, variant=PolyNSDVariant.LOW_RANK, rank=0
        )


# ---------------------------------------------------------------------------
# Hyperparameter storage
# ---------------------------------------------------------------------------


def test_model_stores_hyperparameters():
    """Constructor arguments must be accessible as model attributes."""
    model = PolyNSDModel(
        in_channels=10, out_channels=2, hidden_dim=32, degree=5, stalk_dim=3
    )
    assert model.hidden_dim == 32
    assert model.degree == 5
    assert model.stalk_dim == 3


def test_model_stores_rank():
    """The rank attribute must be stored for the LOW_RANK variant."""
    model = PolyNSDModel(
        in_channels=10, out_channels=2, variant=PolyNSDVariant.LOW_RANK, rank=3
    )
    assert model.rank == 3


# ---------------------------------------------------------------------------
# Enum to layer class mapping
# ---------------------------------------------------------------------------


def test_enum_to_layer_mapping():
    """PolyNSDVariant must map to the correct polynomial layer classes."""
    from sheaf_mpnn.nsd.poly_nsd_layers import (
        DiagonalPolyNSDConv,
        LowRankPolyNSDConv,
        OrthogonalPolyNSDConv,
    )

    assert PolyNSDVariant.DIAGONAL.layer_class == DiagonalPolyNSDConv
    assert PolyNSDVariant.ORTHOGONAL.layer_class == OrthogonalPolyNSDConv
    assert PolyNSDVariant.LOW_RANK.layer_class == LowRankPolyNSDConv


# ---------------------------------------------------------------------------
# Dropout behaviour
# ---------------------------------------------------------------------------

_EDGE_INDEX = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)


class TestPolyNSDModelDropout:
    def test_dropout_probabilities_stored(self):
        model = PolyNSDModel(
            in_channels=10, out_channels=3, input_dropout=0.3, dropout=0.5
        )
        assert model.input_dropout_layer.p == pytest.approx(0.3)
        assert model.dropout_layer.p == pytest.approx(0.5)

    def test_eval_suppresses_dropout(self):
        """eval() must make consecutive forward passes identical with high dropout."""
        torch.manual_seed(0)
        x = torch.randn(4, 10)
        model = PolyNSDModel(
            in_channels=10, out_channels=3, input_dropout=0.9, dropout=0.9
        )
        model.eval()
        with torch.no_grad():
            out1 = model(x, _EDGE_INDEX)
            out2 = model(x, _EDGE_INDEX)
        torch.testing.assert_close(out1, out2)

    def test_train_mode_is_stochastic(self):
        """train() with high dropout must produce different outputs each call."""
        torch.manual_seed(0)
        x = torch.randn(4, 10)
        model = PolyNSDModel(
            in_channels=10, out_channels=3, input_dropout=0.9, dropout=0.9
        )
        model.train()
        with torch.no_grad():
            out1 = model(x, _EDGE_INDEX)
            out2 = model(x, _EDGE_INDEX)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# JKNet
# ---------------------------------------------------------------------------

_JKNET_EDGE_INDEX = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)


class TestPolyNSDModelJKNet:
    def _model(self, normalize_output: bool = False) -> PolyNSDModel:
        set_seed(42)
        return PolyNSDModel(
            in_channels=8,
            out_channels=3,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=3,
            degree=2,
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
