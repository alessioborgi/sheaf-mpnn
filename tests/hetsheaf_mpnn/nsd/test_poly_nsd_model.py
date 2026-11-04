# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import random
from typing import Any

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.nsd.poly_nsd_layers import (
    DiagonalPolyHetNSDConv,
    GeneralPolyHetNSDConv,
    LowRankPolyHetNSDConv,
    OrthogonalPolyHetNSDConv,
)
from hetsheaf_mpnn.nsd.poly_nsd_model import PolyHetNSDModel, PolyHetNSDVariant


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


NUM_NODES = 12
IN_CHANNELS = 16
OUT_CHANNELS = 3
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4


def _graph():
    return erdos_renyi_graph(NUM_NODES, edge_prob=0.4)


def _types(edge_index):
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    return nt, et


def _model(
    variant: PolyHetNSDVariant = PolyHetNSDVariant.GENERAL, **kwargs: Any
) -> PolyHetNSDModel:
    defaults: dict[str, Any] = {
        "in_channels": IN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "stalk_dim": 2,
        "hidden_dim": 4,
        "num_layers": 2,
        "variant": variant,
        "degree": 2,
        "num_node_types": NUM_NODE_TYPES,
        "num_edge_types": NUM_EDGE_TYPES,
    }
    defaults.update(kwargs)
    return PolyHetNSDModel(**defaults)


# ---------------------------------------------------------------------------
# Parametrised forward / gradient tests for all variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", list(PolyHetNSDVariant))
class TestPolyHetNSDModel:
    """Shared forward and gradient tests for every PolyHetNSD variant."""

    @pytest.fixture
    def setup(self, variant):
        set_seed(42)
        edge_index = _graph()
        nt, et = _types(edge_index)
        x = torch.randn(NUM_NODES, IN_CHANNELS)
        model = _model(
            variant,
            use_features=True,
            use_node_types=True,
            use_edge_types=True,
        )
        return model, x, edge_index, nt, et

    def test_construction_and_forward(self, setup):
        """Forward must return [N, out_channels] with no NaN."""
        model, x, edge_index, nt, et = setup
        out = model(x, edge_index, nt, et)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert not torch.isnan(out).any()

    def test_forward_without_types(self, setup):
        """Model works when the type tensors are omitted (fallback to zeros)."""
        model, x, edge_index, _, _ = setup
        out = model(x, edge_index)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert not torch.isnan(out).any()

    def test_gradient_backprop(self, setup):
        """Gradients must reach the encoder and the first poly layer.

        theta_raw is perturbed before the backward pass so the polynomial
        weights are unequal; at the zero initialisation the Chebyshev recurrence
        can cancel the direct z contribution for even degrees, leaving W1/W2
        gradients numerically zero despite the graph being connected.
        """
        model, x, edge_index, nt, et = setup
        torch.manual_seed(1)
        for layer in model.layers:
            layer.theta_raw.data = torch.randn_like(layer.theta_raw)
        out = model(x, edge_index, nt, et)
        out.pow(2).mean().backward()
        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0
        assert model.layers[0].W1.grad is not None
        assert model.layers[0].W1.grad.abs().sum() > 0
        assert model.layers[0].theta_raw.grad is not None
        assert model.layers[0].theta_raw.grad.abs().sum() > 0

    def test_eval_determinism(self, setup):
        """eval() must produce identical outputs on consecutive forward passes."""
        model, x, edge_index, nt, et = setup
        model.eval()
        with torch.no_grad():
            out1 = model(x, edge_index, nt, et)
            out2 = model(x, edge_index, nt, et)
        torch.testing.assert_close(out1, out2)

    def test_reset_parameters(self, setup):
        """reset_parameters must run over encoder, all layers, and decoder."""
        model, *_ = setup
        for layer in model.layers:
            layer.theta_raw.data.fill_(99.0)
        model.reset_parameters()
        for layer in model.layers:
            assert layer.theta_raw.abs().max().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_stalk_dim():
    with pytest.raises(ValueError, match="stalk_dim must be positive"):
        PolyHetNSDModel(IN_CHANNELS, OUT_CHANNELS, stalk_dim=0)


def test_invalid_hidden_dim():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        PolyHetNSDModel(IN_CHANNELS, OUT_CHANNELS, hidden_dim=0)


def test_invalid_num_layers():
    with pytest.raises(ValueError, match="at least one PolyHetNSD layer"):
        PolyHetNSDModel(IN_CHANNELS, OUT_CHANNELS, num_layers=0)


def test_invalid_degree():
    with pytest.raises(ValueError, match="degree must be at least 1"):
        PolyHetNSDModel(IN_CHANNELS, OUT_CHANNELS, degree=0)


def test_invalid_rank():
    with pytest.raises(ValueError, match="rank must be positive"):
        PolyHetNSDModel(
            IN_CHANNELS, OUT_CHANNELS, variant=PolyHetNSDVariant.LOW_RANK, rank=0
        )


# ---------------------------------------------------------------------------
# Enum to layer class / kwargs mapping
# ---------------------------------------------------------------------------


def test_enum_to_layer_mapping():
    """PolyHetNSDVariant must map to the correct polynomial het layer classes."""
    assert PolyHetNSDVariant.DIAGONAL.layer_class == DiagonalPolyHetNSDConv
    assert PolyHetNSDVariant.GENERAL.layer_class == GeneralPolyHetNSDConv
    assert PolyHetNSDVariant.ORTHOGONAL.layer_class == OrthogonalPolyHetNSDConv
    assert PolyHetNSDVariant.GENERAL_ATTENTION.layer_class == GeneralPolyHetNSDConv
    assert PolyHetNSDVariant.ORTHOGONAL_ATTENTION.layer_class == (
        OrthogonalPolyHetNSDConv
    )
    assert PolyHetNSDVariant.LOW_RANK.layer_class == LowRankPolyHetNSDConv


def test_enum_layer_kwargs():
    """layer_kwargs must set use_attention only for the attention variants."""
    assert PolyHetNSDVariant.DIAGONAL.layer_kwargs == {}
    assert PolyHetNSDVariant.LOW_RANK.layer_kwargs == {}
    assert PolyHetNSDVariant.GENERAL.layer_kwargs == {"use_attention": False}
    assert PolyHetNSDVariant.ORTHOGONAL.layer_kwargs == {"use_attention": False}
    assert PolyHetNSDVariant.GENERAL_ATTENTION.layer_kwargs == {"use_attention": True}
    assert PolyHetNSDVariant.ORTHOGONAL_ATTENTION.layer_kwargs == {
        "use_attention": True
    }


def test_attention_variant_builds_attention_layers():
    """The GENERAL_ATTENTION variant must construct attention-enabled layers."""
    model = _model(PolyHetNSDVariant.GENERAL_ATTENTION)
    for layer in model.layers:
        assert isinstance(layer, GeneralPolyHetNSDConv)
        assert layer.use_attention


# ---------------------------------------------------------------------------
# Variant-specific constructor kwargs propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orth_strategy", ["cayley", "fasth"])
def test_orth_strategy_forwarded(orth_strategy):
    set_seed(42)
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = _model(PolyHetNSDVariant.ORTHOGONAL, orth_strategy=orth_strategy)
    out = model(x, edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert not torch.isnan(out).any()


def test_low_rank_rank_forwarded():
    set_seed(42)
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = _model(PolyHetNSDVariant.LOW_RANK, rank=2)
    assert model.rank == 2
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_ensemble_model():
    set_seed(42)
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    model = _model(
        PolyHetNSDVariant.DIAGONAL,
        use_features=True,
        ensemble=True,
    )
    out = model(x, edge_index, None, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


# ---------------------------------------------------------------------------
# Hyperparameter storage
# ---------------------------------------------------------------------------


def test_model_stores_hyperparameters():
    model = _model(stalk_dim=3, hidden_dim=8)
    assert model.stalk_dim == 3
    assert model.hidden_dim == 8
    assert model.out_channels == OUT_CHANNELS


# ---------------------------------------------------------------------------
# JKNet and output normalisation
# ---------------------------------------------------------------------------


class TestPolyHetNSDModelJKNet:
    def _jk_model(self, normalize_output=True):
        set_seed(42)
        return _model(
            PolyHetNSDVariant.GENERAL,
            num_layers=3,
            jknet=True,
            normalize_output=normalize_output,
        )

    def test_jk_attribute_created(self):
        model = self._jk_model()
        assert hasattr(model, "jk")

    def test_decoder_expanded_for_jknet(self):
        model = self._jk_model()
        context_dim = model.stalk_dim * model.hidden_dim
        assert model.decoder.in_features == 3 * context_dim

    def test_forward_shape(self):
        model = self._jk_model()
        edge_index = _graph()
        nt, et = _types(edge_index)
        x = torch.randn(NUM_NODES, IN_CHANNELS)
        out = model(x, edge_index, nt, et)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert torch.isfinite(out).all()

    def test_forward_shape_no_normalize(self):
        model = self._jk_model(normalize_output=False)
        edge_index = _graph()
        x = torch.randn(NUM_NODES, IN_CHANNELS)
        out = model(x, edge_index)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        model = self._jk_model()
        edge_index = _graph()
        x = torch.randn(NUM_NODES, IN_CHANNELS)
        out = model(x, edge_index)
        out.pow(2).mean().backward()
        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0


def test_no_normalize_output():
    set_seed(42)
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = _model(normalize_output=False)
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Dropout behaviour
# ---------------------------------------------------------------------------


def test_dropout_probabilities_stored():
    model = _model(input_dropout=0.3, dropout=0.5)
    assert model.input_dropout_layer.p == pytest.approx(0.3)
    assert model.dropout_layer.p == pytest.approx(0.5)
