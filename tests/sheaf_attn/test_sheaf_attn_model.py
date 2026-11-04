# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.sheaf_attn.sheaf_attn_model import SheafAttnModel, SheafAttnVariant
from tests.testing import set_seed


@pytest.mark.parametrize("variant", list(SheafAttnVariant))
class TestSheafAttnModel:
    @pytest.fixture
    def setup(self, variant):
        set_seed(42)
        num_nodes, in_channels, out_channels = 15, 10, 3
        d, hidden_dim, num_layers = 2, 4, 2

        model = SheafAttnModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=d,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            variant=variant,
        )

        x = torch.randn(num_nodes, in_channels)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)

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

    def test_multi_head(self, setup, variant):
        # Verify num_heads > 1 produces valid outputs with correct shape.
        set_seed(42)
        num_nodes, in_channels, out_channels = 10, 6, 2
        model = SheafAttnModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=1,
            variant=variant,
            num_heads=4,
        )
        x = torch.randn(num_nodes, in_channels)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        out = model(x, edge_index)
        assert out.shape == (num_nodes, out_channels)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# JKNet
# ---------------------------------------------------------------------------


_JKNET_EDGE_INDEX = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)


class TestSheafAttnModelJKNet:
    def _model(self, normalize_output: bool = False) -> SheafAttnModel:
        set_seed(42)
        return SheafAttnModel(
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


def test_sheaf_attn_model_stores_hyperparameters():
    """Verify SheafAttnModel stores passed hyperparameters as attributes."""
    model = SheafAttnModel(in_channels=10, out_channels=2, hidden_dim=32, stalk_dim=3)
    assert model.hidden_dim == 32
    assert model.stalk_dim == 3
    assert model.out_channels == 2


def test_sheaf_attn_model_validation():
    """Verify SheafAttnModel enforces valid hyperparameter ranges."""
    with pytest.raises(ValueError, match="stalk_dim must be positive"):
        SheafAttnModel(in_channels=10, out_channels=2, stalk_dim=0)

    with pytest.raises(ValueError, match="must have at least one SheafAttn layer"):
        SheafAttnModel(in_channels=10, out_channels=2, num_layers=0)

    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        SheafAttnModel(in_channels=10, out_channels=2, hidden_dim=0)

    with pytest.raises(ValueError, match="num_heads must be positive"):
        SheafAttnModel(in_channels=10, out_channels=2, num_heads=0)


def test_variant_residual_property():
    """Verify the residual property on SheafAttnVariant is correct."""
    assert SheafAttnVariant.SHEAFAN.residual is False
    assert SheafAttnVariant.RES_SHEAFAN.residual is True


def test_variants_produce_different_outputs():
    """SHEAFAN and RES_SHEAFAN must produce different outputs for the same weights."""
    set_seed(42)
    num_nodes, in_channels, out_channels = 10, 6, 2
    common_kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
    )
    model_a = SheafAttnModel(
        **common_kwargs,  # ty: ignore[invalid-argument-type]
        variant=SheafAttnVariant.SHEAFAN,
    )
    model_b = SheafAttnModel(
        **common_kwargs,  # ty: ignore[invalid-argument-type]
        variant=SheafAttnVariant.RES_SHEAFAN,
    )
    model_b.load_state_dict(model_a.state_dict())

    x = torch.randn(num_nodes, in_channels)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(x, edge_index)
        out_b = model_b(x, edge_index)
    assert not torch.allclose(out_a, out_b)


def test_reset_parameters_runs():
    """reset_parameters() must complete without error."""
    model = SheafAttnModel(
        in_channels=6, out_channels=2, stalk_dim=2, hidden_dim=4, num_layers=1
    )
    model.reset_parameters()


def test_enum_to_residual_mapping():
    """SheafAttnVariant enum values and their residual flags."""
    assert len(list(SheafAttnVariant)) == 2
    assert SheafAttnVariant.SHEAFAN.residual is False
    assert SheafAttnVariant.RES_SHEAFAN.residual is True


# ---------------------------------------------------------------------------
# Structural / architectural
# ---------------------------------------------------------------------------


def test_model_encoder_maps_to_stalk_space():
    """Encoder output has size d*hidden_dim and reshapes cleanly to [N, d, f]."""
    d, hidden_dim = 4, 8
    model = SheafAttnModel(
        in_channels=10, out_channels=3, stalk_dim=d, hidden_dim=hidden_dim, num_layers=1
    )
    x = torch.randn(15, 10)
    encoded = model.encoder(x)
    assert encoded.shape == (15, d * hidden_dim)
    assert encoded.view(-1, d, hidden_dim).shape == (15, d, hidden_dim)


def test_model_decoder_maps_from_stalk_space():
    """Decoder takes flattened stalk [N, d*f] and returns [N, out_channels]."""
    d, hidden_dim, out_channels = 4, 8, 3
    model = SheafAttnModel(
        in_channels=10,
        out_channels=out_channels,
        stalk_dim=d,
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    stalk_flat = torch.randn(15, d * hidden_dim)
    assert model.decoder(stalk_flat).shape == (15, out_channels)


def test_model_num_layers_stored():
    """len(model.layers) must equal num_layers for all valid values."""
    for num_layers in [1, 2, 4]:
        model = SheafAttnModel(
            in_channels=10,
            out_channels=3,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=num_layers,
        )
        assert len(model.layers) == num_layers


def test_model_num_heads_propagated_to_layers():
    """Each layer's num_heads attribute must match the constructor argument."""
    num_heads = 3
    model = SheafAttnModel(
        in_channels=4,
        out_channels=2,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        num_heads=num_heads,
    )
    for layer in model.layers:
        assert layer.num_heads == num_heads


def test_model_layers_share_residual_flag():
    """All layers carry the residual flag that matches the chosen variant."""
    for variant in SheafAttnVariant:
        model = SheafAttnModel(
            in_channels=4,
            out_channels=2,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=2,
            variant=variant,
        )
        for layer in model.layers:
            assert layer.residual == variant.residual


@pytest.mark.parametrize("orth_strategy", ["cayley", "fasth"])
def test_orth_strategy_model_forward(orth_strategy):
    """Both orth_strategy values produce finite outputs from SheafAttnModel."""
    set_seed(42)
    num_nodes, in_channels, out_channels = 10, 6, 2
    model = SheafAttnModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=3,
        hidden_dim=8,
        num_layers=2,
        orth_strategy=orth_strategy,
    )
    x = torch.randn(num_nodes, in_channels)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
    out = model(x, edge_index)
    assert out.shape == (num_nodes, out_channels)
    assert torch.isfinite(out).all()


def test_orth_strategy_propagated_to_layers():
    """orth_strategy passed to SheafAttnModel must reach all conv layers."""
    model = SheafAttnModel(
        in_channels=4,
        out_channels=2,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        orth_strategy="fasth",
    )
    for layer in model.layers:
        assert layer.orth_strategy == "fasth"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_model_single_node():
    """A single-node graph receives only a self-loop; must produce finite output."""
    set_seed(42)
    model = SheafAttnModel(
        in_channels=4, out_channels=2, stalk_dim=2, hidden_dim=4, num_layers=1
    )
    x = torch.randn(1, 4)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_model_d1_edge_case():
    """stalk_dim=1 (Cayley maps are always identity) produces valid outputs."""
    set_seed(42)
    num_nodes, in_channels, out_channels = 8, 4, 2
    model = SheafAttnModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=1,
        hidden_dim=4,
        num_layers=1,
    )
    x = torch.randn(num_nodes, in_channels)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
    out = model(x, edge_index)
    assert out.shape == (num_nodes, out_channels)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Behavioural
# ---------------------------------------------------------------------------


def test_model_depth_changes_output():
    """1-layer and 3-layer models (independent weights) produce different outputs."""
    set_seed(42)
    num_nodes, in_channels, out_channels = 10, 6, 2
    x = torch.randn(num_nodes, in_channels)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)

    model1 = SheafAttnModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
    )
    model3 = SheafAttnModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=3,
    )
    model1.eval()
    model3.eval()
    with torch.no_grad():
        out1 = model1(x, edge_index)
        out3 = model3(x, edge_index)
    assert out1.shape == out3.shape == (num_nodes, out_channels)
    assert not torch.allclose(out1, out3)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

_DROPOUT_EDGE_INDEX = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)


class TestSheafAttnModelDropout:
    def test_dropout_probabilities_stored(self):
        model = SheafAttnModel(
            in_channels=10, out_channels=3, input_dropout=0.3, dropout=0.5
        )
        assert model.input_dropout_layer.p == pytest.approx(0.3)
        assert model.dropout_layer.p == pytest.approx(0.5)

    def test_eval_suppresses_dropout(self):
        """eval() must make consecutive forward passes identical with high dropout."""
        torch.manual_seed(0)
        x = torch.randn(4, 10)
        model = SheafAttnModel(
            in_channels=10, out_channels=3, input_dropout=0.9, dropout=0.9
        )
        model.eval()
        with torch.no_grad():
            out1 = model(x, _DROPOUT_EDGE_INDEX)
            out2 = model(x, _DROPOUT_EDGE_INDEX)
        torch.testing.assert_close(out1, out2)

    def test_train_mode_is_stochastic(self):
        """train() with high dropout must produce different outputs each call."""
        torch.manual_seed(0)
        x = torch.randn(4, 10)
        model = SheafAttnModel(
            in_channels=10, out_channels=3, input_dropout=0.9, dropout=0.9
        )
        model.train()
        with torch.no_grad():
            out1 = model(x, _DROPOUT_EDGE_INDEX)
            out2 = model(x, _DROPOUT_EDGE_INDEX)
        assert not torch.allclose(out1, out2)


def test_model_output_differs_by_num_heads():
    """Identical weights but different num_heads produce different outputs."""
    set_seed(42)
    num_nodes, in_channels, out_channels = 8, 4, 2
    x = torch.randn(num_nodes, in_channels)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)

    model1 = SheafAttnModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
    )
    model4 = SheafAttnModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
        num_heads=4,
    )
    model1.eval()
    model4.eval()
    with torch.no_grad():
        out1 = model1(x, edge_index)
        out4 = model4(x, edge_index)
    assert out1.shape == out4.shape == (num_nodes, out_channels)
    assert not torch.allclose(out1, out4)
