# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


from typing import Any

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.dnsd.dnsd_layers import DNSDConv
from sheaf_mpnn.dnsd.dnsd_model import DNSDModel, DNSDVariant
from sheaf_mpnn.sheaf_adj.sheaf_adj_model import SheafAdjModel, SheafAdjVariant
from tests.testing import set_seed


@pytest.mark.parametrize("variant", list(DNSDVariant))
@pytest.mark.parametrize("odd", [False, True], ids=["elu", "tanh"])
@pytest.mark.parametrize("gate", [False, True], ids=["gate_off", "gate_on"])
class TestDNSDModel:
    @pytest.fixture
    def setup(self, variant, odd, gate):
        set_seed(42)
        in_channels, out_channels = 10, 3
        num_nodes = 15
        model = DNSDModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=2,
            variant=variant,
            odd=odd,
            gate=gate,
        )
        x = torch.randn(num_nodes, in_channels)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        return model, x, edge_index

    def test_model_forward_shape(self, setup):
        model, x, edge_index = setup
        out = model(x, edge_index)
        assert out.shape == (x.size(0), model.out_channels)
        assert not torch.isnan(out).any()
        assert torch.isfinite(out).all()

    def test_model_gradient_flow(self, setup):
        model, x, edge_index = setup
        out = model(x, edge_index)
        loss = out.pow(2).sum()
        loss.backward()

        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0
        for layer in model.layers:
            assert layer.sheaf_adj_layer.W1.grad is not None
            assert layer.sheaf_adj_layer.W1.grad.abs().sum() > 0

    def test_model_determinism(self, setup):
        model, x, edge_index = setup
        model.eval()
        with torch.no_grad():
            out1 = model(x, edge_index)
            out2 = model(x, edge_index)
        torch.testing.assert_close(out1, out2)

    def test_isolated_graph_stability(self, setup):
        model, x, _ = setup
        model.eval()
        edge_index = torch.empty((2, 0), dtype=torch.long)
        with torch.no_grad():
            out = model(x, edge_index)
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()


def test_model_invalid_stalk_dim():
    with pytest.raises(ValueError, match="stalk_dim must be positive"):
        DNSDModel(in_channels=10, out_channels=3, stalk_dim=0)


def test_model_invalid_num_layers():
    with pytest.raises(ValueError, match="must have at least one"):
        DNSDModel(in_channels=10, out_channels=3, num_layers=0)


def test_hidden_dim_must_be_positive():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        DNSDModel(in_channels=10, out_channels=3, hidden_dim=0)


def test_model_jknet_forward():
    set_seed(42)
    model = DNSDModel(
        in_channels=10,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=3,
        jknet=True,
    )
    x = torch.randn(12, 10)
    edge_index = erdos_renyi_graph(12, edge_prob=0.4)
    out = model(x, edge_index)
    assert out.shape == (12, 3)
    assert not torch.isnan(out).any()


def test_model_add_eps_forward():
    set_seed(42)
    model = DNSDModel(
        in_channels=10,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        add_eps=True,
    )
    x = torch.randn(12, 10)
    edge_index = erdos_renyi_graph(12, edge_prob=0.4)
    out = model(x, edge_index)
    assert out.shape == (12, 3)
    assert not torch.isnan(out).any()


def test_learn_alpha_false_freezes_alpha():
    model = DNSDModel(in_channels=10, out_channels=3, learn_alpha=False)
    for layer in model.layers:
        assert isinstance(layer, DNSDConv)
        assert not layer.sheaf_adj_layer.alpha.requires_grad


def test_layer_norm_false_disables_norms():
    model = DNSDModel(in_channels=10, out_channels=3, layer_norm=False)
    assert model.norms is None


def test_reset_parameters_changes_output():
    set_seed(42)
    model = DNSDModel(
        in_channels=10,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        second_linear=True,
        jknet=True,
        gate=True,
    ).eval()
    x = torch.randn(15, 10)
    edge_index = erdos_renyi_graph(15, edge_prob=0.4)
    before = model(x, edge_index).clone()
    model.reset_parameters()
    assert not torch.allclose(before, model(x, edge_index))


def test_reset_parameters_resets_layer_norm():
    set_seed(42)
    model = DNSDModel(in_channels=10, out_channels=3, stalk_dim=2, hidden_dim=4)
    assert model.norms is not None
    norm = model.norms[0]
    assert isinstance(norm, torch.nn.LayerNorm)
    with torch.no_grad():
        norm.weight.fill_(3.0)
        norm.bias.fill_(3.0)
    model.reset_parameters()
    torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
    torch.testing.assert_close(norm.bias, torch.zeros_like(norm.bias))


def test_forward_validates_x_shape():
    model = DNSDModel(in_channels=10, out_channels=3)
    with pytest.raises(ValueError, match="x must have shape"):
        model(torch.randn(5, 7), torch.empty((2, 0), dtype=torch.long))


def test_forward_validates_edge_index_shape():
    model = DNSDModel(in_channels=10, out_channels=3)
    with pytest.raises(ValueError, match="edge_index must have shape"):
        model(torch.randn(5, 10), torch.zeros((3, 4), dtype=torch.long))


def test_odd_false_gate_false_matches_sheaf_adj():
    # Core minimal-diff invariant at the model level: DNSDModel(odd=False,
    # gate=False, layer_norm=False) must be numerically identical to
    # SheafAdjModel given the same seed/weights.
    set_seed(7)
    kwargs: dict[str, Any] = dict(
        in_channels=10,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        variant=SheafAdjVariant.DIAGONAL,
    )
    baseline = SheafAdjModel(**kwargs)
    dnsd = DNSDModel(**kwargs, layer_norm=False)
    dnsd.encoder.load_state_dict(baseline.encoder.state_dict())
    for dnsd_layer, baseline_layer in zip(dnsd.layers, baseline.layers, strict=True):
        assert isinstance(dnsd_layer, DNSDConv)
        assert isinstance(baseline_layer, torch.nn.Module)
        dnsd_layer.sheaf_adj_layer.load_state_dict(baseline_layer.state_dict())
    dnsd.decoder.load_state_dict(baseline.decoder.state_dict())
    dnsd.eval()
    baseline.eval()

    x = torch.randn(12, 10)
    edge_index = erdos_renyi_graph(12, edge_prob=0.4)
    with torch.no_grad():
        out_baseline = baseline(x, edge_index)
        out_dnsd = dnsd(x, edge_index)
    torch.testing.assert_close(out_baseline, out_dnsd, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("flag", ["odd", "gate", "layer_norm"])
def test_each_flag_changes_output_vs_baseline(flag):
    # Guards against a silently no-op flag: with identical weights/seed, each
    # of odd/gate/layer_norm toggled on alone must actually change the
    # forward output relative to the all-off baseline. gate in particular is
    # zero-initialised (sigmoid(0)=0.5 at init), so this also confirms a
    # uniform 0.5x rescale of the diffusion term is not silently cancelled
    # out by the skip connection.
    set_seed(7)
    kwargs: dict[str, Any] = dict(
        in_channels=10,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        variant=DNSDVariant.DIAGONAL,
    )
    baseline = DNSDModel(**kwargs, odd=False, gate=False, layer_norm=False)

    toggled = DNSDModel(**kwargs, **{flag: True})
    toggled.encoder.load_state_dict(baseline.encoder.state_dict())
    for toggled_layer, baseline_layer in zip(
        toggled.layers, baseline.layers, strict=True
    ):
        assert isinstance(toggled_layer, DNSDConv)
        assert isinstance(baseline_layer, DNSDConv)
        toggled_layer.sheaf_adj_layer.load_state_dict(
            baseline_layer.sheaf_adj_layer.state_dict()
        )
    toggled.decoder.load_state_dict(baseline.decoder.state_dict())
    baseline.eval()
    toggled.eval()

    x = torch.randn(12, 10)
    edge_index = erdos_renyi_graph(12, edge_prob=0.4)
    with torch.no_grad():
        out_baseline = baseline(x, edge_index)
        out_toggled = toggled(x, edge_index)
    assert not torch.allclose(out_baseline, out_toggled, atol=1e-6)
