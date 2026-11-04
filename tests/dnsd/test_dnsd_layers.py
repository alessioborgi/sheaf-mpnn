# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.dnsd.dnsd_layers import DNSDConv
from sheaf_mpnn.sheaf_adj.sheaf_adj_layers import (
    DiagonalSheafAdjConv,
    GeneralSheafAdjConv,
    LowRankSheafAdjConv,
    OrthogonalSheafAdjConv,
)
from tests.testing import set_seed


@pytest.mark.parametrize(
    "model_class,layer_kwargs",
    [
        (DiagonalSheafAdjConv, {}),
        (GeneralSheafAdjConv, {}),
        (OrthogonalSheafAdjConv, {}),
        (GeneralSheafAdjConv, {"use_attention": True}),
        (OrthogonalSheafAdjConv, {"use_attention": True}),
        (OrthogonalSheafAdjConv, {"orth_strategy": "fasth"}),
        (LowRankSheafAdjConv, {"rank": 2}),
    ],
    ids=[
        "diagonal",
        "general",
        "orthogonal",
        "general_attention",
        "orthogonal_attention",
        "orthogonal_fasth",
        "low_rank",
    ],
)
@pytest.mark.parametrize("gate", [False, True], ids=["gate_off", "gate_on"])
class TestDNSDConvVariants:
    @pytest.fixture
    def setup(self, model_class, layer_kwargs, gate):
        set_seed(42)
        num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)

        sheaf_adj_layer = model_class(
            d, stalk_features, hidden_dim, context_dim=context_dim, **layer_kwargs
        )
        sheaf_adj_layer.residual = False
        conv = DNSDConv(
            sheaf_adj_layer, stalk_dim=d, in_channels=stalk_features, gate=gate
        )
        return conv, x_feat, x_stalk, edge_index

    def test_output_shape(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()
        assert torch.isfinite(out).all()

    def test_isolated_node_stability(self, setup):
        conv, x_feat, x_stalk, _ = setup
        edge_index = torch.empty((2, 0), dtype=torch.long)
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()

    def test_alpha_zero_identity(self, setup):
        # Zeroing alpha zeroes the wrapped layer's diffusion term, so the
        # gate (which only scales that term) leaves the skip untouched
        # regardless of gate=True/False -- output must still equal x_stalk.
        conv, x_feat, x_stalk, edge_index = setup
        conv.sheaf_adj_layer.alpha.data.zero_()
        out = conv(x_feat, x_stalk, edge_index)
        torch.testing.assert_close(out, x_stalk)

    def test_gradient_flow(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        loss = out.pow(2).sum()
        loss.backward()

        assert conv.sheaf_adj_layer.W1.grad is not None
        assert conv.sheaf_adj_layer.W1.grad.abs().sum() > 0
        if conv.gate:
            assert conv.gate_lin.weight.grad is not None

    def test_permutation_invariance(self, setup, model_class):
        conv, x_feat, x_stalk, edge_index = setup
        conv.eval()
        out_orig = conv(x_feat, x_stalk, edge_index)

        perm = torch.randperm(x_feat.size(0))
        rev_perm = torch.argsort(perm)
        row, col = edge_index
        mapping = torch.zeros(x_feat.size(0), dtype=torch.long)
        mapping[perm] = torch.arange(x_feat.size(0))
        edge_index_p = torch.stack([mapping[row], mapping[col]], dim=0)

        out_perm = conv(x_feat[perm], x_stalk[perm], edge_index_p)
        tol = 1e-4 if model_class is LowRankSheafAdjConv else 1e-5
        torch.testing.assert_close(out_orig, out_perm[rev_perm], atol=tol, rtol=tol)


def test_gate_starts_at_half():
    # Zero-initialised gate_lin -> sigmoid(0) = 0.5 for every stalk at init.
    set_seed(0)
    num_nodes, d, f, h = 6, 3, 4, 4
    ctx = d * f
    x_feat = torch.randn(num_nodes, ctx)
    x_stalk = torch.randn(num_nodes, d, f)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    layer = DiagonalSheafAdjConv(d, f, h, context_dim=ctx)
    layer.residual = False
    gated = DNSDConv(layer, stalk_dim=d, in_channels=f, gate=True)
    ungated_layer = DiagonalSheafAdjConv(d, f, h, context_dim=ctx)
    ungated_layer.load_state_dict(layer.state_dict())
    ungated_layer.residual = False
    ungated = DNSDConv(ungated_layer, stalk_dim=d, in_channels=f, gate=False)

    out_gated = gated(x_feat, x_stalk, edge_index)
    out_ungated = ungated(x_feat, x_stalk, edge_index)
    msg_ungated = out_ungated - x_stalk  # bare diffusion term (no skip added twice)
    expected = x_stalk + 0.5 * msg_ungated
    torch.testing.assert_close(out_gated, expected, atol=1e-6, rtol=1e-6)


def test_gate_computed_from_x_stalk_only():
    # gate_lin takes x_stalk alone (shape [N, d, in_channels] ->
    # Linear(in_channels, 1)), not a concatenation with the diffusion term or
    # the raw aggregation.
    set_seed(0)
    num_nodes, d, f, h = 6, 2, 4, 4
    ctx = d * f
    x_feat = torch.randn(num_nodes, ctx)
    x_stalk = torch.randn(num_nodes, d, f)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    layer = DiagonalSheafAdjConv(d, f, h, context_dim=ctx)
    layer.residual = False
    conv = DNSDConv(layer, stalk_dim=d, in_channels=f, gate=True)
    assert conv.gate_lin.weight.shape == (1, f)
    # non-zero: gate must vary with input
    torch.nn.init.normal_(conv.gate_lin.weight, std=1.0)

    out = conv(x_feat, x_stalk, edge_index)

    msg_bare = layer(x_feat, x_stalk, edge_index)
    expected_gate = torch.sigmoid(
        conv.gate_lin(x_stalk.reshape(num_nodes * d, f))
    ).view(num_nodes, d, 1)
    torch.testing.assert_close(
        out, x_stalk + expected_gate * msg_bare, atol=1e-6, rtol=1e-6
    )


def test_odd_uses_tanh_activation():
    # odd is applied by the caller overwriting sheaf_adj_layer.sigma, exactly
    # like SheafAttnConv does for its own activation -- verify DNSDConv
    # actually uses whatever `sigma` the wrapped layer carries.
    set_seed(0)
    num_nodes, d, f, h = 6, 2, 4, 4
    ctx = d * f
    x_feat = torch.randn(num_nodes, ctx)
    x_stalk = torch.randn(num_nodes, d, f)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    layer = DiagonalSheafAdjConv(d, f, h, context_dim=ctx)
    layer.residual = False
    assert isinstance(layer.sigma, torch.nn.ELU)
    layer.sigma = torch.nn.Tanh()
    conv = DNSDConv(layer, stalk_dim=d, in_channels=f)
    out = conv(x_feat, x_stalk, edge_index)
    assert torch.isfinite(out).all()
    # tanh is bounded in (-1, 1), so the bare diffusion term (out - skip) must be too.
    msg = out - x_stalk
    assert (msg.abs() <= 1.0 + 1e-6).all()


def test_add_eps_forward():
    set_seed(0)
    num_nodes, d, f, h = 6, 2, 4, 4
    ctx = d * f
    x_feat = torch.randn(num_nodes, ctx)
    x_stalk = torch.randn(num_nodes, d, f)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    layer = DiagonalSheafAdjConv(d, f, h, context_dim=ctx)
    layer.residual = False
    conv = DNSDConv(layer, stalk_dim=d, in_channels=f, add_eps=True)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


def test_requires_residual_false():
    layer = DiagonalSheafAdjConv(stalk_dim=4, in_channels=16, hidden_dim=8)
    assert layer.residual  # default is True
    with pytest.raises(ValueError, match="residual=False"):
        DNSDConv(layer, stalk_dim=4, in_channels=16)


def test_odd_false_gate_false_matches_sheaf_adj():
    # Core minimal-diff invariant: DNSDConv(odd=False, gate=False) wrapping a
    # residual=False layer must reproduce a standalone residual=True layer
    # with identical weights exactly.
    set_seed(0)
    num_nodes, d, f, h = 8, 3, 5, 6
    ctx = d * f
    x_feat = torch.randn(num_nodes, ctx)
    x_stalk = torch.randn(num_nodes, d, f)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)

    baseline = DiagonalSheafAdjConv(d, f, h, context_dim=ctx)
    wrapped = DiagonalSheafAdjConv(d, f, h, context_dim=ctx)
    wrapped.load_state_dict(baseline.state_dict())
    wrapped.residual = False

    conv = DNSDConv(wrapped, stalk_dim=d, in_channels=f)

    out_baseline = baseline(x_feat, x_stalk, edge_index)
    out_dnsd = conv(x_feat, x_stalk, edge_index)
    torch.testing.assert_close(out_baseline, out_dnsd, atol=1e-6, rtol=1e-6)
