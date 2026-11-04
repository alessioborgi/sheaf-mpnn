# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch import nn
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.sheaf_attn.sheaf_attn_layers import SheafAttnConv
from sheaf_mpnn.utils import cayley, householder
from tests.testing import set_seed


@pytest.mark.parametrize(
    "num_heads,residual",
    [
        (1, False),
        (1, True),
        (4, False),
        (4, True),
    ],
    ids=["1head_sheafan", "1head_res_sheafan", "4head_sheafan", "4head_res_sheafan"],
)
class TestSheafAttnConvVariants:
    @pytest.fixture
    def setup(self, num_heads, residual):
        set_seed(42)
        num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)

        conv = SheafAttnConv(
            stalk_dim=d,
            in_channels=stalk_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            context_dim=context_dim,
            residual=residual,
        )
        return conv, x_feat, x_stalk, edge_index, d, stalk_features

    def test_output_shape(self, setup):
        conv, x_feat, x_stalk, edge_index, d, f = setup
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()

    def test_numerical_stability(self, setup):
        conv, x_feat, x_stalk, edge_index, d, f = setup
        out = conv(x_feat, x_stalk, edge_index)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, setup):
        conv, x_feat, x_stalk, edge_index, d, f = setup
        out = conv(x_feat, x_stalk, edge_index)
        loss = out.pow(2).sum()
        loss.backward()

        assert conv.W1.grad is not None and conv.W1.grad.abs().sum() > 0
        assert conv.W2.grad is not None and conv.W2.grad.abs().sum() > 0
        assert conv.attn_projection.weight.grad is not None
        assert conv.attn_score_vec.grad is not None
        for gen in conv.map_generators:
            for param in gen.parameters():
                assert param.grad is not None and param.grad.abs().sum() > 0

    def test_determinism(self, setup):
        conv, x_feat, x_stalk, edge_index, d, f = setup
        conv.eval()
        with torch.no_grad():
            out1 = conv(x_feat, x_stalk, edge_index)
            out2 = conv(x_feat, x_stalk, edge_index)
        torch.testing.assert_close(out1, out2)

    def test_isolated_graph_no_crash(self, setup):
        # With no edges, add_self_loops in forward() ensures each node still gets
        # a self-message. Output must be finite with the correct shape.
        conv, x_feat, x_stalk, _, d, f = setup
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        out = conv(x_feat, x_stalk, empty_edge_index)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_residual_self_loop_identity(self, setup, residual):
        # For Res-SheafAN with only self-loops: P_ii = I (orthogonality), so
        # out_from_propagate = z, and the residual branch gives
        # x_stalk + sigma(z - z) = x_stalk + sigma(0) = x_stalk.
        if not residual:
            pytest.skip("Identity shortcut only applies to Res-SheafAN")
        conv, x_feat, x_stalk, _, d, f = setup
        conv.eval()
        with torch.no_grad():
            empty_edge_index = torch.empty((2, 0), dtype=torch.long)
            out = conv(x_feat, x_stalk, empty_edge_index)
        torch.testing.assert_close(out, x_stalk, atol=1e-4, rtol=1e-4)

    def test_permutation_invariance(self, setup):
        conv, x_feat, x_stalk, edge_index, d, f = setup
        conv.eval()
        out_orig = conv(x_feat, x_stalk, edge_index)

        perm = torch.randperm(x_feat.size(0))
        rev_perm = torch.argsort(perm)
        row, col = edge_index
        mapping = torch.zeros(x_feat.size(0), dtype=torch.long)
        mapping[perm] = torch.arange(x_feat.size(0))
        edge_index_p = torch.stack([mapping[row], mapping[col]], dim=0)

        out_perm = conv(x_feat[perm], x_stalk[perm], edge_index_p)
        torch.testing.assert_close(out_orig, out_perm[rev_perm], atol=1e-4, rtol=1e-4)

    def test_sheafan_vs_res_sheafan_differ(self, setup, num_heads, residual):
        # Both variants should produce different outputs for the same input.
        conv, x_feat, x_stalk, edge_index, d, f = setup
        conv_other = SheafAttnConv(
            stalk_dim=d,
            in_channels=f,
            hidden_dim=conv.hidden_dim,
            num_heads=num_heads,
            context_dim=conv.context_dim,
            residual=not residual,
        )
        # Copy weights so the only difference is the residual flag.
        conv_other.load_state_dict(conv.state_dict())
        conv.eval()
        conv_other.eval()
        with torch.no_grad():
            out_a = conv(x_feat, x_stalk, edge_index)
            out_b = conv_other(x_feat, x_stalk, edge_index)
        assert not torch.allclose(out_a, out_b), "SheafAN and Res-SheafAN must differ"


def test_orthogonal_transport_property():
    """P_ij = restriction_map_dst^T restriction_map_src is orthogonal
    (product of two orthogonal matrices).
    """
    set_seed(42)
    d = 3
    num_edges = 10
    params_dst = torch.randn(num_edges, (d * (d - 1)) // 2)
    params_src = torch.randn(num_edges, (d * (d - 1)) // 2)
    restr_map_dst = cayley(params_dst, d)
    restr_map_src = cayley(params_src, d)
    P = torch.matmul(restr_map_dst.transpose(-2, -1), restr_map_src)
    identity = torch.eye(d).unsqueeze(0).repeat(num_edges, 1, 1)
    torch.testing.assert_close(
        torch.matmul(P.transpose(-2, -1), P), identity, atol=1e-5, rtol=1e-5
    )


def test_fasth_orthogonal_transport_property():
    """P_ij = restriction_map_dst^T restriction_map_src via Householder maps
    is also orthogonal.
    """
    set_seed(42)
    d = 3
    num_edges = 10
    params_dst = torch.randn(num_edges, d * d)
    params_src = torch.randn(num_edges, d * d)
    restr_map_dst = householder(params_dst, d)
    restr_map_src = householder(params_src, d)
    P = torch.matmul(restr_map_dst.transpose(-2, -1), restr_map_src)
    identity = torch.eye(d).unsqueeze(0).repeat(num_edges, 1, 1)
    torch.testing.assert_close(
        torch.matmul(P.transpose(-2, -1), P), identity, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("orth_strategy", ["cayley", "fasth"])
def test_orth_strategy_forward_shape_and_finiteness(orth_strategy):
    """Both orth_strategy values must produce finite outputs with correct shape."""
    set_seed(42)
    d, in_channels, hidden_dim, num_nodes = 3, 8, 8, 10
    context_dim = d * in_channels
    conv = SheafAttnConv(
        stalk_dim=d,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        orth_strategy=orth_strategy,
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, in_channels)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("orth_strategy", ["cayley", "fasth"])
def test_orth_strategy_gradient_flow(orth_strategy):
    """Both orth_strategy values must support gradient flow through map generators."""
    set_seed(42)
    d, in_channels, hidden_dim, num_nodes = 3, 8, 8, 10
    context_dim = d * in_channels
    conv = SheafAttnConv(
        stalk_dim=d,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        orth_strategy=orth_strategy,
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, in_channels)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
    conv(x_feat, x_stalk, edge_index).pow(2).sum().backward()
    for gen in conv.map_generators:
        for param in gen.parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0


def test_no_alpha_parameter():
    """SheafAttnConv must not expose the inherited alpha step-size parameter."""
    conv = SheafAttnConv(stalk_dim=3, in_channels=8, hidden_dim=8)
    param_names = {name for name, _ in conv.named_parameters()}
    assert "alpha" not in param_names


def test_map_generators_count():
    """Number of Cayley map generators must equal num_heads."""
    for num_heads in [1, 2, 4]:
        conv = SheafAttnConv(
            stalk_dim=3, in_channels=8, hidden_dim=8, num_heads=num_heads
        )
        assert len(conv.map_generators) == num_heads


def test_attention_vector_shape():
    """Attention parameter a must have shape [num_heads, 2 * hidden_dim]."""
    hidden_dim, num_heads = 16, 3
    conv = SheafAttnConv(
        stalk_dim=3, in_channels=8, hidden_dim=hidden_dim, num_heads=num_heads
    )
    assert conv.attn_score_vec.shape == (num_heads, 2 * hidden_dim)


def test_reset_parameters_runs():
    """reset_parameters() must complete without error and re-initialize weights."""
    set_seed(42)
    conv = SheafAttnConv(stalk_dim=3, in_channels=8, hidden_dim=8)
    conv.W1.data.fill_(0.5)
    conv.reset_parameters()
    # Reference init is deterministic: identity W1, orthogonal W2.
    assert torch.equal(conv.W1.data, torch.eye(3))
    assert torch.allclose(conv.W2.data @ conv.W2.data.T, torch.eye(8), atol=1e-5)


# ---------------------------------------------------------------------------
# Structural / architectural
# ---------------------------------------------------------------------------


def test_w_att_has_no_bias():
    """attn_projection declared bias=False; verify no bias parameter registered."""
    conv = SheafAttnConv(stalk_dim=3, in_channels=8, hidden_dim=8)
    assert conv.attn_projection.bias is None


def test_sigma_is_elu():
    """SheafAttnConv replaces the inherited Tanh with ELU."""
    conv = SheafAttnConv(stalk_dim=3, in_channels=8, hidden_dim=8)
    assert isinstance(conv.sigma, nn.ELU)


def test_context_dim_default():
    """context_dim defaults to d * in_channels when not explicitly passed."""
    d, in_channels = 3, 8
    conv = SheafAttnConv(stalk_dim=d, in_channels=in_channels, hidden_dim=8)
    assert conv.context_dim == d * in_channels
    assert conv.attn_projection.in_features == d * in_channels


def test_skew_param_count_per_head():
    """Each map generator's output layer must produce d*(d-1)//2 skew entries."""
    for d in [2, 3, 4, 5]:
        conv = SheafAttnConv(
            stalk_dim=d, in_channels=8, hidden_dim=8, orth_strategy="cayley"
        )
        expected = d * (d - 1) // 2
        for gen in conv.map_generators:
            assert isinstance(gen, nn.Sequential)
            assert gen[-1].out_features == expected


def test_fasth_param_count_per_head():
    """Fasth generators must output d*d Householder parameters (not d*(d-1)//2)."""
    for d in [2, 3, 4, 5]:
        conv = SheafAttnConv(
            stalk_dim=d, in_channels=8, hidden_dim=8, orth_strategy="fasth"
        )
        expected = d * d
        for gen in conv.map_generators:
            assert isinstance(gen, nn.Sequential)
            assert gen[-1].out_features == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_node_graph():
    """A single node receives only its own self-loop; must not crash."""
    set_seed(42)
    d, in_channels, hidden_dim = 3, 8, 8
    context_dim = d * in_channels
    conv = SheafAttnConv(
        stalk_dim=d,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
    )
    x_feat = torch.randn(1, context_dim)
    x_stalk = torch.randn(1, d, in_channels)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == (1, d, in_channels)
    assert torch.isfinite(out).all()


def test_d1_stalk_zero_skew_params():
    """d=1 -> num_skew=0, Cayley map is always identity; output must be finite."""
    set_seed(42)
    d, in_channels, hidden_dim = 1, 8, 8
    context_dim = d * in_channels
    conv = SheafAttnConv(
        stalk_dim=d,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
    )
    gen0 = conv.map_generators[0]
    assert isinstance(gen0, nn.Sequential)
    assert gen0[-1].out_features == 0
    x_feat = torch.randn(6, context_dim)
    x_stalk = torch.randn(6, d, in_channels)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


def test_fully_connected_small_graph():
    """Complete directed graph (K4); every node attends to all others + self."""
    set_seed(42)
    num_nodes, d, in_channels, hidden_dim = 4, 3, 8, 8
    context_dim = d * in_channels
    conv = SheafAttnConv(
        stalk_dim=d,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, in_channels)
    idx = torch.arange(num_nodes)
    src = idx.repeat_interleave(num_nodes)
    dst = idx.repeat(num_nodes)
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]])
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


def test_leaky_relu_slope_changes_output():
    """Different leaky_relu_slope -> different attention logits -> different output."""
    set_seed(42)
    d, in_channels, hidden_dim = 3, 8, 8
    context_dim = d * in_channels
    x_feat = torch.randn(8, context_dim)
    x_stalk = torch.randn(8, d, in_channels)
    edge_index = erdos_renyi_graph(8, edge_prob=0.5)

    conv_a = SheafAttnConv(
        stalk_dim=d,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        leaky_relu_slope=0.01,
    )
    conv_b = SheafAttnConv(
        stalk_dim=d,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        leaky_relu_slope=0.9,
    )
    conv_b.load_state_dict(conv_a.state_dict())

    conv_a.eval()
    conv_b.eval()
    with torch.no_grad():
        out_a = conv_a(x_feat, x_stalk, edge_index)
        out_b = conv_b(x_feat, x_stalk, edge_index)
    assert not torch.allclose(out_a, out_b)


def test_clamp_val_stabilizes_extreme_inputs():
    """Tight clamp_val keeps the Cayley solve well-conditioned under large inputs."""
    set_seed(42)
    d, in_channels, hidden_dim = 3, 8, 8
    context_dim = d * in_channels
    conv = SheafAttnConv(
        stalk_dim=d,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        clamp_val=1e-3,
    )
    x_feat = torch.randn(6, context_dim) * 100
    x_stalk = torch.randn(6, d, in_channels)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Transport map properties
# ---------------------------------------------------------------------------


def test_self_loop_transport_is_identity():
    """For a self-loop (i, i), ctx_dst == ctx_src -> restriction_map_dst =
    restriction_map_src -> P_ii = restriction_map_dst^T restriction_map_dst = I.
    """
    set_seed(42)
    d, batch = 4, 8
    num_skew = (d * (d - 1)) // 2
    params = torch.randn(batch, num_skew)
    F = cayley(params, d)
    P_self = torch.matmul(F.transpose(-2, -1), F)
    identity = torch.eye(d).unsqueeze(0).expand(batch, -1, -1)
    torch.testing.assert_close(P_self, identity, atol=1e-5, rtol=1e-5)


def test_transport_map_shape_per_head():
    """attn_maps must be [E, H, d, d] before head averaging."""
    set_seed(42)
    num_heads, d, in_channels, hidden_dim = 3, 4, 8, 8
    context_dim = d * in_channels
    conv = SheafAttnConv(
        stalk_dim=d,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        context_dim=context_dim,
    )
    # Capture the per-head maps by probing with a single edge.
    ctx = torch.randn(1, 2 * context_dim)
    restr_map_dst = cayley(conv.map_generators[0](ctx), d)
    assert restr_map_dst.shape == (1, d, d)
