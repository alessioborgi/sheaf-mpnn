# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import random

import pytest
import torch
from torch import nn
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.csnn.csnn_layers import ConformalSheafLearner, CSNNConv
from sheaf_mpnn.csnn.gnn_builder import GNN_TYPES, SumGNN, build_gnn, get_layer_type
from sheaf_mpnn.utils import orth_param_count


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.fixture(autouse=True)
def _deterministic_algorithms():
    """Enables deterministic kernels for this module, then restores the flag.

    ``use_deterministic_algorithms`` is process-global: left enabled it makes
    later CuBLAS matmuls raise unless CUBLAS_WORKSPACE_CONFIG is set, which
    breaks the sweep end-to-end test that trains on a real GPU.
    """
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


# The four learner backbones the shipped presets exercise:
# the two GraphSAGE presets, the custom stack, and the plain MLP.
_LEARNERS = [
    pytest.param(1, 2, id="sage_layernorm"),
    pytest.param(2, 3, id="sage_gelu_project"),
    pytest.param(0, 2, id="custom_stack"),
    pytest.param(1, 0, id="mlp"),
]


@pytest.mark.parametrize(("gnn_default", "gnn_layers"), _LEARNERS)
class TestCSNNConv:
    @pytest.fixture
    def setup(self, gnn_default, gnn_layers):
        set_seed(42)
        num_nodes, d, hidden_dim = 15, 3, 4
        layer = CSNNConv(
            stalk_dim=d,
            hidden_dim=hidden_dim,
            gnn_default=gnn_default,
            gnn_layers=gnn_layers,
            gnn_hidden=8,
        )
        x_feat = torch.randn(num_nodes, d * hidden_dim)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        return layer, x_feat, edge_index

    def test_output_shape(self, setup):
        layer, x_feat, edge_index = setup
        out = layer(x_feat, x_feat.view(15, 3, 4), edge_index)
        assert out.shape == x_feat.shape
        assert not torch.isnan(out).any()

    def test_isolated_graph_is_identity(self, setup):
        """With no edges, both Laplacian passes vanish and eps is zero at init.

        The update reduces to ``(1 + tanh(0)) * x0 - 0 == x0``, so any drift
        here means the residual or the message convention is wrong.
        """
        layer, x_feat, _ = setup
        layer.eval()
        empty = torch.empty((2, 0), dtype=torch.long)
        out = layer(x_feat, x_feat.view(15, 3, 4), empty)
        torch.testing.assert_close(out, x_feat, atol=1e-6, rtol=1e-5)

    def test_gradient_flow(self, setup):
        layer, x_feat, edge_index = setup
        out = layer(x_feat, x_feat.view(15, 3, 4), edge_index)
        out.pow(2).mean().backward()

        assert layer.epsilons.grad is not None
        assert layer.epsilons.grad.abs().sum() > 0
        assert isinstance(layer.lin_left_weights, nn.Linear)
        assert layer.lin_left_weights.weight.grad.abs().sum() > 0
        assert isinstance(layer.lin_right_weights, nn.Linear)
        assert layer.lin_right_weights.weight.grad.abs().sum() > 0
        for name, param in layer.out_maps_learner.named_parameters():
            assert param.grad is not None, name

    def test_eval_is_deterministic(self, setup):
        layer, x_feat, edge_index = setup
        layer.eval()
        a = layer(x_feat, x_feat.view(15, 3, 4), edge_index)
        b = layer(x_feat, x_feat.view(15, 3, 4), edge_index)
        torch.testing.assert_close(a, b)


class TestCSNNConvConfiguration:
    def test_epsilons_start_at_zero(self):
        layer = CSNNConv(stalk_dim=3, hidden_dim=4)
        assert torch.equal(layer.epsilons, torch.zeros(3, 1))

    def test_left_right_disabled_are_identity(self):
        layer = CSNNConv(
            stalk_dim=3, hidden_dim=4, left_weights=False, right_weights=False
        )
        assert isinstance(layer.lin_left_weights, nn.Identity)
        assert isinstance(layer.lin_right_weights, nn.Identity)

    def test_left_weights_init_to_identity(self):
        layer = CSNNConv(stalk_dim=3, hidden_dim=4)
        assert isinstance(layer.lin_left_weights, nn.Linear)
        torch.testing.assert_close(layer.lin_left_weights.weight, torch.eye(3))

    def test_right_weights_init_orthogonal(self):
        layer = CSNNConv(stalk_dim=3, hidden_dim=5)
        assert isinstance(layer.lin_right_weights, nn.Linear)
        w = layer.lin_right_weights.weight
        torch.testing.assert_close(w @ w.t(), torch.eye(5), atol=1e-5, rtol=1e-4)

    def test_stalk_dim_must_exceed_one(self):
        with pytest.raises(ValueError, match="stalk_dim must be greater than 1"):
            CSNNConv(stalk_dim=1, hidden_dim=4)

    def test_hidden_dim_must_be_positive(self):
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            CSNNConv(stalk_dim=3, hidden_dim=0)

    def test_flow_is_target_to_source(self):
        """The cooperative Laplacian scatters into edge_index[0], not [1]."""
        assert CSNNConv(stalk_dim=3, hidden_dim=4).flow == "target_to_source"


@pytest.mark.parametrize("conformal", [True, False])
@pytest.mark.parametrize("strategy", ["cayley", "householder", "fasthpp"])
class TestConformalSheafLearner:
    def test_output_shapes_and_bounds(self, conformal, strategy):
        set_seed(42)
        d, hidden_dim, num_nodes = 4, 4, 12
        num_params = orth_param_count(strategy, d)
        learner = ConformalSheafLearner(
            stalk_dim=d,
            hidden_dim=hidden_dim,
            num_params=num_params,
            conformal=conformal,
            gnn_layers=1,
            gnn_hidden=8,
        )
        x = torch.randn(num_nodes, d * hidden_dim)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        params, c = learner(x, edge_index)

        assert params.shape == (num_nodes, num_params)
        assert c.shape == (num_nodes,)
        assert params.abs().max() <= 1.0  # tanh-bounded
        assert (c > 0).all()  # positive scale
        if conformal:
            assert c.max() <= 10.0 + 1e-5  # exp(clamp_max(log 10))
        else:
            torch.testing.assert_close(c, torch.ones(num_nodes))


class TestConformalSheafLearnerConfiguration:
    def test_positional_encoding_widens_input(self):
        d, hidden_dim, pe_size, num_nodes = 3, 4, 5, 10
        learner = ConformalSheafLearner(
            stalk_dim=d, hidden_dim=hidden_dim, num_params=3, pe_size=pe_size
        )
        x = torch.randn(num_nodes, d * hidden_dim)
        pe = torch.randn(num_nodes, pe_size)
        params, _ = learner(x, torch.empty((2, 0), dtype=torch.long), pe=pe)
        assert params.shape == (num_nodes, 3)

    def test_edge_attr_reaches_conv_that_accepts_it(self):
        """GAT takes edge_attr; the custom stack must forward it."""
        d, hidden_dim, num_nodes = 3, 4, 10
        learner = ConformalSheafLearner(
            stalk_dim=d,
            hidden_dim=hidden_dim,
            num_params=3,
            gnn_default=0,
            gnn_layers=2,
            gnn_hidden=8,
            gnn_type="GAT",
        )
        assert learner._accepts_edge_attr()
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
        x = torch.randn(num_nodes, d * hidden_dim)
        params, _ = learner(x, edge_index, edge_attr=torch.randn(edge_index.size(1), 1))
        assert params.shape == (num_nodes, 3)

    def test_sage_does_not_accept_edge_attr(self):
        learner = ConformalSheafLearner(
            stalk_dim=3, hidden_dim=4, num_params=3, gnn_default=0, gnn_layers=1
        )
        assert not learner._accepts_edge_attr()

    def test_invalid_gnn_default(self):
        with pytest.raises(ValueError, match="gnn_default must be 0, 1 or 2"):
            ConformalSheafLearner(
                stalk_dim=3, hidden_dim=4, num_params=3, gnn_default=3
            )

    def test_sgc_rejects_residual(self):
        with pytest.raises(ValueError, match="SGC does not support residual"):
            ConformalSheafLearner(
                stalk_dim=3,
                hidden_dim=4,
                num_params=3,
                gnn_type="SGC",
                gnn_residual=True,
            )

    def test_gps_requires_linear_emb(self):
        """GPSConv preserves width, so the plain custom stack cannot use it."""
        with pytest.raises(ValueError, match="GPS.*linear_emb"):
            ConformalSheafLearner(
                stalk_dim=3,
                hidden_dim=4,
                num_params=3,
                gnn_default=0,
                gnn_layers=2,
                gnn_hidden=8,
                gnn_type="GPS",
            )

    def test_gps_with_linear_emb_forward(self):
        set_seed(42)
        learner = ConformalSheafLearner(
            stalk_dim=3,
            hidden_dim=4,
            num_params=3,
            gnn_default=0,
            gnn_layers=2,
            gnn_hidden=8,
            gnn_type="GPS",
            linear_emb=True,
        )
        x = torch.randn(10, 12)
        params, c = learner(x, erdos_renyi_graph(10, edge_prob=0.5))
        assert params.shape == (10, 3)
        assert c.shape == (10,)

    def test_nnconv_requires_edge_attr(self):
        """NNConv computes weights from edge features; fail loudly without them."""
        learner = ConformalSheafLearner(
            stalk_dim=3,
            hidden_dim=4,
            num_params=3,
            gnn_default=0,
            gnn_layers=2,
            gnn_hidden=8,
            gnn_type="NNConv",
        )
        x = torch.randn(10, 12)
        with pytest.raises(ValueError, match="NNConv.*requires edge_attr"):
            learner(x, erdos_renyi_graph(10, edge_prob=0.5))

    def test_nnconv_with_edge_attr_forward(self):
        set_seed(42)
        learner = ConformalSheafLearner(
            stalk_dim=3,
            hidden_dim=4,
            num_params=3,
            gnn_default=0,
            gnn_layers=2,
            gnn_hidden=8,
            gnn_type="NNConv",
        )
        x = torch.randn(10, 12)
        edge_index = erdos_renyi_graph(10, edge_prob=0.5)
        params, c = learner(x, edge_index, edge_attr=torch.randn(edge_index.size(1), 2))
        assert params.shape == (10, 3)
        assert c.shape == (10,)

    def test_linear_emb_reset_parameters_changes_output(self):
        set_seed(42)
        learner = ConformalSheafLearner(
            stalk_dim=3,
            hidden_dim=4,
            num_params=3,
            gnn_default=0,
            gnn_layers=2,
            gnn_hidden=8,
            linear_emb=True,
        )
        x = torch.randn(10, 12)
        edge_index = erdos_renyi_graph(10, edge_prob=0.5)
        before, _ = learner(x, edge_index)
        learner.reset_parameters()
        after, _ = learner(x, edge_index)
        assert not torch.allclose(before, after)

    def test_sgc_with_linear_emb_forward(self):
        """SGC inside the embed/project wrapper folds depth into one SGConv."""
        set_seed(42)
        learner = ConformalSheafLearner(
            stalk_dim=3,
            hidden_dim=4,
            num_params=3,
            gnn_default=0,
            gnn_layers=3,
            gnn_hidden=8,
            gnn_type="SGC",
            linear_emb=True,
        )
        x = torch.randn(10, 12)
        params, c = learner(x, erdos_renyi_graph(10, edge_prob=0.5))
        assert params.shape == (10, 3)
        assert c.shape == (10,)

    @pytest.mark.parametrize("linear_emb", [False, True])
    def test_residual_stack_forward(self, linear_emb):
        """gnn_residual adds skips: every layer with linear_emb, middle only without."""
        set_seed(42)
        learner = ConformalSheafLearner(
            stalk_dim=3,
            hidden_dim=4,
            num_params=3,
            gnn_default=0,
            gnn_layers=3,
            gnn_hidden=8,
            gnn_residual=True,
            linear_emb=linear_emb,
        )
        x = torch.randn(10, 12)
        params, c = learner(x, erdos_renyi_graph(10, edge_prob=0.5))
        assert params.shape == (10, 3)
        assert c.shape == (10,)

    @pytest.mark.parametrize("gnn_type", ["GCN", "GAT", "SAGE", "SGC", "SumGNN"])
    def test_custom_stack_types_forward(self, gnn_type):
        """Every edge-attr-free custom backbone runs and emits the right width."""
        set_seed(42)
        learner = ConformalSheafLearner(
            stalk_dim=3,
            hidden_dim=4,
            num_params=3,
            gnn_default=0,
            gnn_layers=2,
            gnn_hidden=8,
            gnn_type=gnn_type,
        )
        x = torch.randn(10, 12)
        params, c = learner(x, erdos_renyi_graph(10, edge_prob=0.5))
        assert params.shape == (10, 3)
        assert c.shape == (10,)


class TestGNNBuilder:
    @pytest.mark.parametrize("gnn_type", GNN_TYPES)
    def test_every_type_resolves(self, gnn_type):
        assert get_layer_type(gnn_type) is not None

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported GNN layer type"):
            get_layer_type("NotAGNN")

    def test_gps_hidden_stack_raises(self):
        with pytest.raises(ValueError, match="GPS does not support a hidden"):
            build_gnn("GPS", 12, 3, 3, 8)

    def test_gps_width_change_raises(self):
        with pytest.raises(ValueError, match="in_channels == out_channels"):
            build_gnn("GPS", 12, 3, 1)

    def test_sum_gnn_forward(self):
        set_seed(42)
        conv = SumGNN(6, 3)
        x = torch.randn(10, 6)
        out = conv(x, erdos_renyi_graph(10, edge_prob=0.4))
        assert out.shape == (10, 3)
        assert not torch.isnan(out).any()

    def test_sum_gnn_reset_parameters(self):
        set_seed(42)
        conv = SumGNN(6, 3)
        before = conv.W_s.weight.clone()
        conv.reset_parameters()
        assert not torch.equal(before, conv.W_s.weight)

    def test_num_layers_must_be_positive(self):
        with pytest.raises(ValueError, match="num_layers must be positive"):
            build_gnn("SAGE", 8, 3, 0)

    def test_sgc_no_hidden_folds_depth_into_k(self):
        from torch_geometric.nn import SGConv

        conv = build_gnn("SGC", 8, 3, 3)
        assert isinstance(conv, SGConv)
        assert conv.K == 3

    def test_nnconv_no_hidden_shares_edge_net(self):
        stack = build_gnn("NNConv", 8, 8, 2)
        assert isinstance(stack, nn.ModuleList)
        assert len(stack) == 2
        assert stack[0].nn is stack[1].nn

    def test_nnconv_hidden_stack_widths(self):
        stack = build_gnn("NNConv", 8, 3, 3, 6)
        assert isinstance(stack, nn.ModuleList)
        assert len(stack) == 3
        assert stack[0].in_channels == 8
        assert stack[-1].out_channels == 3
