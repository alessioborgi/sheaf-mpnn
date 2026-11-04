# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from typing import Any

import pytest
import torch
from torch import nn
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.sheaf_attn.sheaf_attn_layers import HetSheafAttnConv

NUM_NODES = 10
STALK_F = 8
D = 4
HIDDEN = 8
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4


def _make_graph():
    return erdos_renyi_graph(NUM_NODES, edge_prob=0.4)


def _reinit_mlp(mlp: nn.Module) -> None:
    if isinstance(mlp, nn.Sequential):
        for m in mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)


def _make_layer(**kwargs: Any) -> HetSheafAttnConv:
    defaults: dict[str, Any] = dict(
        stalk_dim=D,
        in_channels=STALK_F,
        hidden_dim=HIDDEN,
        num_heads=2,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        context_dim=D * STALK_F,
    )
    defaults.update(kwargs)
    return HetSheafAttnConv(**defaults)


@pytest.mark.parametrize("orth_strategy", ["cayley", "fasth"])
def test_output_shape_orth_strategy(orth_strategy):
    torch.manual_seed(42)
    conv = _make_layer(
        orth_strategy=orth_strategy, use_node_types=True, use_edge_types=True
    )
    x_feat = torch.randn(NUM_NODES, D * STALK_F)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    out = conv(x_feat, x_stalk, edge_index, nt, et)
    assert out.shape == (NUM_NODES, D, STALK_F)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("residual", [False, True])
def test_residual_mode(residual):
    torch.manual_seed(42)
    conv = _make_layer(residual=residual)
    x_feat = torch.randn(NUM_NODES, D * STALK_F)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == (NUM_NODES, D, STALK_F)


def test_node_type_changes_output():
    torch.manual_seed(0)
    conv = _make_layer(use_node_types=True)
    for gen in conv.map_generators:
        _reinit_mlp(gen)
    conv.eval()
    x_feat = torch.randn(NUM_NODES, D * STALK_F)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    nt_a = torch.zeros(NUM_NODES, dtype=torch.long)
    nt_b = torch.ones(NUM_NODES, dtype=torch.long)
    with torch.no_grad():
        out_a = conv(x_feat, x_stalk, edge_index, nt_a, None)
        out_b = conv(x_feat, x_stalk, edge_index, nt_b, None)
    assert not torch.allclose(out_a, out_b)


def test_edge_type_changes_output():
    torch.manual_seed(0)
    conv = _make_layer(use_edge_types=True)
    for gen in conv.map_generators:
        _reinit_mlp(gen)
    conv.eval()
    x_feat = torch.randn(NUM_NODES, D * STALK_F)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    E = edge_index.size(1)
    et_a = torch.zeros(E, dtype=torch.long)
    et_b = torch.ones(E, dtype=torch.long)
    with torch.no_grad():
        out_a = conv(x_feat, x_stalk, edge_index, None, et_a)
        out_b = conv(x_feat, x_stalk, edge_index, None, et_b)
    assert not torch.allclose(out_a, out_b)


def test_gradient_flow():
    conv = _make_layer(use_node_types=True, use_edge_types=True)
    x_feat = torch.randn(NUM_NODES, D * STALK_F)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    out = conv(x_feat, x_stalk, edge_index, nt, et)
    out.pow(2).sum().backward()
    assert conv.W1.grad is not None and conv.W1.grad.abs().sum() > 0
    assert conv.attn_projection.weight.grad is not None


def test_fallback_without_types():
    conv = _make_layer(use_node_types=True, use_edge_types=True)
    x_feat = torch.randn(NUM_NODES, D * STALK_F)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    out = conv(x_feat, x_stalk, edge_index, None, None)
    assert out.shape == (NUM_NODES, D, STALK_F)
    assert not torch.isnan(out).any()


def test_ensemble_mode():
    conv = _make_layer(ensemble=True, num_heads=2)
    x_feat = torch.randn(NUM_NODES, D * STALK_F)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    out = conv(x_feat, x_stalk, edge_index, None, et)
    assert out.shape == (NUM_NODES, D, STALK_F)
    assert not torch.isnan(out).any()


def test_ensemble_no_edge_types():
    """Ensemble mode with edge_types=None takes the edge_types_idx is None branch."""
    torch.manual_seed(42)
    conv = _make_layer(ensemble=True, num_heads=2)
    x_feat = torch.randn(NUM_NODES, D * STALK_F)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    # Pass edge_types=None to trigger the `if edge_types_idx is None` branch
    # inside _ensemble_transport.
    out = conv(x_feat, x_stalk, edge_index, None, None)
    assert out.shape == (NUM_NODES, D, STALK_F)
    assert not torch.isnan(out).any()


def test_ensemble_with_node_types_no_shape_error():
    """Ensemble + use_node_types: generator input must be 2*_attn_ctx_dim.

    Regression test: ensemble map_generators were previously built with
    2 * context_dim but message() passes 2 * _attn_ctx_dim (which includes
    the node-type one-hot), causing a runtime matmul shape mismatch.
    """
    conv = _make_layer(ensemble=True, num_heads=1, use_node_types=True)
    x_feat = torch.randn(NUM_NODES, D * STALK_F)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    # Would crash before the fix: "mat1 and mat2 shapes cannot be multiplied"
    out = conv(x_feat, x_stalk, edge_index, nt, et)
    assert out.shape == (NUM_NODES, D, STALK_F)
    assert not torch.isnan(out).any()
