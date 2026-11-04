import pytest
import torch

from hetsheaf_mpnn.nsd.nsd_layers import (
    DiagonalHetNSDConv,
    GeneralHetNSDConv,
    LowRankHetNSDConv,
    OrthogonalHetNSDConv,
)
from hetsheaf_mpnn.nsp.nsp_layers import (
    DiagonalHetNSPConv,
    GeneralHetNSPConv,
    LowRankHetNSPConv,
    OrthogonalHetNSPConv,
)
from hetsheaf_mpnn.sheaf_attn.sheaf_attn_layers import HetSheafAttnConv
from sheaf_mpnn.nsd.nsd_layers import (
    DiagonalNSDConv,
    GeneralNSDConv,
    LowRankNSDConv,
    OrthogonalNSDConv,
)
from sheaf_mpnn.nsp.nsp_layers import (
    DiagonalNSPConv,
    GeneralNSPConv,
    LowRankNSPConv,
    OrthogonalNSPConv,
)
from sheaf_mpnn.sheaf_attn.sheaf_attn_layers import SheafAttnConv

HOMOGENEOUS_CONVS = [
    DiagonalNSDConv,
    GeneralNSDConv,
    OrthogonalNSDConv,
    LowRankNSDConv,
    DiagonalNSPConv,
    GeneralNSPConv,
    OrthogonalNSPConv,
    LowRankNSPConv,
    SheafAttnConv,
]

HETEROGENEOUS_CONVS = [
    DiagonalHetNSDConv,
    GeneralHetNSDConv,
    OrthogonalHetNSDConv,
    LowRankHetNSDConv,
    DiagonalHetNSPConv,
    GeneralHetNSPConv,
    OrthogonalHetNSPConv,
    LowRankHetNSPConv,
    HetSheafAttnConv,
]


@pytest.mark.parametrize("conv_class", HOMOGENEOUS_CONVS)
def test_homogeneous_directionality(conv_class):
    """Verify that all homogeneous layers handle directed edges correctly."""
    num_nodes = 3
    d = 2
    f = 3
    hidden = 8
    context_dim = d * f

    # Layer initialization
    kwargs = {
        "stalk_dim": d,
        "in_channels": f,
        "hidden_dim": hidden,
        "context_dim": context_dim,
    }
    if conv_class in [LowRankNSDConv, LowRankNSPConv]:
        kwargs["rank"] = 1

    layer = conv_class(**kwargs)

    x_feat = torch.randn(num_nodes, context_dim)

    # NSP uses augmented state [N, 2d, f]
    is_nsp = "NSP" in conv_class.__name__
    x_stalk = torch.randn(num_nodes, 2 * d if is_nsp else d, f)

    # Directed edges: 0->1, 1->2
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    out = layer(x_feat, x_stalk, edge_index)

    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("conv_class", HETEROGENEOUS_CONVS)
def test_heterogeneous_directionality(conv_class):
    """Verify that all heterogeneous layers handle directed edges correctly."""
    num_nodes = 3
    d = 2
    f = 3
    hidden = 8
    context_dim = d * f
    num_nt = 2
    num_et = 2

    # Layer initialization
    kwargs = {
        "stalk_dim": d,
        "in_channels": f,
        "hidden_dim": hidden,
        "context_dim": context_dim,
        "num_node_types": num_nt,
        "num_edge_types": num_et,
        "use_node_types": True,
        "use_edge_types": True,
    }
    if conv_class in [LowRankHetNSDConv, LowRankHetNSPConv]:
        kwargs["rank"] = 1

    layer = conv_class(**kwargs)

    x_feat = torch.randn(num_nodes, context_dim)

    # NSP uses augmented state [N, 2d, f]
    is_nsp = "NSP" in conv_class.__name__
    x_stalk = torch.randn(num_nodes, 2 * d if is_nsp else d, f)

    # Directed edges: 0->1, 1->2
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    node_types = torch.tensor([0, 1, 0], dtype=torch.long)
    edge_types = torch.tensor([0, 1], dtype=torch.long)

    out = layer(
        x_feat, x_stalk, edge_index, node_types=node_types, edge_types=edge_types
    )

    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


def test_isolated_nodes_forward():
    """Verify that a representative layer (DiagonalNSDConv) handles
    isolated nodes correctly.
    """
    num_nodes = 4
    d = 2
    f = 3
    hidden = 4
    context_dim = d * f

    layer = DiagonalNSDConv(
        stalk_dim=d, in_channels=f, hidden_dim=hidden, context_dim=context_dim
    )

    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, f)

    # Only one edge: 0 -> 1. Nodes 2 and 3 are isolated.
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    out = layer(x_feat, x_stalk, edge_index)

    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


def test_self_loops_forward():
    """Verify that a representative layer (DiagonalNSDConv) handles
    self-loops correctly.
    """
    num_nodes = 2
    d = 2
    f = 3
    hidden = 4
    context_dim = d * f

    layer = DiagonalNSDConv(
        stalk_dim=d, in_channels=f, hidden_dim=hidden, context_dim=context_dim
    )

    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, f)

    # Self loops: 0->0, 1->1
    edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    out = layer(x_feat, x_stalk, edge_index)

    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()
