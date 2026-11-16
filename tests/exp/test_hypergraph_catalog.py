"""Prediction domains and graph/hypergraph namespace regression checks."""

import pytest
import torch

from exp.datasets.hypergraph import HYPERGRAPHS, artifact_path
from exp.registries.presets import preset_registry
from sheaf_mpnn.nsd.dhsnn_model import DHSNNModel
from sheaf_mpnn.nsd.hsnn_model import HSNNModel


def test_public_names_do_not_replace_graph_presets():
    assert len(HYPERGRAPHS) == 14
    assert preset_registry.get("cora").dataset.name == "cora"
    for name in HYPERGRAPHS:
        assert preset_registry.get(name).dataset.name == name
    assert artifact_path("cora-hypergraph", "data").name == "cora-dshn-q0.npz"


@pytest.mark.parametrize("directed", [False, True])
def test_hyperedge_readout_is_sum_before_decoder(directed):
    model_type = DHSNNModel if directed else HSNNModel
    model = model_type(in_channels=3, out_channels=2, stalk_dim=2, hidden_dim=2)
    model.eval()
    x = torch.randn(4, 3)
    index = torch.tensor([[0, 1, 2, 2, 3], [0, 0, 0, 1, 1]])
    kwargs = {"incidence_roles": torch.tensor([-1, 1, 1, -1, 1])} if directed else {}
    captured = []
    hook = model.decoder.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0])
    )
    model(x, index, num_hyperedges=2, **kwargs)
    edges = model(x, index, num_hyperedges=2, task="hyperedge", **kwargs)
    hook.remove()
    expected = torch.stack((captured[0][:3].sum(0), captured[0][2:].sum(0)))
    torch.testing.assert_close(captured[1], expected)
    torch.testing.assert_close(edges, model.decoder(expected))
    edges.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
