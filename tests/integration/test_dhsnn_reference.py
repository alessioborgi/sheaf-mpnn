"""Execute pinned upstream components; no downloads during collection/tests."""

import hashlib
import importlib
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sheaf_mpnn import DHSNNModel, DHSNNVariant

COMMIT = "d94f81bc7d86bd8d41e416083063f3446c1ad796"
ROOT = Path("tmp/dshn-source") / COMMIT
pytestmark = pytest.mark.integration


@pytest.fixture
def upstream(monkeypatch):
    if not (ROOT / "manifest.json").exists():
        pytest.skip("Run scripts/prepare_dshn_reference.py first")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["commit"] == COMMIT
    for relative, checksum in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == checksum
    for name, relative in (
        ("models", "models"),
        ("models.sheaf_utils", "models/sheaf_utils"),
        ("models.complex_utils", "models/complex_utils"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str((ROOT / relative).resolve())]
        monkeypatch.setitem(sys.modules, name, package)
    # The upstream file imports this compiled extension even for Cayley. Only
    # bypass the unused import: attempting Householder here must fail explicitly.
    extension = types.ModuleType("torch_householder")

    def unavailable(*args, **kwargs):
        raise RuntimeError("Compiled upstream Householder is not tested")

    extension.torch_householder_orgqr = unavailable
    monkeypatch.setitem(sys.modules, "torch_householder", extension)
    before = set(sys.modules)
    module = importlib.import_module("models.sheafgedi")
    yield module
    for name in set(sys.modules) - before:
        if name.startswith("models."):
            sys.modules.pop(name, None)


def reference_args(
    family,
    dynamic=False,
    norm="sym_block_norm",
    prediction="MLP_var1",
    input_norm=False,
):
    return SimpleNamespace(
        sheaf_num_layers=2,
        dropout=0.0,
        MLP_hidden=3,
        d=2,
        init_hedge="avg",
        sheaf_normtype=norm,
        sheaf_act="sigmoid",
        sheaf_left_proj=True,
        light=False,
        AllSet_input_norm=input_norm,
        dynamic_sheaf=dynamic,
        elementwise_affine=True,
        residual=True,
        add_identity=True,
        cuda=-1,
        sheaf_dropout=False,
        sheaf_special_head=False,
        sheaf_pred_block=prediction,
        shared_weights=False,
        orthogonal_map="cayley",
        Classifier_hidden=5,
        Classifier_num_layers=2,
        normalization="ln",
    )


def copy_weights(model, reference):
    pairs = [(model.encoder.weight, reference.lin.linear.weight)]
    for learner, builder in zip(
        model.map_learners, reference.sheaf_builder, strict=True
    ):
        name = {
            "diagonal": "sheaf_lin",
            "general": "general_sheaf_lin",
            "orthogonal": "orth_lin",
        }[learner.family]
        pairs.append((learner.linear.weight, getattr(builder, name).linear.weight))
        if learner.edge_linear is not None:
            extra = {
                "diagonal": "sheaf_lin2",
                "general": "general_sheaf_lin2",
                "orthogonal": "orth_lin2",
            }[learner.family]
            pairs.append(
                (learner.edge_linear.weight, getattr(builder, extra).linear.weight)
            )
    for layer, conv in zip(model.layers, reference.convs, strict=True):
        pairs.extend(
            [
                (layer.W2, conv.lin.linear.weight.T),
                (layer.bias, conv.bias),
            ]
        )
        norms = [(layer.feature_norm, conv.lin)]
        if layer.left_projection:
            pairs.append((layer.W1, conv.lin_left.linear.weight))
            norms.append((layer.stalk_norm, conv.lin_left))
        for ours, theirs in norms:
            if theirs.apply_norm:
                pairs.extend(
                    [(ours.weight, theirs.norm.weight), (ours.bias, theirs.norm.bias)]
                )
    linear = [m for m in model.decoder if isinstance(m, torch.nn.Linear)]
    for ours, theirs in zip(linear, reference.Classifier.lins, strict=True):
        pairs.extend([(ours.weight, theirs.weight), (ours.bias, theirs.bias)])
    norms = [m for m in model.decoder if isinstance(m, torch.nn.LayerNorm)]
    for ours, theirs in zip(
        norms, reference.Classifier.normalizations[1:], strict=True
    ):
        pairs.extend([(ours.weight, theirs.weight), (ours.bias, theirs.bias)])
    with torch.no_grad():
        for ours, theirs in pairs:
            ours.copy_(theirs)
    return pairs


@pytest.mark.parametrize("family", ["diagonal", "orthogonal", "general"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("light", [False, True])
@pytest.mark.parametrize("init_hedge", ["avg", "rand"])
@pytest.mark.parametrize("orthogonal_map", ["native", "matrix_exp"])
def test_light_and_undirected_reference_gradients(
    upstream, family, directed, light, init_hedge, orthogonal_map
):
    torch.set_num_threads(2)
    torch.manual_seed(29)
    index = torch.tensor([[0, 1, 2, 1, 3, 4], [0, 0, 0, 1, 1, 1]])
    roles = (
        torch.tensor([-1, 1, 1, -1, 1, 1])
        if directed
        else torch.ones(6, dtype=torch.long)
    )
    args = reference_args(family)
    args.light = light
    args.init_hedge = init_hedge
    if orthogonal_map != "native":
        args.orthogonal_map = orthogonal_map
    reference = upstream.SheafGeDi(
        4,
        2,
        roles,
        0.25,
        args,
        {
            "diagonal": "DiagSheafs",
            "general": "GeneralSheafs",
            "orthogonal": "OrthoSheafs",
        }[family],
    ).double()
    reference.reset_parameters()
    model = DHSNNModel(
        4,
        2,
        stalk_dim=2,
        hidden_dim=3,
        variant=DHSNNVariant[family.upper()],
        input_norm=False,
        left_projection=True,
        residual=True,
        classifier_hidden=5,
        light=light,
        init_hedge=init_hedge,
        orthogonal_map=orthogonal_map,
    ).double()
    pairs = copy_weights(model, reference)
    x = torch.randn(5, 4, dtype=torch.float64)
    if init_hedge == "rand":
        reference.hyperedge_attr = torch.randn(2, 4, dtype=torch.complex128)
    expected = reference(SimpleNamespace(x=x, edge_index=index))
    if init_hedge == "rand":
        model.random_edge_features = reference.hyperedge_attr.detach().clone()
    actual = model(x, index, roles)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for ours, theirs in pairs:
        gradient = theirs._base.grad.T if theirs._base is not None else theirs.grad
        assert (ours.grad is None) == (gradient is None)
        if ours.grad is not None:
            torch.testing.assert_close(ours.grad, gradient, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("family", ["diagonal", "orthogonal", "general"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("prediction", ["MLP_var1", "MLP_var2", "MLP_var3"])
@pytest.mark.parametrize("input_norm", [False, True])
def test_upstream_network_forward(
    upstream, family, dynamic, prediction, input_norm, record_property
):
    torch.set_num_threads(2)
    torch.manual_seed(18)
    index = torch.tensor([[0, 1, 2, 1, 3, 4], [0, 0, 0, 1, 1, 1]])
    roles = torch.tensor([-1, 1, 1, -1, 1, 1])
    args = reference_args(family, dynamic, prediction=prediction, input_norm=input_norm)
    reference = upstream.SheafGeDi(
        4,
        2,
        roles,
        0.25,
        args,
        {
            "diagonal": "DiagSheafs",
            "general": "GeneralSheafs",
            "orthogonal": "OrthoSheafs",
        }[family],
    )
    reference.reset_parameters()
    model = DHSNNModel(
        4,
        2,
        stalk_dim=2,
        hidden_dim=3,
        variant=DHSNNVariant[family.upper()],
        dynamic_sheaf=dynamic,
        prediction=prediction,
        input_norm=input_norm,
        left_projection=True,
        residual=True,
        classifier_hidden=5,
    )
    copy_weights(model, reference)
    x = torch.randn(5, 4)
    data = SimpleNamespace(x=x, edge_index=index)
    with torch.no_grad():
        expected = reference(data)
        actual = model(x, index, roles)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    record_property("node_max_abs_error", float((actual - expected).abs().max()))
    h = torch.sparse_coo_tensor(index, torch.ones(index.size(1)), (5, 2)).coalesce()
    with torch.no_grad():
        expected_edge = reference(data, reduce_matrix=h)
        actual_edge = model(x, index, roles, task="hyperedge")
    torch.testing.assert_close(actual_edge, expected_edge, atol=3e-5, rtol=3e-5)
    record_property(
        "edge_max_abs_error", float((actual_edge - expected_edge).abs().max())
    )


@pytest.mark.parametrize("family", ["diagonal", "orthogonal", "general"])
@pytest.mark.parametrize("input_norm", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("task", ["node", "hyperedge"])
def test_upstream_training_gradients_and_steps(
    upstream, family, input_norm, dynamic, task, record_property
):
    """Compare actual upstream backward and three Adam updates, without adapters."""
    torch.set_num_threads(2)
    torch.manual_seed(24)
    index = torch.tensor([[0, 1, 2, 1, 3, 4], [0, 0, 0, 1, 1, 1]])
    roles = torch.tensor([-1, 1, 1, -1, 1, 1])
    args = reference_args(family, dynamic, input_norm=input_norm)
    reference = upstream.SheafGeDi(
        4,
        2,
        roles,
        0.25,
        args,
        {
            "diagonal": "DiagSheafs",
            "general": "GeneralSheafs",
            "orthogonal": "OrthoSheafs",
        }[family],
    ).double()
    reference.reset_parameters()
    model = DHSNNModel(
        4,
        2,
        stalk_dim=2,
        hidden_dim=3,
        variant=DHSNNVariant[family.upper()],
        dynamic_sheaf=dynamic,
        input_norm=input_norm,
        left_projection=True,
        residual=True,
        classifier_hidden=5,
    ).double()
    pairs = copy_weights(model, reference)
    x = torch.randn(5, 4, dtype=torch.float64)
    labels = torch.tensor([0, 1, 0, 0, 1] if task == "node" else [0, 1])
    data = SimpleNamespace(x=x, edge_index=index)
    reduction = (
        torch.sparse_coo_tensor(
            index, torch.ones(index.size(1), dtype=torch.float64), (5, 2)
        ).coalesce()
        if task == "hyperedge"
        else None
    )
    optimizers = [
        torch.optim.Adam(m.parameters(), lr=0.001) for m in (model, reference)
    ]
    max_error = 0.0
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad()
        a = model(x, index, roles, task=task)
        b = reference(data, reduce_matrix=reduction)
        # Upstream creates degree/count tensors in float32 even for a double
        # model; whitening can amplify that rounding near rank-one covariance.
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        loss_a, loss_b = [torch.nn.functional.cross_entropy(v, labels) for v in (a, b)]
        loss_a.backward()
        loss_b.backward()
        for ours, theirs in pairs:
            grad = theirs.grad if theirs.is_leaf else theirs._base.grad.T
            torch.testing.assert_close(ours.grad, grad, atol=2e-6, rtol=2e-6)
            max_error = max(max_error, float((ours.grad - grad).abs().max()))
        for optimizer in optimizers:
            optimizer.step()
        for ours, theirs in pairs:
            torch.testing.assert_close(ours, theirs, atol=2e-6, rtol=2e-6)
    record_property("gradient_max_abs_error", max_error)


@pytest.mark.parametrize(
    "norm", ["degree_norm", "sym_degree_norm", "block_norm", "sym_block_norm"]
)
@pytest.mark.parametrize("family", ["diagonal", "general", "orthogonal"])
def test_double_precision_normalizations(upstream, norm, family):
    torch.manual_seed(31)
    index = torch.tensor([[0, 1, 2, 1, 3, 4], [0, 0, 0, 1, 1, 1]])
    roles = torch.tensor([-1, 1, 1, -1, 1, 1])
    args = reference_args(family, norm=norm)
    args.sheaf_left_proj = False
    reference = upstream.SheafGeDi(
        4,
        2,
        roles,
        0.17,
        args,
        {
            "diagonal": "DiagSheafs",
            "general": "GeneralSheafs",
            "orthogonal": "OrthoSheafs",
        }[family],
    ).double()
    reference.reset_parameters()
    model = DHSNNModel(
        4,
        2,
        stalk_dim=2,
        hidden_dim=3,
        variant=DHSNNVariant[family.upper()],
        norm_type=norm,
        input_norm=False,
        residual=True,
        classifier_hidden=5,
        q=0.17,
    ).double()
    copy_weights(model, reference)
    x = torch.randn(5, 4, dtype=torch.float64)
    with torch.no_grad():
        expected = reference(SimpleNamespace(x=x, edge_index=index))
        actual = model(x, index, roles)
    # Upstream encode_directionality stores its phase as complex64.
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_performance(upstream, device):
    """Export warmed training-step timings; this is not a speed assertion."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(25)
    edge = torch.arange(128).repeat_interleave(8)
    node = (edge * 7 + torch.arange(8).repeat(128) * 19) % 256
    index = torch.stack((node, edge)).to(device)
    roles = torch.where(torch.arange(edge.numel()) % 8 == 0, -1, 1).to(device)
    args = reference_args("general", input_norm=True)
    args.cuda = 0 if device == "cuda" else -1
    reference = upstream.SheafGeDi(4, 2, roles, 0.25, args, "GeneralSheafs").to(device)
    reference.reset_parameters()
    model = DHSNNModel(
        4,
        2,
        stalk_dim=2,
        hidden_dim=3,
        variant=DHSNNVariant.GENERAL,
        input_norm=True,
        left_projection=True,
        residual=True,
        classifier_hidden=5,
    ).to(device)
    copy_weights(model, reference)
    x = torch.randn(256, 4, device=device)
    data = SimpleNamespace(x=x, edge_index=index)
    results = {}
    for name, module, forward in (
        ("message_passing", model, lambda: model(x, index, roles)),
        ("official_sparse", reference, lambda: reference(data)),
    ):

        def step(module=module, forward=forward):
            module.zero_grad(set_to_none=True)
            forward().square().mean().backward()

        for _ in range(3):
            step()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(10):
            step()
        if device == "cuda":
            torch.cuda.synchronize()
        results[name] = {
            "step_ms": (time.perf_counter() - start) * 100,
            "parameters": sum(p.numel() for p in module.parameters()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if device == "cuda"
            else None,
        }
    output = Path("tmp/dshn-performance.json")
    report = json.loads(output.read_text()) if output.exists() else {}
    report[device] = {
        "nodes": 256,
        "hyperedges": 128,
        "incidences": 1024,
        "d": 2,
        "f": 3,
        "device": torch.cuda.get_device_name(0)
        if device == "cuda"
        else "CPU, 2 threads",
        **results,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def test_reference_input_gradients(upstream):
    torch.manual_seed(19)
    index = torch.tensor([[0, 1, 2, 1, 3, 4], [0, 0, 0, 1, 1, 1]])
    roles = torch.tensor([-1, 1, 1, -1, 1, 1])
    args = reference_args("diagonal")
    reference = upstream.SheafGeDi(4, 2, roles, 0.25, args, "DiagSheafs").double()
    reference.reset_parameters()
    model = DHSNNModel(
        4,
        2,
        stalk_dim=2,
        hidden_dim=3,
        input_norm=False,
        left_projection=True,
        residual=True,
        classifier_hidden=5,
    ).double()
    copy_weights(model, reference)
    x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    model(x, index, roles).square().mean().backward()
    reference(SimpleNamespace(x=other, edge_index=index)).square().mean().backward()
    torch.testing.assert_close(x.grad, other.grad, atol=2e-6, rtol=2e-6)
