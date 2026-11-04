# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import random

import pytest
import torch
from torch import nn
from torch_geometric.utils import erdos_renyi_graph

from exp.registries.models import model_registry
from exp.registries.presets import preset_registry
from sheaf_mpnn.csnn import CSNNModel, CSNNVariant


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


@pytest.mark.parametrize("variant", list(CSNNVariant))
class TestCSNNModel:
    @pytest.fixture
    def setup(self, variant):
        set_seed(42)
        num_nodes, in_channels, out_channels = 15, 10, 3
        d, hidden_dim, num_layers = 3, 4, 2

        model = CSNNModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=d,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            variant=variant,
            gnn_layers=1,
            gnn_hidden=8,
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
        model(x, edge_index).pow(2).mean().backward()

        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0
        assert model.layers[0].epsilons.grad is not None
        assert model.layers[0].epsilons.grad.abs().sum() > 0

    def test_model_determinism(self, setup):
        model, x, edge_index = setup
        model.eval()
        torch.testing.assert_close(model(x, edge_index), model(x, edge_index))

    def test_isolated_graph_stability(self, setup):
        model, x, _ = setup
        model.eval()
        out = model(x, torch.empty((2, 0), dtype=torch.long))
        assert torch.isfinite(out).all()

    def test_reset_parameters(self, setup):
        model, x, edge_index = setup
        before = model(x, edge_index).clone()
        model.reset_parameters()
        assert not torch.allclose(before, model(x, edge_index))


class TestCSNNModelContracts:
    def test_decoder_identity_returns_stalk_features(self):
        """Graph classification swaps `decoder` for Identity and pools itself.

        The model must then emit the flat stalk representation, since
        GCLightningModule sizes its own classifier as stalk_dim * hidden_dim.
        """
        set_seed(42)
        d, hidden_dim, num_nodes = 3, 4, 12
        model = CSNNModel(
            in_channels=10, out_channels=3, stalk_dim=d, hidden_dim=hidden_dim
        ).eval()
        model.decoder = nn.Identity()
        out = model(torch.randn(num_nodes, 10), erdos_renyi_graph(num_nodes, 0.4))
        assert out.shape == (num_nodes, d * hidden_dim)

    def test_reset_parameters_tolerates_swapped_decoder(self):
        model = CSNNModel(in_channels=10, out_channels=3, stalk_dim=3, hidden_dim=4)
        model.decoder = nn.Identity()
        model.reset_parameters()  # must not raise

    def test_head_layers_stack(self):
        model = CSNNModel(
            in_channels=10,
            out_channels=3,
            stalk_dim=3,
            hidden_dim=4,
            csnn_head_layers=3,
        )
        # 2 hidden stages -> Linear, GELU, Linear, GELU
        assert isinstance(model.head, nn.Sequential)
        assert len(model.head) == 4
        assert isinstance(model.decoder, nn.Linear)

    def test_single_head_layer_is_identity(self):
        model = CSNNModel(in_channels=10, out_channels=3, stalk_dim=3, hidden_dim=4)
        assert isinstance(model.head, nn.Identity)

    def test_norm_modules_are_per_layer(self):
        model = CSNNModel(
            in_channels=10,
            out_channels=3,
            stalk_dim=3,
            hidden_dim=4,
            num_layers=3,
            layer_norm=True,
            batch_norm=True,
        )
        assert model.layer_norms is not None and len(model.layer_norms) == 3
        assert model.batch_norms is not None and len(model.batch_norms) == 3

    def test_norms_off_by_default(self):
        model = CSNNModel(in_channels=10, out_channels=3, stalk_dim=3, hidden_dim=4)
        assert model.layer_norms is None
        assert model.batch_norms is None

    def test_optional_kwargs_are_accepted(self):
        set_seed(42)
        num_nodes, pe_size = 12, 5
        model = CSNNModel(
            in_channels=10,
            out_channels=3,
            stalk_dim=3,
            hidden_dim=4,
            num_layers=1,
            pe_size=pe_size,
            gnn_default=0,
            gnn_layers=2,
            gnn_hidden=8,
            gnn_type="GAT",
        ).eval()
        x = torch.randn(num_nodes, 10)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
        out = model(
            x,
            edge_index,
            edge_attr=torch.randn(edge_index.size(1), 1),
            pe=torch.randn(num_nodes, pe_size),
        )
        assert out.shape == (num_nodes, 3)

    def test_pe_width_is_validated(self):
        model = CSNNModel(
            in_channels=10, out_channels=3, stalk_dim=3, hidden_dim=4, pe_size=5
        )
        with pytest.raises(ValueError, match="pe has width 2 but pe_size is 5"):
            model(
                torch.randn(6, 10),
                torch.empty((2, 0), dtype=torch.long),
                pe=torch.randn(6, 2),
            )

    def test_norms_apply_in_forward_and_reset(self):
        set_seed(42)
        model = CSNNModel(
            in_channels=10,
            out_channels=3,
            stalk_dim=3,
            hidden_dim=4,
            num_layers=2,
            layer_norm=True,
            batch_norm=True,
            csnn_head_layers=2,
        )
        x = torch.randn(12, 10)
        edge_index = erdos_renyi_graph(12, edge_prob=0.4)
        out = model(x, edge_index)
        assert out.shape == (12, 3)
        assert torch.isfinite(out).all()

        model.eval()
        before = model(x, edge_index).clone()
        model.reset_parameters()
        assert not torch.allclose(before, model(x, edge_index))

    def test_normalize_output_unit_norm(self):
        set_seed(42)
        model = CSNNModel(
            in_channels=10,
            out_channels=3,
            stalk_dim=3,
            hidden_dim=4,
            normalize_output=True,
        ).eval()
        model.decoder = nn.Identity()
        out = model(torch.randn(12, 10), erdos_renyi_graph(12, edge_prob=0.4))
        torch.testing.assert_close(
            out.norm(dim=-1), torch.ones(12), atol=1e-5, rtol=1e-5
        )

    def test_sheaf_learners_avoid_map_generator_group(self):
        """The reference optimizer uses one flat group, so the learner names
        must not match the module's "map_generator" sheaf_decay routing.
        """
        model = CSNNModel(in_channels=10, out_channels=3, stalk_dim=3, hidden_dim=4)
        assert not any("map_generator" in n for n, _ in model.named_parameters())

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (
                lambda: CSNNModel(
                    in_channels=10, out_channels=3, stalk_dim=1, hidden_dim=4
                ),
                "stalk_dim must be greater than 1",
            ),
            (
                lambda: CSNNModel(
                    in_channels=10, out_channels=3, stalk_dim=3, hidden_dim=0
                ),
                "hidden_dim must be positive",
            ),
            (
                lambda: CSNNModel(
                    in_channels=10,
                    out_channels=3,
                    stalk_dim=3,
                    hidden_dim=4,
                    num_layers=0,
                ),
                "must have at least one CSNN layer",
            ),
            (
                lambda: CSNNModel(
                    in_channels=10,
                    out_channels=3,
                    stalk_dim=3,
                    hidden_dim=4,
                    csnn_head_layers=0,
                ),
                "csnn_head_layers must be positive",
            ),
            (
                lambda: CSNNModel(
                    in_channels=10,
                    out_channels=3,
                    stalk_dim=3,
                    hidden_dim=4,
                    pe_size=-1,
                ),
                "pe_size must be non-negative",
            ),
        ],
    )
    def test_validation(self, build, message):
        with pytest.raises(ValueError, match=message):
            build()


class TestCSNNRegistryAndPresets:
    def test_registered(self):
        assert "csnn" in model_registry

    def test_build_returns_csnn_model(self):
        cfg = preset_registry.get("tolokers_csnn")
        model = model_registry.build("csnn", 10, 2, cfg.model, cfg.reg)
        assert isinstance(model, CSNNModel)

    def test_jknet_is_rejected(self):
        import dataclasses

        cfg = preset_registry.get("tolokers_csnn")
        with pytest.raises(ValueError, match="CSNN does not implement jknet"):
            model_registry.build(
                "csnn", 10, 2, dataclasses.replace(cfg.model, jknet=True), cfg.reg
            )

    @pytest.mark.parametrize(
        ("preset", "in_channels", "out_channels", "expected"),
        [
            # Parameter counts logged by the reference implementation. The
            # source uses a single logit for binary tasks, so out_channels is 1
            # for minesweeper/questions/tolokers here.
            ("roman_empire_csnn", 300, 18, 339_900),
            ("amazon_ratings_csnn", 300, 5, 192_299),
            ("minesweeper_csnn", 7, 1, 642_840),
            ("questions_csnn", 301, 1, 53_257),
            ("squirrel_csnn", 2089, 5, 270_665),
            ("tolokers_csnn", 10, 1, 16_151),
        ],
    )
    def test_parameter_count_matches_reference(
        self, preset, in_channels, out_channels, expected
    ):
        """Structural parity with the reference, without training anything."""
        cfg = preset_registry.get(preset)
        model = model_registry.build(
            str(cfg.model.type), in_channels, out_channels, cfg.model, cfg.reg
        )
        assert sum(p.numel() for p in model.parameters()) == expected

    @pytest.mark.parametrize(
        "preset",
        [
            "roman_empire_csnn",
            "amazon_ratings_csnn",
            "minesweeper_csnn",
            "questions_csnn",
            "squirrel_csnn",
            "tolokers_csnn",
        ],
    )
    def test_preset_uses_accuracy_stopping(self, preset):
        """The reference parser defaults to acc; OptimConfig defaults to loss."""
        assert preset_registry.get(preset).optim.stop_strategy == "acc"
