# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""End-to-end BuNNModel: shapes, options, training behaviour."""

import pytest
import torch

from sheaf_mpnn.bunn import BuNNModel
from tests.bunn.test_bunn_bundle import ring_edge_index

NUM_NODES = 8
IN_CHANNELS = 10
OUT_CHANNELS = 3


@pytest.fixture
def edge_index() -> torch.Tensor:
    return ring_edge_index(NUM_NODES)


@pytest.fixture
def x() -> torch.Tensor:
    return torch.randn(NUM_NODES, IN_CHANNELS)


def make_model(**kwargs) -> BuNNModel:
    defaults = {
        "in_channels": IN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "bundle_dim": 2,
        "hidden_dim": 4,
        "num_bundles": 3,
        "num_layers": 2,
    }
    defaults.update(kwargs)
    return BuNNModel(**defaults)  # ty: ignore[invalid-argument-type]


class TestForward:
    @pytest.mark.parametrize("solver", ["taylor", "spectral"])
    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_output_shape(self, solver, num_layers, x, edge_index):
        model = make_model(solver=solver, num_layers=num_layers)
        out = model(x, edge_index)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("bundle_dim", [2, 3, 4])
    def test_bundle_dimensions(self, bundle_dim, x, edge_index):
        model = make_model(bundle_dim=bundle_dim)
        assert model.width == 3 * bundle_dim * 4
        assert model(x, edge_index).shape == (NUM_NODES, OUT_CHANNELS)

    @pytest.mark.parametrize("phi", ["mlp", "sum_gnn"])
    @pytest.mark.parametrize("shared_phi", [False, True])
    def test_map_learner_options(self, phi, shared_phi, x, edge_index):
        model = make_model(phi=phi, shared_phi=shared_phi)
        assert (model.shared_map_learner is not None) == shared_phi
        assert (model.layers[0].map_learner is None) == shared_phi
        assert model(x, edge_index).shape == (NUM_NODES, OUT_CHANNELS)

    def test_shared_phi_uses_one_learner_for_every_layer(self, x, edge_index):
        shared = make_model(shared_phi=True, num_layers=3)
        separate = make_model(shared_phi=False, num_layers=3)

        shared_learners = sum(
            1 for n, _ in shared.named_parameters() if "map_learner" in n
        )
        separate_learners = sum(
            1 for n, _ in separate.named_parameters() if "map_learner" in n
        )
        assert shared_learners * 3 == separate_learners

    def test_jknet_concatenates_layer_states(self, x, edge_index):
        model = make_model(jknet=True, num_layers=3)
        assert model.decoder.in_features == 3 * model.width
        assert model(x, edge_index).shape == (NUM_NODES, OUT_CHANNELS)

    def test_normalize_output(self, x, edge_index):
        model = make_model(normalize_output=True)
        model.decoder = torch.nn.Identity()  # ty: ignore[invalid-assignment]
        out = model(x, edge_index)
        assert torch.allclose(out.norm(dim=-1), torch.ones(NUM_NODES), atol=1e-5)

    def test_rejects_malformed_inputs(self, edge_index):
        model = make_model()
        with pytest.raises(ValueError, match="x must have shape"):
            model(torch.randn(NUM_NODES, IN_CHANNELS + 1), edge_index)
        with pytest.raises(ValueError, match="edge_index must have shape"):
            model(
                torch.randn(NUM_NODES, IN_CHANNELS), torch.zeros(3, 4, dtype=torch.long)
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bundle_dim": 0},
            {"hidden_dim": 0},
            {"num_bundles": 0},
            {"num_layers": 0},
            {"phi": "transformer"},
        ],
    )
    def test_invalid_arguments_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            make_model(**kwargs)


class TestTraining:
    @pytest.mark.parametrize("solver", ["taylor", "spectral"])
    def test_gradients_reach_every_parameter(self, solver, x, edge_index):
        model = make_model(solver=solver, learn_tau=True)
        model(x, edge_index).sum().backward()

        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == []

    def test_loss_decreases_on_a_separable_task(self, edge_index):
        """A short optimisation run: the model must be able to fit something."""
        torch.manual_seed(0)
        model = make_model(num_layers=2, tau=1.0)
        x = torch.randn(NUM_NODES, IN_CHANNELS)
        y = torch.randint(0, OUT_CHANNELS, (NUM_NODES,))
        opt = torch.optim.Adam(model.parameters(), lr=0.05)

        losses = []
        for _ in range(40):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x, edge_index), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]

    def test_dropout_only_perturbs_training_mode(self, x, edge_index):
        model = make_model(dropout=0.5, input_dropout=0.5)
        model.eval()
        assert torch.allclose(model(x, edge_index), model(x, edge_index))

        model.train()
        assert not torch.allclose(model(x, edge_index), model(x, edge_index))

    def test_reset_parameters_changes_the_output(self, x, edge_index):
        model = make_model()
        model.eval()
        before = model(x, edge_index)
        model.reset_parameters()
        assert not torch.allclose(before, model(x, edge_index))


class TestSpectralCache:
    def test_eigendecomposition_is_cached_per_graph(self, x, edge_index):
        model = make_model(solver="spectral")
        model.eval()
        assert model._eig_cache is None

        model(x, edge_index)
        cached = model._eig_cache
        assert cached is not None

        model(x, edge_index)
        assert model._eig_cache is cached

    def test_cache_is_invalidated_by_a_different_graph(self, x, edge_index):
        model = make_model(solver="spectral")
        model.eval()
        model(x, edge_index)
        first = model._eig_cache

        # A different graph, still undirected: drop one edge in both directions.
        keep = torch.ones(edge_index.size(1), dtype=torch.bool)
        keep[0] = keep[edge_index.size(1) // 2] = False
        model(x, edge_index[:, keep])
        assert model._eig_cache is not first

    def test_cached_and_uncached_outputs_agree(self, x, edge_index):
        model = make_model(solver="spectral")
        model.eval()
        first = model(x, edge_index)
        second = model(x, edge_index)  # served from cache
        assert torch.allclose(first, second, atol=1e-6)


class TestPositionalEncodings:
    def test_pe_reaches_phi(self, x, edge_index):
        """Eq. 1 is ``phi(G, P, X, v)``: changing P must change the maps, and
        so the output.
        """
        model = make_model(pe_size=4)
        model.eval()
        pe = torch.randn(NUM_NODES, 4)
        assert not torch.allclose(model(x, edge_index, pe), model(x, edge_index, -pe))

    def test_pe_is_required_when_declared(self, x, edge_index):
        model = make_model(pe_size=4)
        with pytest.raises(ValueError, match="pass pe to forward"):
            model(x, edge_index)

    def test_pe_width_is_checked(self, x, edge_index):
        model = make_model(pe_size=4)
        with pytest.raises(ValueError, match="pe has width"):
            model(x, edge_index, torch.randn(NUM_NODES, 3))

    def test_pe_works_with_shared_phi(self, x, edge_index):
        model = make_model(pe_size=4, shared_phi=True, num_layers=2)
        out = model(x, edge_index, torch.randn(NUM_NODES, 4))
        assert out.shape == (NUM_NODES, OUT_CHANNELS)

    def test_negative_pe_size_is_rejected(self):
        with pytest.raises(ValueError, match="pe_size"):
            make_model(pe_size=-1)


class TestOverSmoothing:
    r"""Prop. 4.1/4.2: on a non-trivial bundle the heat flow's fixed points obey
    :math:`O_v y_v = O_u y_u`, so node information survives depth.

    Both propositions assume distinguishable nodes -- Prop. 4.2 states it
    outright ("assume that there are two nodes u, v with :math:`p_u \\neq
    p_v`") -- and that assumption is load-bearing rather than cosmetic, as the
    contrast between these two tests shows.
    """

    def _spread(self, model, x, edge_index, pe=None):
        model.eval()
        model.decoder = torch.nn.Identity()
        out = model(x, edge_index, pe) if pe is not None else model(x, edge_index)
        pairwise = torch.cdist(out, out)
        return pairwise[~torch.eye(NUM_NODES, dtype=torch.bool)].mean()

    def test_deep_stacks_stay_distinguishable_with_positional_encodings(
        self, x, edge_index
    ):
        torch.manual_seed(0)
        model = make_model(
            num_layers=8, tau=10.0, solver="spectral", hidden_dim=8, pe_size=8
        )
        pe = torch.eye(NUM_NODES)[:, :8]  # injective, as Theorem 5.3 assumes
        assert self._spread(model, x, edge_index, pe) > 1e-2

    def test_feature_only_phi_drifts_towards_a_trivial_bundle(self, x, edge_index):
        """Without P, phi sees only features it produced itself. A long
        diffusion flattens those, which flattens the maps, which flattens the
        features further -- the bundle degenerates and over-smoothing returns.
        This is why the paper feeds phi positional encodings.
        """
        torch.manual_seed(0)
        model = make_model(num_layers=8, tau=10.0, solver="spectral", hidden_dim=8)
        assert self._spread(model, x, edge_index) < 1e-4
