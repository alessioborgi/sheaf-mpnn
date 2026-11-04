# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for the end-to-end NLSD model."""

import random

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph, to_undirected

from sheaf_mpnn.nlsd.nlsd_model import NLSDModel, NLSDNonlinearity, NLSDVariant
from sheaf_mpnn.nlsd.nonlinearity import (
    EdgeThresholdNonlinearity,
    IdentityNonlinearity,
    MLPNonlinearity,
    ScalarThresholdNonlinearity,
)
from sheaf_mpnn.nsd.nsd_model import NSDModel, NSDVariant

NUM_NODES, IN_CHANNELS, OUT_CHANNELS = 15, 10, 3
STALK_DIM, HIDDEN_DIM, NUM_LAYERS = 2, 4, 2
RANK = 2


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_model(**kwargs):
    defaults = dict(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        stalk_dim=STALK_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )
    defaults.update(kwargs)
    return NLSDModel(**defaults)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("variant", list(NLSDVariant))
@pytest.mark.parametrize("nonlinearity", list(NLSDNonlinearity))
class TestNLSDModel:
    @pytest.fixture
    def setup(self, variant, nonlinearity):
        set_seed(42)
        x = torch.randn(NUM_NODES, IN_CHANNELS)
        edge_index = to_undirected(erdos_renyi_graph(NUM_NODES, edge_prob=0.4))
        model = make_model(variant=variant, nonlinearity=nonlinearity)
        return model, x, edge_index

    def test_model_forward_shape(self, setup):
        model, x, edge_index = setup
        out = model(x, edge_index)
        assert out.shape == (x.size(0), model.out_channels)
        assert torch.isfinite(out).all()

    def test_model_gradient_flow(self, setup):
        model, x, edge_index = setup
        model(x, edge_index).pow(2).mean().backward()
        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0
        assert model.layers[0].W1.grad is not None
        assert model.layers[0].W1.grad.abs().sum() > 0

    def test_nonlinearity_parameters_receive_gradient(self, setup):
        """Phi is learned, so it must actually be on the backward path."""
        model, x, edge_index = setup
        phi = model.layers[0].nonlinearity
        params = list(phi.parameters())
        if not params:  # IdentityNonlinearity has nothing to learn.
            pytest.skip("nonlinearity has no parameters")
        model(x, edge_index).pow(2).mean().backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in params)

    def test_model_determinism(self, setup):
        model, x, edge_index = setup
        model.eval()
        with torch.no_grad():
            first = model(x, edge_index)
            second = model(x, edge_index)
        torch.testing.assert_close(first, second)

    def test_isolated_graph_stability(self, setup):
        model, x, _ = setup
        out = model(x, torch.empty((2, 0), dtype=torch.long))
        assert out.shape == (x.size(0), model.out_channels)
        assert torch.isfinite(out).all()

    def test_permutation_equivariance(self, setup):
        """Relabelling the nodes must permute the predictions, nothing more.

        The layer evaluates the node-wise Eq. 8 rather than fixing an edge
        orientation, so this holds for the learned MLP too -- not only for the
        odd psi gates.
        """
        model, x, edge_index = setup
        model.eval()
        perm = torch.randperm(NUM_NODES)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(NUM_NODES)
        with torch.no_grad():
            out = model(x, edge_index)
            relabelled = model(x[perm], inverse[edge_index])
        torch.testing.assert_close(relabelled, out[perm], atol=1e-5, rtol=1e-4)

    def test_reset_parameters_runs(self, setup):
        model, x, edge_index = setup
        model.reset_parameters()
        assert torch.isfinite(model(x, edge_index)).all()


class TestNLSDModelWiring:
    """The layer stack has to receive what the constructor was asked for."""

    @pytest.mark.parametrize(
        "nonlinearity,expected",
        [
            (NLSDNonlinearity.MLP, MLPNonlinearity),
            (NLSDNonlinearity.SCALAR_THRESHOLD, ScalarThresholdNonlinearity),
            (NLSDNonlinearity.EDGE_THRESHOLD, EdgeThresholdNonlinearity),
            (NLSDNonlinearity.IDENTITY, IdentityNonlinearity),
        ],
    )
    def test_nonlinearity_class(self, nonlinearity, expected):
        model = make_model(nonlinearity=nonlinearity)
        assert all(isinstance(x.nonlinearity, expected) for x in model.layers)

    def test_each_layer_owns_its_own_phi(self):
        """Phi is learned per diffusion step, exactly like the restriction maps."""
        model = make_model(nonlinearity=NLSDNonlinearity.MLP, num_layers=3)
        phis = [layer.nonlinearity for layer in model.layers]
        assert len({id(p) for p in phis}) == 3

    def test_per_stalk_alpha_defaults_follow_the_reference(self):
        """Threshold variants gate the diffusion per stalk; the MLP does not."""
        gated = make_model(nonlinearity=NLSDNonlinearity.SCALAR_THRESHOLD)
        plain = make_model(nonlinearity=NLSDNonlinearity.MLP)
        assert all(hasattr(layer, "alpha_gate") for layer in gated.layers)
        assert all(not hasattr(layer, "alpha_gate") for layer in plain.layers)

    def test_per_stalk_alpha_can_be_overridden(self):
        model = make_model(nonlinearity=NLSDNonlinearity.MLP, per_stalk_alpha=True)
        assert all(hasattr(layer, "alpha_gate") for layer in model.layers)

    def test_psi_reaches_the_nonlinearity(self):
        model = make_model(nonlinearity=NLSDNonlinearity.SCALAR_THRESHOLD, psi="cos")
        assert all(layer.nonlinearity.psi == "cos" for layer in model.layers)

    def test_mlp_depth_reaches_the_nonlinearity(self):
        model = make_model(nonlinearity=NLSDNonlinearity.MLP, num_mlp_layers=3)
        linears = [
            m
            for m in model.layers[0].nonlinearity.mlp
            if isinstance(m, torch.nn.Linear)
        ]
        assert len(linears) == 3

    def test_learn_alpha_false_freezes_step_size(self):
        model = make_model(learn_alpha=False)
        assert all(not layer.alpha.requires_grad for layer in model.layers)

    def test_jknet_widens_the_decoder(self):
        model = make_model(jknet=True, num_layers=3)
        assert model.decoder.in_features == 3 * STALK_DIM * HIDDEN_DIM

    def test_second_linear_adds_an_encoder_stage(self):
        assert make_model(second_linear=True).encoder2 is not None
        assert make_model().encoder2 is None

    def test_sparse_learner_shrinks_the_map_generator(self):
        sparse = make_model(sparse_learner=True)
        dense = make_model()
        assert (
            sparse.layers[0].map_generator[0].in_features
            < dense.layers[0].map_generator[0].in_features
        )

    def test_edge_weights_add_a_weight_generator(self):
        model = make_model(variant=NLSDVariant.ORTHOGONAL, use_edge_weights=True)
        assert all(hasattr(layer, "weight_generator") for layer in model.layers)

    def test_rank_reaches_the_low_rank_layers(self):
        model = make_model(variant=NLSDVariant.LOW_RANK, rank=3)
        assert all(layer.rank == 3 for layer in model.layers)
        # 2*d*r generator outputs: the factorisation is what the family buys.
        assert model.layers[0].map_generator[0].out_features == 2 * STALK_DIM * 3

    def test_rank_is_ignored_by_the_other_variants(self):
        """``rank`` is a low-rank-only knob and must not leak elsewhere."""
        model = make_model(variant=NLSDVariant.GENERAL, rank=3)
        assert model.layers[0].map_generator[0].out_features == STALK_DIM * STALK_DIM


class TestIdentityPhiRecoversNSDModel:
    """End to end, NLSD with Phi = identity is the linear NSD model."""

    @pytest.mark.parametrize(
        "nlsd_variant,nsd_variant",
        [
            (NLSDVariant.DIAGONAL, NSDVariant.DIAGONAL),
            (NLSDVariant.ORTHOGONAL, NSDVariant.ORTHOGONAL),
            (NLSDVariant.GENERAL, NSDVariant.GENERAL),
            (NLSDVariant.LOW_RANK, NSDVariant.LOW_RANK),
        ],
        ids=["diagonal", "orthogonal", "general", "low_rank"],
    )
    def test_matches_nsd_model(self, nlsd_variant, nsd_variant):
        set_seed(0)
        x = torch.randn(NUM_NODES, IN_CHANNELS)
        edge_index = to_undirected(erdos_renyi_graph(NUM_NODES, edge_prob=0.4))

        nlsd = make_model(
            variant=nlsd_variant,
            nonlinearity=NLSDNonlinearity.IDENTITY,
            per_stalk_alpha=False,
            rank=RANK,
        )
        nsd = NSDModel(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            stalk_dim=STALK_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            variant=nsd_variant,
            rank=RANK,
        )
        incompatible = nsd.load_state_dict(nlsd.state_dict(), strict=False)
        assert not incompatible.unexpected_keys
        nlsd.eval()
        nsd.eval()

        with torch.no_grad():
            torch.testing.assert_close(
                nlsd(x, edge_index), nsd(x, edge_index), atol=1e-5, rtol=1e-4
            )


class TestNLSDModelValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"stalk_dim": 0}, "stalk_dim must be positive"),
            ({"hidden_dim": 0}, "hidden_dim must be positive"),
            ({"num_layers": 0}, "at least one NLSD layer"),
            ({"psi": "quadratic"}, "psi must be one of"),
        ],
    )
    def test_invalid_construction_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            make_model(**kwargs)

    def test_bad_feature_width_raises(self):
        model = make_model()
        with pytest.raises(ValueError, match="x must have shape"):
            model(
                torch.randn(NUM_NODES, IN_CHANNELS + 1),
                torch.empty((2, 0), dtype=torch.long),
            )

    def test_bad_edge_index_shape_raises(self):
        model = make_model()
        with pytest.raises(ValueError, match="edge_index must have shape"):
            model(
                torch.randn(NUM_NODES, IN_CHANNELS), torch.zeros(3, 4, dtype=torch.long)
            )
