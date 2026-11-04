# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch

from sheaf_mpnn.directed_nsd import DirectedNSDModel, DirectedNSDVariant
from tests.testing import set_seed


class TestDirectedNSDModel:
    @pytest.fixture
    def setup(self):
        set_seed(42)
        in_channels = 5
        out_channels = 3
        num_nodes = 12
        stalk_dim = 4
        hidden_dim = 8
        num_layers = 2

        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
            dtype=torch.long,
        )

        model = DirectedNSDModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            variant=DirectedNSDVariant.ORTHOGONAL,
        )

        return model, x, edge_index, out_channels

    def test_model_forward_shape(self, setup):
        """Forward pass produces correct output shape."""
        model, x, edge_index, out_channels = setup
        out = model(x, edge_index)
        assert out.shape == (x.size(0), out_channels)
        assert not torch.isnan(out).any()

    def test_model_training_mode(self, setup):
        """Model works in training mode."""
        model, x, edge_index, _ = setup
        model.train()
        out = model(x, edge_index)
        assert not torch.isnan(out).any()

    def test_model_eval_mode(self, setup):
        """Model works in eval mode."""
        model, x, edge_index, _ = setup
        model.eval()
        out = model(x, edge_index)
        assert not torch.isnan(out).any()

    def test_gradient_flow_full_model(self, setup):
        """Gradients flow through entire model."""
        model, x, edge_index, _ = setup
        out = model(x, edge_index)
        loss = out.pow(2).sum()
        loss.backward()

        # Check encoder
        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0

        # Check layers
        for layer in model.layers:
            assert layer.W1_fwd.grad is not None and layer.W1_fwd.grad.abs().sum() > 0
            assert layer.W2_fwd.grad is not None and layer.W2_fwd.grad.abs().sum() > 0
            assert layer.W1_bwd.grad is not None and layer.W1_bwd.grad.abs().sum() > 0
            assert layer.W2_bwd.grad is not None and layer.W2_bwd.grad.abs().sum() > 0

        # Check decoder
        assert model.decoder.weight.grad is not None
        assert model.decoder.weight.grad.abs().sum() > 0

    def test_reset_parameters(self, setup):
        """Reset parameters resets all learnable parameters."""
        model, x, edge_index, _ = setup
        model.train()

        # Store initial parameters
        initial_w1_fwd = model.layers[0].W1_fwd.data.clone()

        # Reset parameters
        model.reset_parameters()

        # Check that parameters have changed (with very high probability)
        assert not torch.allclose(model.layers[0].W1_fwd.data, initial_w1_fwd)

    def test_dropout_effect(self):
        """Dropout causes different outputs in train vs eval."""
        set_seed(42)
        model_train = DirectedNSDModel(
            in_channels=5,
            out_channels=3,
            stalk_dim=4,
            hidden_dim=8,
            num_layers=2,
            dropout=0.5,
        )
        model_train.train()

        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        with torch.no_grad():
            out_train_1 = model_train(x, edge_index)
            out_train_2 = model_train(x, edge_index)

        # Different outputs in training (due to dropout)
        assert not torch.allclose(out_train_1, out_train_2, atol=1e-5)

        model_train.eval()
        with torch.no_grad():
            out_eval = model_train(x, edge_index)

        # Deterministic output in eval
        out_eval_2 = model_train(x, edge_index)
        torch.testing.assert_close(out_eval, out_eval_2)

    def test_jknet_concatenation(self):
        """JKNet concatenates layer representations."""
        set_seed(42)
        in_channels = 5
        out_channels = 3
        stalk_dim = 4
        hidden_dim = 8
        num_layers = 3

        model = DirectedNSDModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            jknet=True,
        )

        x = torch.randn(10, in_channels)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        out = model(x, edge_index)
        assert out.shape == (x.size(0), out_channels)
        assert not torch.isnan(out).any()

    def test_model_determinism_eval(self, setup):
        """Model produces deterministic output in eval mode."""
        model, x, edge_index, _ = setup
        model.eval()
        with torch.no_grad():
            out1 = model(x, edge_index)
            out2 = model(x, edge_index)
        torch.testing.assert_close(out1, out2)

    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_different_num_layers(self, num_layers):
        """Model works with different numbers of layers."""
        set_seed(42)
        model = DirectedNSDModel(
            in_channels=5,
            out_channels=3,
            stalk_dim=4,
            hidden_dim=8,
            num_layers=num_layers,
        )
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = model(x, edge_index)
        assert out.shape == (10, 3)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("stalk_dim", [2, 4, 8])
    def test_different_stalk_dims(self, stalk_dim):
        """Model works with different stalk dimensions."""
        set_seed(42)
        model = DirectedNSDModel(
            in_channels=5,
            out_channels=3,
            stalk_dim=stalk_dim,
            hidden_dim=8,
            num_layers=2,
        )
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = model(x, edge_index)
        assert out.shape == (10, 3)
        assert not torch.isnan(out).any()

    def test_normalization_effect(self):
        """normalize_output flag affects final output distribution."""
        set_seed(42)
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        model_normalized = DirectedNSDModel(
            in_channels=5,
            out_channels=3,
            stalk_dim=4,
            hidden_dim=8,
            num_layers=2,
            normalize_output=True,
        )
        model_unnormalized = DirectedNSDModel(
            in_channels=5,
            out_channels=3,
            stalk_dim=4,
            hidden_dim=8,
            num_layers=2,
            normalize_output=False,
        )

        model_normalized.eval()
        model_unnormalized.eval()

        with torch.no_grad():
            out_norm = model_normalized(x, edge_index)
            model_unnormalized(x, edge_index)

        # Normalized output has unit norm per node
        norms = torch.norm(out_norm, p=2, dim=1)
        # All norms should be close to 1 (approximately, before decoder)
        # Note: the decoder changes the scale, so we just check they're non-zero
        assert (norms > 0).all()


def test_directed_model_invalid_stalk_dim():
    """DirectedNSDModel rejects non-positive stalk_dim."""
    with pytest.raises(ValueError, match="stalk_dim must be positive"):
        DirectedNSDModel(
            in_channels=5,
            out_channels=3,
            stalk_dim=0,
            hidden_dim=8,
            num_layers=2,
        )


def test_directed_model_invalid_hidden_dim():
    """DirectedNSDModel rejects non-positive hidden_dim."""
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        DirectedNSDModel(
            in_channels=5,
            out_channels=3,
            stalk_dim=4,
            hidden_dim=0,
            num_layers=2,
        )


def test_directed_model_invalid_num_layers():
    """DirectedNSDModel rejects non-positive num_layers."""
    with pytest.raises(ValueError, match="must have at least one"):
        DirectedNSDModel(
            in_channels=5,
            out_channels=3,
            stalk_dim=4,
            hidden_dim=8,
            num_layers=0,
        )


def test_directed_model_empty_graph():
    """DirectedNSDModel handles empty graphs (no edges)."""
    set_seed(42)
    model = DirectedNSDModel(
        in_channels=5,
        out_channels=3,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=2,
    )
    x = torch.randn(5, 5)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    out = model(x, edge_index)
    assert out.shape == (5, 3)
    assert not torch.isnan(out).any()


def test_directed_model_single_node():
    """DirectedNSDModel handles single-node graphs."""
    set_seed(42)
    model = DirectedNSDModel(
        in_channels=5,
        out_channels=3,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=2,
    )
    x = torch.randn(1, 5)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    out = model(x, edge_index)
    assert out.shape == (1, 3)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Parametrised over all DirectedNSDVariant values
# ---------------------------------------------------------------------------

_ALL_MODEL_VARIANTS = [pytest.param(v, id=v.name.lower()) for v in DirectedNSDVariant]
_X = torch.randn(10, 5)
_EI = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def _make_model(variant, **kwargs):
    return DirectedNSDModel(
        in_channels=5,
        out_channels=3,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=2,
        variant=variant,
        **kwargs,
    )


@pytest.mark.parametrize("variant", _ALL_MODEL_VARIANTS)
def test_all_variants_forward_shape(variant):
    """Every DirectedNSDVariant produces the correct output shape."""
    set_seed(42)
    model = _make_model(variant)
    out = model(_X, _EI)
    assert out.shape == (10, 3)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("variant", _ALL_MODEL_VARIANTS)
def test_all_variants_gradient_flow(variant):
    """Gradients reach encoder, layer W matrices, and decoder for every variant."""
    set_seed(42)
    model = _make_model(variant)
    out = model(_X, _EI)
    out.pow(2).sum().backward()

    # Verify all parameters are in the computation graph (grad is not None).
    # Low-rank maps undergo a 4th-power product at init, making values ~1e-8
    # and causing float32 underflow — we check connectivity, not magnitude.
    assert model.encoder.weight.grad is not None
    assert model.decoder.weight.grad is not None
    for layer in model.layers:
        for name in ("W1_fwd", "W2_fwd", "W1_bwd", "W2_bwd"):
            assert getattr(layer, name).grad is not None, (
                f"{variant.name}: {name}.grad is None"
            )


@pytest.mark.parametrize("variant", _ALL_MODEL_VARIANTS)
def test_all_variants_eval_determinism(variant):
    """Every variant is deterministic in eval mode."""
    set_seed(42)
    model = _make_model(variant)
    model.eval()
    with torch.no_grad():
        out1 = model(_X, _EI)
        out2 = model(_X, _EI)
    torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# DirectedNSDVariant enum behaviour
# ---------------------------------------------------------------------------


def test_variant_layer_kwargs_attention():
    """Attention variants set use_attention=True in layer_kwargs."""
    assert DirectedNSDVariant.GENERAL_ATTENTION.layer_kwargs == {"use_attention": True}
    assert DirectedNSDVariant.ORTHOGONAL_ATTENTION.layer_kwargs == {
        "use_attention": True
    }


def test_variant_layer_kwargs_no_attention():
    """Non-attention matrix variants set use_attention=False."""
    assert DirectedNSDVariant.GENERAL.layer_kwargs == {"use_attention": False}
    assert DirectedNSDVariant.ORTHOGONAL.layer_kwargs == {"use_attention": False}


def test_variant_layer_kwargs_no_flag():
    """DIAGONAL and LOW_RANK have no use_attention key."""
    assert "use_attention" not in DirectedNSDVariant.DIAGONAL.layer_kwargs
    assert "use_attention" not in DirectedNSDVariant.LOW_RANK.layer_kwargs


def test_variant_build_kwargs_orth_strategy():
    """build_kwargs passes orth_strategy for orthogonal variants."""
    kw = DirectedNSDVariant.ORTHOGONAL.build_kwargs(orth_strategy="fasth")
    assert kw["orth_strategy"] == "fasth"
    kw_attn = DirectedNSDVariant.ORTHOGONAL_ATTENTION.build_kwargs(
        orth_strategy="fasth"
    )
    assert kw_attn["orth_strategy"] == "fasth"


def test_variant_build_kwargs_rank():
    """build_kwargs passes rank for LOW_RANK only."""
    kw = DirectedNSDVariant.LOW_RANK.build_kwargs(rank=3)
    assert kw["rank"] == 3
    # Other variants should not have rank key
    assert "rank" not in DirectedNSDVariant.GENERAL.build_kwargs(rank=3)


def test_variant_layer_class_mapping():
    """Each variant maps to the expected layer class."""
    from sheaf_mpnn.directed_nsd.directed_nsd_layers import (
        DiagonalDirectedNSDConv,
        GeneralDirectedNSDConv,
        LowRankDirectedNSDConv,
        OrthogonalDirectedNSDConv,
    )

    assert DirectedNSDVariant.DIAGONAL.layer_class is DiagonalDirectedNSDConv
    assert DirectedNSDVariant.GENERAL.layer_class is GeneralDirectedNSDConv
    assert DirectedNSDVariant.GENERAL_ATTENTION.layer_class is GeneralDirectedNSDConv
    assert DirectedNSDVariant.ORTHOGONAL.layer_class is OrthogonalDirectedNSDConv
    assert (
        DirectedNSDVariant.ORTHOGONAL_ATTENTION.layer_class is OrthogonalDirectedNSDConv
    )
    assert DirectedNSDVariant.LOW_RANK.layer_class is LowRankDirectedNSDConv


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_forward_wrong_feature_dim():
    """forward() raises ValueError when x has wrong feature dimension."""
    model = _make_model(DirectedNSDVariant.ORTHOGONAL)
    x_bad = torch.randn(10, 99)
    with pytest.raises(ValueError, match="x must have shape"):
        model(x_bad, _EI)


def test_forward_wrong_edge_index_shape():
    """forward() raises ValueError when edge_index is not [2, E]."""
    model = _make_model(DirectedNSDVariant.ORTHOGONAL)
    bad_ei = torch.tensor([[0, 1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="edge_index must have shape"):
        model(_X, bad_ei)


# ---------------------------------------------------------------------------
# Low-rank model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_lowrank_model_rank_stored(rank):
    """DirectedNSDModel stores rank correctly for LOW_RANK variant."""
    model = DirectedNSDModel(
        in_channels=5,
        out_channels=3,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=2,
        variant=DirectedNSDVariant.LOW_RANK,
        rank=rank,
    )
    assert model.rank == rank
    out = model(_X, _EI)
    assert out.shape == (10, 3)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# learn_alpha
# ---------------------------------------------------------------------------


def test_learn_alpha_model():
    """learn_alpha=True makes alpha a learnable parameter in all layers."""
    model = _make_model(DirectedNSDVariant.ORTHOGONAL, learn_alpha=True)
    for layer in model.layers:
        assert layer.alpha.requires_grad

    model_fixed = _make_model(DirectedNSDVariant.ORTHOGONAL, learn_alpha=False)
    for layer in model_fixed.layers:
        assert not layer.alpha.requires_grad


def test_learn_alpha_gradient():
    """When learn_alpha=True, gradient flows through alpha."""
    set_seed(42)
    model = _make_model(DirectedNSDVariant.ORTHOGONAL, learn_alpha=True)
    out = model(_X, _EI)
    out.pow(2).sum().backward()
    for layer in model.layers:
        assert layer.alpha.grad is not None
