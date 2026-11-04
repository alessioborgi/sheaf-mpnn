# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Model registry: factory-based construction of sheaf models from config."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, TypeVar

from torch import nn

from exp.registries.base import Registry

if TYPE_CHECKING:
    from exp.config import ModelConfig, RegConfig


@dataclass(frozen=True)
class ModelEntry:
    """Bundles a model factory with its registry key.

    The factory receives ``(in_channels, out_channels, cfg, reg)`` and returns a
    fully-constructed ``nn.Module`` ready for training, with dropout already wired in.
    """

    factory: Callable[[int, int, ModelConfig, RegConfig], nn.Module]


class ModelRegistry(Registry[str, ModelEntry]):
    """Registry for model factories with a convenience build method."""

    def build(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        cfg: ModelConfig,
        reg: RegConfig,
    ) -> nn.Module:
        """Construct the named model from dataset dimensions and config.

        Args:
            name: Registry key matching ``ModelType.value`` (e.g. ``"nsd"``).
            in_channels: Raw input feature dimension.
            out_channels: Number of output classes.
            cfg: Model hyperparameter config.
            reg: Regularization config (dropout rates).

        Returns:
            A freshly-constructed ``nn.Module`` with dropout wired in.

        Raises:
            KeyError: If ``name`` has not been registered.
        """
        return self.get(name).factory(in_channels, out_channels, cfg, reg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


E = TypeVar("E", bound=Enum)


def _parse_variant(enum_cls: type[E], name: str) -> E:
    try:
        return enum_cls[name.upper()]
    except KeyError:
        valid = [v.name.lower() for v in enum_cls]
        raise ValueError(
            f"Invalid variant {name!r} for {enum_cls.__name__}. Valid choices: {valid}"
        ) from None


def _build_nsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.nsd import NSDModel, NSDVariant

    return NSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(NSDVariant, cfg.variant),
        alpha=cfg.alpha,
        learn_alpha=cfg.learn_alpha,
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        add_lp=cfg.add_lp,
        add_hp=cfg.add_hp,
        sparse_learner=cfg.sparse_learner,
        second_linear=cfg.second_linear,
        use_edge_weights=cfg.edge_weights,
    )


def _build_bsnn(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.bsnn import BSNNModel, BSNNVariant

    return BSNNModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(BSNNVariant, cfg.variant),
        alpha=cfg.alpha,
        rank=cfg.rank,
        learn_alpha=cfg.learn_alpha,
        orth_strategy=cfg.orth_strategy,
        clamp_val=cfg.clamp_val,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        add_lp=cfg.add_lp,
        add_hp=cfg.add_hp,
        sparse_learner=cfg.sparse_learner,
        second_linear=cfg.second_linear,
        ensemble_size=cfg.ensemble_size,
    )


def _build_nlsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.nlsd import NLSDModel, NLSDNonlinearity, NLSDVariant

    # The paper defines only the diagonal and orthogonal families; general and
    # low_rank are framework extensions sharing the same coboundary normalisation.
    return NLSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(NLSDVariant, cfg.variant),
        nonlinearity=_parse_variant(NLSDNonlinearity, cfg.nlsd_nonlinearity),
        psi=cfg.nlsd_psi,
        num_mlp_layers=cfg.nlsd_mlp_layers,
        init_threshold=cfg.nlsd_init_threshold,
        alpha=cfg.alpha,
        learn_alpha=cfg.learn_alpha,
        orth_strategy=cfg.orth_strategy,
        clamp_val=cfg.clamp_val,
        rank=cfg.rank,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        sparse_learner=cfg.sparse_learner,
        second_linear=cfg.second_linear,
        use_edge_weights=cfg.edge_weights,
    )


def _build_bunn(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.bunn import BuNNModel

    # BuNN's maps are per-node orthogonal (a flat bundle), so cfg.variant --
    # which selects a restriction-map family for the sheaf models -- does not
    # apply; stalk_dim is the bundle dimension d.
    return BuNNModel(
        in_channels=in_channels,
        out_channels=out_channels,
        bundle_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_bundles=cfg.bunn_num_bundles,
        num_layers=cfg.num_layers,
        tau=cfg.bunn_tau,
        learn_tau=cfg.bunn_learn_tau,
        solver=cfg.bunn_solver,
        max_degree=cfg.bunn_max_degree or None,
        num_eigs=cfg.bunn_num_eigs or None,
        normalize=cfg.bunn_normalize,
        phi=cfg.bunn_phi,
        num_phi_layers=cfg.bunn_phi_layers,
        shared_phi=cfg.bunn_shared_phi,
        pe_size=cfg.pe_size,
        orth_strategy=cfg.orth_strategy,
        clamp_val=cfg.clamp_val,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


def _build_directed_nsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.directed_nsd import DirectedNSDModel, DirectedNSDVariant

    # The directed layers only implement the cayley / fasth parametrisations.
    if cfg.orth_strategy not in ("cayley", "fasth"):
        raise ValueError(
            "directed_nsd supports orth_strategy 'cayley' or 'fasth', "
            f"got {cfg.orth_strategy!r}"
        )
    orth_strategy: Literal["cayley", "fasth"] = cfg.orth_strategy

    return DirectedNSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(DirectedNSDVariant, cfg.variant),
        alpha=cfg.alpha,
        rank=cfg.rank,
        orth_strategy=orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


def _build_nsp(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.nsp import NSPModel, NSPVariant

    return NSPModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(NSPVariant, cfg.variant),
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        add_lp=cfg.add_lp,
        add_hp=cfg.add_hp,
        sparse_learner=cfg.sparse_learner,
        second_linear=cfg.second_linear,
        use_edge_weights=cfg.edge_weights,
    )


def _build_sheaf_attn(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.sheaf_attn import SheafAttnModel, SheafAttnVariant

    return SheafAttnModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(SheafAttnVariant, cfg.variant),
        num_heads=cfg.num_heads,
        leaky_relu_slope=cfg.leaky_relu_slope,
        clamp_val=cfg.clamp_val,
        orth_strategy=cfg.orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        sparse_learner=cfg.sparse_learner,
        second_linear=cfg.second_linear,
    )


def _build_sheaf_adj(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.sheaf_adj import SheafAdjModel, SheafAdjVariant

    return SheafAdjModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(SheafAdjVariant, cfg.variant),
        alpha=cfg.alpha,
        learn_alpha=cfg.learn_alpha,
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        add_lp=cfg.add_lp,
        add_hp=cfg.add_hp,
        sparse_learner=cfg.sparse_learner,
        second_linear=cfg.second_linear,
        use_edge_weights=cfg.edge_weights,
    )


def _build_dnsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.dnsd import DNSDModel, DNSDVariant

    return DNSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(DNSDVariant, cfg.variant),
        alpha=cfg.alpha,
        learn_alpha=cfg.learn_alpha,
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
        odd=cfg.odd,
        gate=cfg.gate,
        layer_norm=cfg.layer_norm,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        add_lp=cfg.add_lp,
        add_hp=cfg.add_hp,
        sparse_learner=cfg.sparse_learner,
        second_linear=cfg.second_linear,
        use_edge_weights=cfg.edge_weights,
    )


def _build_poly_nsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.nsd import PolyNSDModel, PolyNSDVariant

    return PolyNSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(PolyNSDVariant, cfg.variant),
        alpha=cfg.alpha,
        degree=cfg.degree,
        lambda_max=cfg.lambda_max,
        hp_alpha_init=cfg.hp_alpha_init,
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        add_lp=cfg.add_lp,
        add_hp=cfg.add_hp,
        sparse_learner=cfg.sparse_learner,
        second_linear=cfg.second_linear,
        use_edge_weights=cfg.edge_weights,
    )


def _build_conn_nsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.nsd import ConnectionLaplacianNSDModel

    # Maps are derived from raw features by local PCA + SVD, so cfg.variant and
    # every learned-map knob (orth_strategy, rank, add_lp/hp, ...) do not apply.
    return ConnectionLaplacianNSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        alpha=cfg.alpha,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
    )


def _build_poly_het_nsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.nsd import PolyHetNSDModel, PolyHetNSDVariant

    return PolyHetNSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(PolyHetNSDVariant, cfg.variant),
        alpha=cfg.alpha,
        degree=cfg.degree,
        lambda_max=cfg.lambda_max,
        hp_alpha_init=cfg.hp_alpha_init,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        ensemble=cfg.ensemble,
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


def _build_het_nsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.nsd import HetNSDModel, HetNSDVariant

    return HetNSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(HetNSDVariant, cfg.variant),
        alpha=cfg.alpha,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        ensemble=cfg.ensemble,
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


def _build_het_nlsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.nlsd import HetNLSDModel, HetNLSDVariant
    from sheaf_mpnn.nlsd import NLSDNonlinearity

    return HetNLSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(HetNLSDVariant, cfg.variant),
        nonlinearity=_parse_variant(NLSDNonlinearity, cfg.nlsd_nonlinearity),
        psi=cfg.nlsd_psi,
        num_mlp_layers=cfg.nlsd_mlp_layers,
        init_threshold=cfg.nlsd_init_threshold,
        alpha=cfg.alpha,
        learn_alpha=cfg.learn_alpha,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        ensemble=cfg.ensemble,
        orth_strategy=cfg.orth_strategy,
        clamp_val=cfg.clamp_val,
        rank=cfg.rank,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


def _build_het_nsp(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.nsp import HetNSPModel, HetNSPVariant

    return HetNSPModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(HetNSPVariant, cfg.variant),
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        ensemble=cfg.ensemble,
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


def _build_het_sheaf_attn(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.sheaf_attn import HetSheafAttnModel, HetSheafAttnVariant

    return HetSheafAttnModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(HetSheafAttnVariant, cfg.variant),
        num_heads=cfg.num_heads,
        leaky_relu_slope=cfg.leaky_relu_slope,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        ensemble=cfg.ensemble,
        clamp_val=cfg.clamp_val,
        orth_strategy=cfg.orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


def _build_id_sheaf(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.id_sheaf.id_sheaf_model import IdentitySheafModel

    return IdentitySheafModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        alpha=cfg.alpha,
        add_self_loops=False,
        add_eps=True,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        add_lp=cfg.add_lp,
        add_hp=cfg.add_hp,
        second_linear=cfg.second_linear,
        learn_alpha=cfg.learn_alpha,
    )


def _build_joint_diff(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.jointdiff.joint_model import JointDiffusionModel

    return JointDiffusionModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        beta=cfg.beta,
        alpha=cfg.alpha,
        add_self_loops=False,
        add_eps=True,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        add_lp=cfg.add_lp,
        add_hp=cfg.add_hp,
        second_linear=cfg.second_linear,
        learn_alpha=cfg.learn_alpha,
    )


def _build_rot_inv(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.jointdiff.joint_model import RotationInvariantModel

    return RotationInvariantModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        alpha=cfg.alpha,
        add_self_loops=False,
        add_eps=True,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        add_lp=cfg.add_lp,
        add_hp=cfg.add_hp,
        second_linear=cfg.second_linear,
        time_dep=cfg.time_dep,
        learn_alpha=cfg.learn_alpha,
    )


def _build_csnn(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from sheaf_mpnn.csnn import CSNNModel, CSNNVariant

    if cfg.jknet:
        raise ValueError("CSNN does not implement jknet")

    return CSNNModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(CSNNVariant, cfg.variant),
        orth_strategy=cfg.orth_strategy,
        clamp_val=cfg.clamp_val,
        left_weights=cfg.left_weights,
        right_weights=cfg.right_weights,
        use_act=cfg.use_act,
        layer_norm=cfg.layer_norm,
        batch_norm=cfg.batch_norm,
        csnn_head_layers=cfg.csnn_head_layers,
        gnn_layers=cfg.gnn_layers,
        gnn_hidden=cfg.gnn_hidden,
        gnn_default=cfg.gnn_default,
        gnn_type=cfg.gnn_type,
        gnn_residual=cfg.gnn_residual,
        linear_emb=cfg.linear_emb,
        pe_size=cfg.pe_size,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
    )


def _build_het_id_sheaf(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.id_sheaf import HetIdentitySheafModel

    return HetIdentitySheafModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        alpha=cfg.alpha,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


def _build_het_joint_diff(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.jointdiff import HetJointDiffusionModel

    return HetJointDiffusionModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        beta=cfg.beta,
        alpha=cfg.alpha,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        ensemble=cfg.ensemble,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


def _build_het_rot_inv(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.jointdiff import HetRotationInvariantModel

    return HetRotationInvariantModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        alpha=cfg.alpha,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        ensemble=cfg.ensemble,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        time_dep=cfg.time_dep,
    )


def _build_het_directed_nsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.directed_nsd import HetDirectedNSDModel, HetDirectedNSDVariant

    # The directed layers only implement the cayley / fasth parametrisations.
    if cfg.orth_strategy not in ("cayley", "fasth"):
        raise ValueError(
            "het_directed_nsd supports orth_strategy 'cayley' or 'fasth', "
            f"got {cfg.orth_strategy!r}"
        )
    orth_strategy: Literal["cayley", "fasth"] = cfg.orth_strategy

    return HetDirectedNSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(HetDirectedNSDVariant, cfg.variant),
        alpha=cfg.alpha,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        ensemble=cfg.ensemble,
        rank=cfg.rank,
        orth_strategy=orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        learn_alpha=cfg.learn_alpha,
    )


def _build_het_csnn(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.csnn import CSNNVariant, HetCSNNModel

    if cfg.jknet:
        raise ValueError("HetCSNN does not implement jknet")
    if cfg.ensemble:
        raise ValueError(
            "HetCSNN has no per-edge-type ensemble mode: its sheaf learners "
            "are per-node, so maps are not generated per edge"
        )

    return HetCSNNModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(CSNNVariant, cfg.variant),
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        orth_strategy=cfg.orth_strategy,
        clamp_val=cfg.clamp_val,
        left_weights=cfg.left_weights,
        right_weights=cfg.right_weights,
        use_act=cfg.use_act,
        layer_norm=cfg.layer_norm,
        batch_norm=cfg.batch_norm,
        csnn_head_layers=cfg.csnn_head_layers,
        gnn_layers=cfg.gnn_layers,
        gnn_hidden=cfg.gnn_hidden,
        gnn_default=cfg.gnn_default,
        gnn_type=cfg.gnn_type,
        gnn_residual=cfg.gnn_residual,
        linear_emb=cfg.linear_emb,
        pe_size=cfg.pe_size,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
    )


def _build_het_bsnn(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.bsnn import BSNNVariant, HetBSNNModel

    if cfg.ensemble:
        raise ValueError(
            "HetBSNN has no per-edge-type ensemble mode: the variational "
            "heads are single linears; use use_edge_types instead"
        )

    return HetBSNNModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(BSNNVariant, cfg.variant),
        alpha=cfg.alpha,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        orth_strategy=cfg.orth_strategy,
        clamp_val=cfg.clamp_val,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        learn_alpha=cfg.learn_alpha,
        ensemble_size=cfg.ensemble_size,
    )


def _build_het_sheaf_adj(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.sheaf_adj import HetSheafAdjModel, HetSheafAdjVariant

    return HetSheafAdjModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(HetSheafAdjVariant, cfg.variant),
        alpha=cfg.alpha,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        ensemble=cfg.ensemble,
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


def _build_het_dnsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.dnsd import HetDNSDModel, HetDNSDVariant

    return HetDNSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        variant=_parse_variant(HetDNSDVariant, cfg.variant),
        alpha=cfg.alpha,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        ensemble=cfg.ensemble,
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
        odd=cfg.odd,
        gate=cfg.gate,
        layer_norm=cfg.layer_norm,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
        learn_alpha=cfg.learn_alpha,
    )


def _build_het_conn_nsd(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.nsd import HetConnectionLaplacianNSDModel

    # As for conn_nsd, the maps come from the raw features, so cfg.variant, the
    # learned-map knobs and every het-context flag (use_*_types, ensemble) do
    # not apply: node/edge types are accepted by forward() and ignored.
    return HetConnectionLaplacianNSDModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        alpha=cfg.alpha,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
    )


def _build_het_bunn(
    in_channels: int, out_channels: int, cfg: ModelConfig, reg: RegConfig
) -> nn.Module:
    from hetsheaf_mpnn.bunn import HetBuNNModel

    if cfg.ensemble:
        raise ValueError(
            "HetBuNN has no per-edge-type ensemble mode: its bundle maps are "
            "per node, so maps are not generated per edge"
        )

    # As for bunn, cfg.variant does not apply and stalk_dim is the bundle
    # dimension d; the het flags reach the bundle-map learner phi.
    return HetBuNNModel(
        in_channels=in_channels,
        out_channels=out_channels,
        bundle_dim=cfg.stalk_dim,
        hidden_dim=cfg.hidden_dim,
        num_bundles=cfg.bunn_num_bundles,
        num_layers=cfg.num_layers,
        tau=cfg.bunn_tau,
        learn_tau=cfg.bunn_learn_tau,
        solver=cfg.bunn_solver,
        max_degree=cfg.bunn_max_degree or None,
        num_eigs=cfg.bunn_num_eigs or None,
        normalize=cfg.bunn_normalize,
        phi=cfg.bunn_phi,
        num_phi_layers=cfg.bunn_phi_layers,
        shared_phi=cfg.bunn_shared_phi,
        pe_size=cfg.pe_size,
        num_node_types=cfg.num_node_types,
        num_edge_types=cfg.num_edge_types,
        use_features=cfg.use_features,
        use_node_types=cfg.use_node_types,
        use_edge_types=cfg.use_edge_types,
        orth_strategy=cfg.orth_strategy,
        clamp_val=cfg.clamp_val,
        input_dropout=reg.input_dropout,
        dropout=reg.dropout,
        normalize_output=cfg.normalize_output,
        jknet=cfg.jknet,
    )


# ---------------------------------------------------------------------------
# Registry instance
# ---------------------------------------------------------------------------

model_registry: ModelRegistry = ModelRegistry()
model_registry.register("nsd", ModelEntry(factory=_build_nsd))
model_registry.register("bsnn", ModelEntry(factory=_build_bsnn))
model_registry.register("nlsd", ModelEntry(factory=_build_nlsd))
model_registry.register("bunn", ModelEntry(factory=_build_bunn))
model_registry.register("directed_nsd", ModelEntry(factory=_build_directed_nsd))
model_registry.register("nsp", ModelEntry(factory=_build_nsp))
model_registry.register("sheaf_attn", ModelEntry(factory=_build_sheaf_attn))
model_registry.register("sheaf_adj", ModelEntry(factory=_build_sheaf_adj))
model_registry.register("dnsd", ModelEntry(factory=_build_dnsd))
model_registry.register("poly_nsd", ModelEntry(factory=_build_poly_nsd))
model_registry.register("conn_nsd", ModelEntry(factory=_build_conn_nsd))
model_registry.register("poly_het_nsd", ModelEntry(factory=_build_poly_het_nsd))
model_registry.register("het_nsd", ModelEntry(factory=_build_het_nsd))
model_registry.register("het_nlsd", ModelEntry(factory=_build_het_nlsd))
model_registry.register("het_nsp", ModelEntry(factory=_build_het_nsp))
model_registry.register("het_sheaf_attn", ModelEntry(factory=_build_het_sheaf_attn))
model_registry.register("id_sheaf", ModelEntry(factory=_build_id_sheaf))
model_registry.register("joint_diff", ModelEntry(factory=_build_joint_diff))
model_registry.register("rot_inv", ModelEntry(factory=_build_rot_inv))
model_registry.register("csnn", ModelEntry(factory=_build_csnn))
model_registry.register("het_id_sheaf", ModelEntry(factory=_build_het_id_sheaf))
model_registry.register("het_joint_diff", ModelEntry(factory=_build_het_joint_diff))
model_registry.register("het_rot_inv", ModelEntry(factory=_build_het_rot_inv))
model_registry.register("het_directed_nsd", ModelEntry(factory=_build_het_directed_nsd))
model_registry.register("het_csnn", ModelEntry(factory=_build_het_csnn))
model_registry.register("het_bsnn", ModelEntry(factory=_build_het_bsnn))
model_registry.register("het_sheaf_adj", ModelEntry(factory=_build_het_sheaf_adj))
model_registry.register("het_dnsd", ModelEntry(factory=_build_het_dnsd))
model_registry.register("het_conn_nsd", ModelEntry(factory=_build_het_conn_nsd))
model_registry.register("het_bunn", ModelEntry(factory=_build_het_bunn))
