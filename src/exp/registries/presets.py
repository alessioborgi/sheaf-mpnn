# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Preset registry: per-dataset best-known configurations.

Presets are generated programmatically rather than written out one by one.
Every preset for a given dataset shares the same per-dataset hyperparameters
(stalk_dim, hidden_dim, num_layers, dropouts, lr, weight_decay, stop_strategy);
only the model family and restriction-map variant change. So the data lives in
two small tables -- _NODE_HPARAMS (one row per dataset) and _FAMILIES (one row
per model family) -- and the cross product is materialised by the loop at the
bottom. Preset keys follow ``<dataset>[_<family>_<variant>]``; the bare
``<dataset>`` key is NSD with the dataset's default variant
(``_HP.default_variant``: ``general`` for most, ``orthogonal`` for
chameleon/squirrel, ``diagonal`` for film). ConnNSD, IdSheaf, JointDiff, and
RotInv have no variant grid, so their keys are just ``<dataset>_<family>``
(e.g. ``<dataset>_conn_nsd``, ``<dataset>_joint_diff``).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from exp.config import (
    Config,
    CVConfig,
    DatasetConfig,
    HardwareConfig,
    ModelConfig,
    ModelType,
    OptimConfig,
    RegConfig,
    WandBConfig,
)
from exp.registries.base import Registry


class PresetRegistry(Registry[str, Config]):
    """Registry for named experiment presets."""

    def get_or_default(self, name: str | None) -> Config:
        """Return the named preset, or a bare ``Config()`` if name is None."""
        if name is None:
            return Config()
        return self.get(name)


_VariantLiteral = Literal[
    "diagonal",
    "general",
    "orthogonal",
    "low_rank",
    "general_attention",
    "orthogonal_attention",
    "sheafan",
    "res_sheafan",
]
_StopStrategyLiteral = Literal["loss", "acc"]
_OrthLiteral = Literal["cayley", "fasth", "fasthpp", "householder"]


def generate_config(
    name: str,
    *,
    variant: _VariantLiteral,
    stalk_dim: int,
    hidden_dim: int,
    num_layers: int,
    input_dropout: float,
    dropout: float,
    lr: float,
    weight_decay: float,
    model_type: ModelType = ModelType.NSD,
    alpha: float = 1.0,
    num_heads: int = 1,
    leaky_relu_slope: float = 0.2,
    clamp_val: float = 10.0,
    stop_strategy: _StopStrategyLiteral = "loss",
    normalize_output: bool = False,
    jknet: bool = False,
    orth_strategy: _OrthLiteral = "cayley",
    add_lp: bool = False,
    add_hp: bool = False,
    sparse_learner: bool = False,
    second_linear: bool = False,
    edge_weights: bool = False,
    learn_alpha: bool = True,
    bunn_num_bundles: int = 1,
    odd: bool = False,
    gate: bool = False,
    layer_norm: bool = False,
    epochs: int = 1000,
    early_stopping: int = 200,
    sheaf_decay: float | None = None,
) -> Config:
    return Config(
        dataset=DatasetConfig(name=name),
        model=ModelConfig(
            type=model_type,
            variant=variant,
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            alpha=alpha,
            num_heads=num_heads,
            leaky_relu_slope=leaky_relu_slope,
            clamp_val=clamp_val,
            normalize_output=normalize_output,
            jknet=jknet,
            orth_strategy=orth_strategy,
            add_lp=add_lp,
            add_hp=add_hp,
            sparse_learner=sparse_learner,
            second_linear=second_linear,
            edge_weights=edge_weights,
            learn_alpha=learn_alpha,
            bunn_num_bundles=bunn_num_bundles,
            odd=odd,
            gate=gate,
            layer_norm=layer_norm,
        ),
        reg=RegConfig(input_dropout=input_dropout, dropout=dropout),
        optim=OptimConfig(
            lr=lr,
            weight_decay=weight_decay,
            stop_strategy=stop_strategy,
            epochs=epochs,
            early_stopping=early_stopping,
            sheaf_decay=sheaf_decay,
        ),
        cv=CVConfig(),
        hardware=HardwareConfig(),
        wandb=WandBConfig(),
    )


class _HP(NamedTuple):
    """Per-dataset hyperparameters shared across every family and variant."""

    stalk_dim: int
    hidden_dim: int
    num_layers: int
    input_dropout: float
    dropout: float
    lr: float
    weight_decay: float
    stop_strategy: _StopStrategyLiteral = "loss"
    # Variant used by the bare ``<dataset>`` alias (the per-dataset NSD default).
    default_variant: _VariantLiteral = "general"
    # Training budget; Pei et al. rows carry the Bodnar et al. script values.
    epochs: int = 1000
    early_stopping: int = 200
    sheaf_decay: float | None = None


def _config(
    name: str,
    model_type: ModelType,
    variant: _VariantLiteral,
    hp: _HP,
    *,
    num_heads: int = 1,
    jknet: bool = False,
    normalize_output: bool = False,
) -> Config:
    """Expand one (dataset, model_type, variant) triple into a Config."""
    return generate_config(
        name,
        model_type=model_type,
        variant=variant,
        stalk_dim=hp.stalk_dim,
        hidden_dim=hp.hidden_dim,
        num_layers=hp.num_layers,
        input_dropout=hp.input_dropout,
        dropout=hp.dropout,
        lr=hp.lr,
        weight_decay=hp.weight_decay,
        stop_strategy=hp.stop_strategy,
        num_heads=num_heads,
        jknet=jknet,
        normalize_output=normalize_output,
        # DNSD's paper always trains with LayerNorm on (as does its het mirror);
        # other families leave this at the ModelConfig default (False).
        layer_norm=model_type in (ModelType.DNSD, ModelType.HET_DNSD),
        epochs=hp.epochs,
        early_stopping=hp.early_stopping,
        sheaf_decay=hp.sheaf_decay,
    )


# Homophilic + heterophilic node-classification datasets that receive the full
# family x variant grid of presets.
_NODE_HPARAMS: dict[str, _HP] = {
    "cora": _HP(4, 32, 2, 0.5, 0.0, 0.01, 5e-4, epochs=1500),
    "citeseer": _HP(4, 32, 2, 0.5, 0.0, 0.01, 5e-4, epochs=1500),
    "pubmed": _HP(4, 32, 2, 0.5, 0.0, 0.01, 5e-4, epochs=1500),
    "chameleon": _HP(
        4,
        32,
        5,
        0.7,
        0.0,
        0.01,
        0.0002969905682317406,
        "acc",
        default_variant="diagonal",
        epochs=1000,
        early_stopping=100,
        sheaf_decay=0.0012638885974822734,
    ),
    "squirrel": _HP(
        3,
        32,
        5,
        0.7,
        0.0,
        0.01,
        0.00011215791366362148,
        "acc",
        default_variant="orthogonal",
        epochs=1000,
        early_stopping=100,
    ),
    "chameleon_filtered": _HP(
        4, 32, 3, 0.0, 0.0, 0.01, 1e-3, default_variant="orthogonal"
    ),
    "squirrel_filtered": _HP(
        4, 32, 3, 0.0, 0.0, 0.01, 1e-3, default_variant="orthogonal"
    ),
    "cornell": _HP(
        4,
        16,
        2,
        0.2,
        0.7,
        0.02,
        0.0006914841722570725,
        default_variant="diagonal",
        epochs=500,
        sheaf_decay=0.00031764232712732976,
    ),
    "texas": _HP(
        3,
        20,
        4,
        0.0,
        0.7,
        0.02,
        5e-3,
        default_variant="orthogonal",
        epochs=1500,
    ),
    "wisconsin": _HP(
        3,
        32,
        2,
        0.0,
        0.7276458263736642,
        0.02,
        0.0006685729356079199,
        default_variant="orthogonal",
        epochs=500,
    ),
    "film": _HP(
        4, 32, 3, 0.5, 0.5, 0.005, 5e-4, default_variant="diagonal", epochs=1500
    ),
    "amazon_ratings": _HP(4, 32, 4, 0.2, 0.2, 0.01, 5e-4),
    "minesweeper": _HP(3, 16, 4, 0.0, 0.0, 0.01, 5e-4),
    "questions": _HP(4, 32, 3, 0.2, 0.2, 0.005, 5e-4),
    "roman_empire": _HP(4, 32, 4, 0.5, 0.5, 0.01, 5e-4),
    "tolokers": _HP(3, 16, 3, 0.2, 0.0, 0.01, 5e-4),
    # OGB (Hu et al. 2020): one official split (num_splits=1), so every fold
    # trains the same split and cross-fold variance comes from seeds alone.
    # Row follows the common full-batch ogbn-arxiv recipe (width 256 = d*f,
    # 3 layers, dropout 0.5, no weight decay), stopped on accuracy.
    "ogbn_arxiv": _HP(4, 64, 3, 0.0, 0.5, 0.01, 0.0, "acc", epochs=500),
    # ogbn-products (2.45M nodes / 124M edge slots): full-batch training only
    # fits at small widths; keep d*f low and the budget short.
    "ogbn_products": _HP(
        2, 32, 3, 0.0, 0.5, 0.003, 0.0, "acc", epochs=300, early_stopping=50
    ),
    # ogbn-proteins: 112 binary targets, mean ROC-AUC. stop_strategy "acc"
    # monitors val_<metric>, i.e. val_roc_auc here.
    "ogbn_proteins": _HP(
        2, 32, 4, 0.0, 0.0, 0.01, 0.0, "acc", epochs=500, early_stopping=100
    ),
    # pokec (LINKX, Lim et al. 2021): 1.63M nodes / 30.6M directed edges,
    # binary gender labels over the five official 50/25/25 splits.
    "pokec": _HP(2, 32, 3, 0.0, 0.5, 0.01, 0.0, "acc", epochs=300, early_stopping=50),
}

# Restriction-map variants per family.
_NSD_NSP_VARIANTS: list[_VariantLiteral] = [
    "diagonal",
    "general",
    "orthogonal",
    "general_attention",
    "orthogonal_attention",
    "low_rank",
]
# BSNN (Gillespie et al., 2025) restriction-map families: Gaussian posteriors
# for diagonal/general maps (and, as a framework extension, over the factors
# of low_rank maps), Cayley distributions on SO(d) for orthogonal.
_BSNN_VARIANTS: list[_VariantLiteral] = [
    "diagonal",
    "general",
    "orthogonal",
    "low_rank",
]
# The NLSD paper defines diagonal and orthogonal; general and low_rank are
# framework extensions. No attention variants: NLSD has no attention learner.
_NLSD_VARIANTS: list[_VariantLiteral] = [
    "diagonal",
    "orthogonal",
    "general",
    "low_rank",
]
# Each family contributes presets per dataset and variant. Only attention
# families consume num_heads; other families retain its default value.
_FAMILIES: list[tuple[ModelType, int, list[tuple[_VariantLiteral, str]]]] = [
    (ModelType.NSD, 1, [(v, f"nsd_{v}") for v in _NSD_NSP_VARIANTS]),
    (ModelType.BSNN, 1, [(v, f"bsnn_{v}") for v in _BSNN_VARIANTS]),
    (ModelType.NLSD, 1, [(v, f"nlsd_{v}") for v in _NLSD_VARIANTS]),
    # BuNN (Bamberger et al., 2025) has no restriction-map variant grid: a flat
    # bundle assigns one orthogonal map per node, so the recorded variant just
    # documents that those maps are orthogonal. Its node state is
    # bunn_num_bundles * stalk_dim * hidden_dim wide, so generate_config pins
    # bunn_num_bundles to 1 -- otherwise these rows would carry several times
    # the width of every other family at the same shared _HP row and the grid
    # would stop being a like-for-like comparison. The configs/<dataset>/bunn
    # sweeps explore wider bundle counts.
    (ModelType.BUNN, 1, [("orthogonal", "bunn")]),
    (ModelType.POLY_NSD, 1, [(v, f"poly_nsd_{v}") for v in _NSD_NSP_VARIANTS]),
    (ModelType.SHEAF_ADJ, 1, [(v, f"sheaf_adj_{v}") for v in _NSD_NSP_VARIANTS]),
    (ModelType.DNSD, 1, [(v, f"dnsd_{v}") for v in _NSD_NSP_VARIANTS]),
    (ModelType.SHEAF_ATTN, 4, [("sheafan", "sheafan"), ("res_sheafan", "res_sheafan")]),
    (ModelType.NSP, 1, [(v, f"nsp_{v}") for v in _NSD_NSP_VARIANTS]),
    # ConnNSD derives its maps from the features, so it has no variant grid;
    # "orthogonal" records that the derived transports are orthogonal.
    (ModelType.CONN_NSD, 1, [("orthogonal", "conn_nsd")]),
    # The Hernandez Caralt et al. (2024) family has no variant grid either;
    # the recorded variant documents the map structure (identity maps are
    # diagonal, joint-diffusion and rotation-invariant maps are general).
    (ModelType.ID_SHEAF, 1, [("diagonal", "id_sheaf")]),
    (ModelType.JOINT_DIFF, 1, [("general", "joint_diff")]),
    (ModelType.ROT_INV, 1, [("general", "rot_inv")]),
]

_PRESETS: dict[str, Config] = {}

# Node-classification datasets: bare alias (NSD, per-dataset default_variant)
# plus the full family/variant grid.
for _ds, _hp in _NODE_HPARAMS.items():
    _PRESETS[_ds] = _config(_ds, ModelType.NSD, _hp.default_variant, _hp)
    for _model_type, _num_heads, _variants in _FAMILIES:
        for _variant, _suffix in _variants:
            _PRESETS[f"{_ds}_{_suffix}"] = _config(
                _ds, _model_type, _variant, _hp, num_heads=_num_heads
            )

# HGB heterogeneous -> homogeneous node-classification datasets: NSD general only.
_HGB_HP = _HP(4, 32, 2, 0.5, 0.0, 0.01, 5e-4)
_HGB_DATASETS = ("imdb", "dblp", "acm", "freebase")
for _ds in _HGB_DATASETS:
    _PRESETS[_ds] = _config(_ds, ModelType.NSD, "general", _HGB_HP)

# Het NLSD on the same HGB datasets, one preset per restriction-map family.
# These are the first per-family het presets: the other het families are run by
# borrowing a homogeneous model preset and overriding --dataset.name (see
# exp/het_run.py). num_node_types / num_edge_types stay 0 on purpose --
# HetSheafLightningModule back-fills them from the dataset's HetDatasetInfo, so
# pinning them here would only risk contradicting the loaded graph.
_HET_NLSD_VARIANTS: list[_VariantLiteral] = [
    "diagonal",
    "orthogonal",
    "general",
    "low_rank",
]
for _ds in _HGB_DATASETS:
    for _variant in _HET_NLSD_VARIANTS:
        _het_cfg = _config(_ds, ModelType.HET_NLSD, _variant, _HGB_HP)
        _het_cfg.model.use_node_types = True
        _het_cfg.model.use_edge_types = True
        _PRESETS[f"{_ds}_het_nlsd_{_variant}"] = _het_cfg

# Het SheafAdj and its deep DNSD extension get the same per-variant HGB grid as
# the homogeneous families, with type-aware map generators switched on.
for _ds in _HGB_DATASETS:
    for _model_type, _prefix in (
        (ModelType.HET_SHEAF_ADJ, "het_sheaf_adj"),
        (ModelType.HET_DNSD, "het_dnsd"),
    ):
        for _variant in _NSD_NSP_VARIANTS:
            _het_cfg = _config(_ds, _model_type, _variant, _HGB_HP)
            _het_cfg.model.use_node_types = True
            _het_cfg.model.use_edge_types = True
            _PRESETS[f"{_ds}_{_prefix}_{_variant}"] = _het_cfg

# Het ConnNSD derives its maps from the raw features, so it has no variant grid
# and no het-context flags. Freebase is skipped: its nodes carry no features,
# leaving nothing for the local PCA to span.
for _ds in _HGB_DATASETS:
    if _ds != "freebase":
        _PRESETS[f"{_ds}_het_conn_nsd"] = _config(
            _ds, ModelType.HET_CONN_NSD, "orthogonal", _HGB_HP
        )

# Het BuNN has no variant grid either ("orthogonal" records that bundle maps are
# orthogonal) and, like bunn, pins bunn_num_bundles to 1. Its edge types enter
# phi as relation-specific SumGNN weights, so the presets switch phi to
# sum_gnn; an MLP phi never reads the edges and would reject use_edge_types.
for _ds in _HGB_DATASETS:
    _het_cfg = _config(_ds, ModelType.HET_BUNN, "orthogonal", _HGB_HP)
    _het_cfg.model.use_node_types = True
    _het_cfg.model.use_edge_types = True
    _het_cfg.model.bunn_phi = "sum_gnn"
    _PRESETS[f"{_ds}_het_bunn"] = _het_cfg

# Graph-classification datasets: NSD general only.
_GC_HP = _HP(4, 32, 3, 0.2, 0.2, 0.01, 5e-4)
for _ds in (
    "mutag",
    "proteins",
    "enzymes",
    "nci1",
    "nci109",
    "ptcmr",
    "dd",
    "collab",
    "imdb_b",
    "imdb_m",
    "reddit_b",
    "reddit_5k",
):
    _PRESETS[_ds] = _config(_ds, ModelType.NSD, "general", _GC_HP)

# Bodnar et al. defaults: the plain <dataset> presets for the five scripted
# Pei et al. datasets ARE his exact configurations (BundleSheaf -> orthogonal
# + householder + edge weights; DiagSheaf -> diagonal; learn_alpha=False is
# his exact update), overriding the grid-generated aliases above.
_PRESETS["texas"] = generate_config(
    "texas",
    variant="orthogonal",
    stalk_dim=3,
    hidden_dim=20,
    num_layers=4,
    input_dropout=0.0,
    dropout=0.7,
    lr=0.02,
    weight_decay=5e-3,
    orth_strategy="householder",
    edge_weights=True,
    sparse_learner=True,
    learn_alpha=False,
    epochs=1500,
)
_PRESETS["wisconsin"] = generate_config(
    "wisconsin",
    variant="orthogonal",
    stalk_dim=3,
    hidden_dim=32,
    num_layers=2,
    input_dropout=0.0,
    dropout=0.7276458263736642,
    lr=0.02,
    weight_decay=0.0006685729356079199,
    orth_strategy="householder",
    edge_weights=True,
    add_lp=True,
    add_hp=True,
    learn_alpha=False,
    epochs=500,
)
_PRESETS["cornell"] = generate_config(
    "cornell",
    variant="diagonal",
    stalk_dim=4,
    hidden_dim=16,
    num_layers=2,
    input_dropout=0.2,
    dropout=0.7,
    lr=0.02,
    weight_decay=0.0006914841722570725,
    sheaf_decay=0.00031764232712732976,
    add_lp=True,
    learn_alpha=False,
    epochs=500,
)
_PRESETS["chameleon"] = generate_config(
    "chameleon",
    variant="diagonal",
    stalk_dim=4,
    hidden_dim=32,
    num_layers=5,
    input_dropout=0.7,
    dropout=0.0,
    lr=0.01,
    weight_decay=0.0002969905682317406,
    sheaf_decay=0.0012638885974822734,
    add_lp=True,
    second_linear=True,
    learn_alpha=False,
    epochs=1000,
    early_stopping=100,
    stop_strategy="acc",
)
_PRESETS["squirrel"] = generate_config(
    "squirrel",
    variant="orthogonal",
    stalk_dim=3,
    hidden_dim=32,
    num_layers=5,
    input_dropout=0.7,
    dropout=0.0,
    lr=0.01,
    weight_decay=0.00011215791366362148,
    orth_strategy="householder",
    edge_weights=True,
    add_lp=True,
    second_linear=True,
    learn_alpha=False,
    epochs=1000,
    early_stopping=100,
    stop_strategy="acc",
)

# ---------------------------------------------------------------------------
# CSNN reproduction presets
# ---------------------------------------------------------------------------
# Transcribed 1:1 from the original repository's exp/scripts/run_*.sh. CSNN's
# hyperparameters are per-dataset and it needs a dozen fields _HP does not
# carry, so it gets its own table instead of joining the _FAMILIES grid.
#
# Shared by every row (reference parser defaults unless the script overrides):
#   folds=10, adam, batch_size=1, use_act=True, pe_size=0, gnn_type=SAGE,
#   linear_emb=False, gnn_residual=False, csnn_head_layers=1 (--num_heads=1),
#   orth_strategy=householder, variant=conformal (--conformal defaults True).
#
# stop_strategy is "acc" on every row: the reference parser defaults to acc,
# while OptimConfig here defaults to "loss". tolokers passes no flag at all and
# still means acc.


class _CSNNHP(NamedTuple):
    """Per-dataset CSNN hyperparameters from the reference run scripts."""

    stalk_dim: int
    hidden_dim: int
    num_layers: int
    gnn_layers: int
    gnn_hidden: int
    gnn_default: int
    input_dropout: float
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    early_stopping: int
    left_weights: bool = True
    right_weights: bool = True
    layer_norm: bool = False
    # "acc" is the reference parser default; the sweep-anchor rows below mirror
    # their dataset's _HP convention instead.
    stop_strategy: _StopStrategyLiteral = "acc"


def _csnn_config(name: str, hp: _CSNNHP) -> Config:
    """Expands one CSNN reproduction row into a Config."""
    return Config(
        dataset=DatasetConfig(name=name),
        model=ModelConfig(
            type=ModelType.CSNN,
            variant="conformal",
            stalk_dim=hp.stalk_dim,
            hidden_dim=hp.hidden_dim,
            num_layers=hp.num_layers,
            orth_strategy="householder",
            gnn_layers=hp.gnn_layers,
            gnn_hidden=hp.gnn_hidden,
            gnn_default=hp.gnn_default,
            left_weights=hp.left_weights,
            right_weights=hp.right_weights,
            use_act=True,
            layer_norm=hp.layer_norm,
            csnn_head_layers=1,
        ),
        reg=RegConfig(input_dropout=hp.input_dropout, dropout=hp.dropout),
        optim=OptimConfig(
            lr=hp.lr,
            weight_decay=hp.weight_decay,
            stop_strategy=hp.stop_strategy,
            epochs=hp.epochs,
            early_stopping=hp.early_stopping,
            # None so the map-learner group falls back to weight_decay, matching
            # the reference's single flat Adam group.
            sheaf_decay=None,
        ),
        cv=CVConfig(),
        hardware=HardwareConfig(),
        wandb=WandBConfig(),
    )


_CSNN_HPARAMS: dict[str, _CSNNHP] = {
    #                    d  hid  L  gL  gH  gD  in_do  do   lr     wd    ep    es
    "roman_empire": _CSNNHP(4, 32, 5, 3, 32, 2, 0.2, 0.2, 0.02, 1e-7, 2000, 200),
    "amazon_ratings": _CSNNHP(4, 64, 3, 2, 32, 1, 0.2, 0.2, 0.02, 1e-7, 2000, 200),
    "minesweeper": _CSNNHP(
        3,
        32,
        5,
        5,
        64,
        2,
        0.2,
        0.2,
        0.002,
        1e-7,
        2000,
        2000,
        left_weights=False,
        layer_norm=True,
    ),
    "questions": _CSNNHP(
        4, 32, 4, 1, 32, 1, 0.2, 0.2, 0.002, 1e-7, 2000, 200, right_weights=False
    ),
    # gnn_default=0 -> the custom SAGE stack, the only learner path that reads
    # edge_attr. early_stopping == epochs means no early stopping.
    #
    # gnn_default=0 is an interpretation, not a transcription: the released
    # run_squirrel.sh passes --gnn_default=True to an int-typed flag, and
    # int("True") raises, so that script cannot have run verbatim. 0 is pinned
    # by the script's explicit --gnn_type=SAGE/--linear_emb=False (meaningful
    # only on the custom path) and by parameter parity with the reference:
    # 270,665 params under gnn_default=0 vs 271,305 under gnn_default=1.
    "squirrel": _CSNNHP(3, 32, 5, 2, 32, 0, 0.5, 0.5, 0.02, 1e-7, 1000, 1000),
    # gnn_layers=0 -> MLP learner; gnn_hidden still sizes its hidden width.
    "tolokers": _CSNNHP(
        3, 32, 2, 0, 32, 1, 0.2, 0.2, 0.02, 1e-8, 2000, 200, left_weights=False
    ),
}

for _ds, _csnn_hp in _CSNN_HPARAMS.items():
    _PRESETS[f"{_ds}_csnn"] = _csnn_config(_ds, _csnn_hp)

# The reference `squirrel` is the *directed filtered* Wikipedia graph, which is
# closer to squirrel_filtered here than to the Geom-GCN `squirrel`. Register the
# filtered variant too so both can be benchmarked and compared.
_PRESETS["squirrel_filtered_csnn"] = _csnn_config(
    "squirrel_filtered", _CSNN_HPARAMS["squirrel"]
)

# CSNN sweep-anchor presets for the remaining node datasets. The reference
# paper tunes only the six datasets above; these rows exist so every
# configs/<dataset>/csnn sweep has a preset to anchor its identity and training
# budget. Shared fields mirror the dataset's _HP row; the map learner is the
# reference GraphSAGE preset at a middle-of-the-road size. They are NOT
# reference transcriptions -- treat sweep results, not these rows, as the tuned
# configuration.
for _ds in (
    "cora",
    "citeseer",
    "pubmed",
    "chameleon",
    "chameleon_filtered",
    "cornell",
    "texas",
    "wisconsin",
    "film",
):
    _node_hp = _NODE_HPARAMS[_ds]
    _PRESETS[f"{_ds}_csnn"] = _csnn_config(
        _ds,
        _CSNNHP(
            stalk_dim=_node_hp.stalk_dim,
            hidden_dim=_node_hp.hidden_dim,
            num_layers=_node_hp.num_layers,
            gnn_layers=2,
            gnn_hidden=32,
            gnn_default=1,
            input_dropout=_node_hp.input_dropout,
            dropout=_node_hp.dropout,
            lr=_node_hp.lr,
            weight_decay=_node_hp.weight_decay,
            epochs=_node_hp.epochs,
            early_stopping=_node_hp.early_stopping,
            stop_strategy=_node_hp.stop_strategy,
        ),
    )

# ---------------------------------------------------------------------------
# Directed NSD presets
# ---------------------------------------------------------------------------
# One preset per (directed dataset, variant), keyed
# ``<dataset>_directed_nsd_<variant>``, mirroring the _FAMILIES grid. Kept out
# of _FAMILIES because these presets must also flip ``dataset.directed`` --
# training on the symmetrised graph would make the directed model pointless.
#
# The curated directed sources (exp/datasets/directed.py) are the 7 Platonov
# directed .npz graphs, the 3 directed citation graphs, and telegram. Shared
# fields reuse each dataset's _HP row; orth_strategy stays the generate_config
# default "cayley", one of the two parameterizations the directed layers
# implement.

# telegram has no _HP row (it is not part of the undirected benchmark grid);
# this is a sweep anchor, not a tuned configuration.
_TELEGRAM_HP = _HP(4, 32, 2, 0.2, 0.2, 0.01, 5e-4, "acc")

_DIRECTED_NSD_DATASETS = (
    "roman_empire",
    "amazon_ratings",
    "minesweeper",
    "questions",
    "tolokers",
    "chameleon_filtered",
    "squirrel_filtered",
    "cora",
    "citeseer",
    "pubmed",
    "telegram",
)

for _ds in _DIRECTED_NSD_DATASETS:
    _dir_hp = _NODE_HPARAMS.get(_ds, _TELEGRAM_HP)
    for _variant in _NSD_NSP_VARIANTS:
        _dir_cfg = _config(_ds, ModelType.DIRECTED_NSD, _variant, _dir_hp)
        _dir_cfg.dataset.directed = True
        _PRESETS[f"{_ds}_directed_nsd_{_variant}"] = _dir_cfg

# Shared preset for the 24 DSBM synthetic directed graphs
# (``dataset_nodes500_alpha*_beta*``); their sweep YAMLs pin the concrete
# dataset via the ``dataset:`` section, so one anchor preset serves the family.
_SYNTH_DIR_CFG = _config(
    "dataset_nodes500_alpha0.05_beta0.05",
    ModelType.DIRECTED_NSD,
    "general",
    _TELEGRAM_HP,
)
_SYNTH_DIR_CFG.dataset.directed = True
_PRESETS["synthetic_directed_nsd"] = _SYNTH_DIR_CFG

# Link-prediction datasets: NSD general, JKNet on, output left unnormalised.
_LP_HP = _HP(2, 32, 2, 0.0, 0.0, 0.001, 5e-4)
for _ds in ("lastfm", "movielens"):
    _PRESETS[_ds] = _config(
        _ds, ModelType.NSD, "general", _LP_HP, jknet=True, normalize_output=False
    )


# ---------------------------------------------------------------------------
# Registry instance  populated from _PRESETS at import time
# ---------------------------------------------------------------------------

preset_registry: PresetRegistry = PresetRegistry()
for _name, _cfg in _PRESETS.items():
    preset_registry.register(_name, _cfg)
