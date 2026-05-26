# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Model registry: factory-based construction of sheaf models from config."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
# Factory functions
# ---------------------------------------------------------------------------


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
        variant=NSDVariant[cfg.variant.upper()],
        alpha=cfg.alpha,
        rank=cfg.rank,
        orth_strategy=cfg.orth_strategy,
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
