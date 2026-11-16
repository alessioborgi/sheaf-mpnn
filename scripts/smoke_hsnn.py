# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Lio

"""Train HSNN on a tiny synthetic hypergraph; no downloads or experiment extras.

Run from an installed checkout: ``python scripts/smoke_hsnn.py --variant all``.
This is an optimization smoke test, not a benchmark or generalization result.
"""

import argparse
import json
import math

import torch
import torch.nn.functional as F

from sheaf_mpnn import HSNNModel, HSNNVariant


def make_hypergraph(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create overlapping size-2/3/4 edges, mixed edges and one isolated node."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(41) // 20
    labels[-1] = 0
    x = F.one_hot(labels, 2).float()
    x = x + 0.1 * torch.randn(41, 2, generator=generator)
    edges = []
    for start in (0, 20):
        for offset in range(20):
            size = 2 + offset % 3
            edges.append([start + (offset + j) % 20 for j in range(size)])
    edges.extend([[0, 20, 21], [5, 6, 25], [10]])
    incidence = [(node, edge) for edge, nodes in enumerate(edges) for node in nodes]
    return x, labels, torch.tensor(incidence, dtype=torch.long).T.contiguous()


def run_smoke(
    variant: HSNNVariant, steps: int, seed: int, device: str
) -> dict[str, str | int | float | bool]:
    """Verify loss reduction and finite, nonzero gradients for every parameter."""
    torch.manual_seed(seed)
    x, labels, index = (value.to(device) for value in make_hypergraph(seed))
    model = HSNNModel(
        2, 2, stalk_dim=3, hidden_dim=8, num_layers=2, variant=variant
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    initial = F.cross_entropy(model(x, index), labels).item()
    finite_gradients = True
    nonzero_gradients = {name: False for name, _ in model.named_parameters()}
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, index), labels)
        loss.backward()
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                finite_gradients = False
            elif bool(gradient.abs().sum() > 0):
                nonzero_gradients[name] = True
        if not finite_gradients:
            raise RuntimeError(f"Non-finite or missing gradients for {variant.name}")
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = F.cross_entropy(model(x, index), labels).item()
    updated = all(
        not torch.equal(parameter.detach(), initial_parameters[name])
        for name, parameter in model.named_parameters()
    )
    if not (
        math.isfinite(final)
        and final < initial
        and updated
        and all(nonzero_gradients.values())
    ):
        raise RuntimeError(f"Optimization smoke failed for {variant.name}")
    return {
        "variant": variant.name.lower(),
        "seed": seed,
        "steps": steps,
        "device": device,
        "initial_loss": initial,
        "final_loss": final,
        "finite_gradients": finite_gradients,
        "all_parameters_received_nonzero_gradients": all(nonzero_gradients.values()),
        "all_parameters_updated": updated,
    }


def main() -> None:
    """Parse smoke-test options and print one JSON result per variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["all", *(v.name.lower() for v in HSNNVariant)],
        default="all",
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    variants = (
        list(HSNNVariant)
        if args.variant == "all"
        else [HSNNVariant[args.variant.upper()]]
    )
    for variant in variants:
        print(json.dumps(run_smoke(variant, args.steps, args.seed, args.device)))


if __name__ == "__main__":
    main()
