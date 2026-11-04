# Neural Sheaf Propagation (NSP) Layers

This package implements **Neural Sheaf Propagation**, a second-order sheaf-based
message-passing model. It extends Neural Sheaf Diffusion (NSD) by replacing the
first-order residual update with a **wave-like recurrence** that carries momentum
across layers, drawing inspiration from Chebyshev / momentum methods in classical
graph signal processing.

## Core Concept

Like NSD, NSP assigns a stalk $\mathbf{X}_v \in \mathbb{R}^{d \times f}$ to each node
and learns edge restriction maps that build a dynamic Sheaf Laplacian. The key
difference is the **second-order recurrence** that governs propagation:

### The Propagation Equation

$$\mathbf{X}^{(t+1)} = 2\,\mathbf{X}^{(t)} - \mathbf{X}^{(t-1)} - \sigma\!\left(\Delta_{\mathcal{F}^{(t)}} \left(\mathbf{I}_{nd} \otimes \mathbf{W}_1^{(t)}\right) \mathbf{X}^{(t)} \mathbf{W}_2^{(t)}\right)$$

with the boundary condition $\mathbf{X}^{(-1)} = \mathbf{0}$.

The $2\mathbf{X}^{(t)} - \mathbf{X}^{(t-1)}$ term is the **inertial extrapolation** step:
instead of stepping from the current position alone (as in NSD), NSP first extrapolates
forward using the previous velocity $\mathbf{X}^{(t)} - \mathbf{X}^{(t-1)}$, then
corrects by the Laplacian force. This is analogous to Nesterov momentum.

## Reduction to First-Order via State Augmentation

The recurrence depends on both $\mathbf{X}^{(t)}$ and $\mathbf{X}^{(t-1)}$, so it is not
directly a SheafMPNN. We reduce it to first-order form via **state augmentation**:

$$\tilde{\mathbf{x}}_u^{(t)} := \begin{pmatrix} \mathbf{x}_u^{(t)} \\ \mathbf{x}_u^{(t-1)} \end{pmatrix} \in \mathbb{R}^{2d \times f}$$

with $\tilde{\mathbf{x}}_u^{(0)} = \bigl(\mathbf{x}_u^{(0)},\, \mathbf{0}\bigr)$.

The augmented sheaf is constructed as follows:

| Component | Original | Augmented |
|:----------|:---------|:----------|
| Node stalk | $\mathcal{F}(u) = \mathbb{R}^d$ | $\tilde{\mathcal{F}}(u) = \mathbb{R}^{2d}$ |
| Edge stalk | $\mathcal{F}(e) = \mathbb{R}^d$ | unchanged |
| Restriction map | $\mathcal{F}_{u \trianglelefteq e} : \mathbb{R}^d \to \mathbb{R}^d$ | $\tilde{\mathcal{F}}_{u \trianglelefteq e} = \mathcal{F}_{u \trianglelefteq e} \circ \pi_1 : \mathbb{R}^{2d} \to \mathbb{R}^d$ |
| Pre-transport weight | $\mathbf{W}_1 \in \mathbb{R}^{d \times d}$ | $\tilde{\mathbf{W}}_1 = \begin{pmatrix} \mathbf{W}_1 & \mathbf{0} \\ \mathbf{0} & \mathbf{0} \end{pmatrix} \in \mathbb{R}^{2d \times 2d}$ |

The first-order update function is:

$$\varphi_u\!\left(\tilde{\mathbf{x}}_u^{(k)},\, \mathbf{m}_u^{(k+1)}\right) = \begin{pmatrix} 2\,\pi_1\!\left(\tilde{\mathbf{x}}_u^{(k)}\right) - \pi_2\!\left(\tilde{\mathbf{x}}_u^{(k)}\right) - \mathbf{m}_u^{(k+1)} \\ \pi_1\!\left(\tilde{\mathbf{x}}_u^{(k)}\right) \end{pmatrix}$$

### Implementation simplification

Because $\tilde{\mathbf{W}}_1 = \mathrm{diag}(\mathbf{W}_1, \mathbf{0})$ zeros the
history block before transport, we never materialise the $2d$-wide augmented maps.
Instead:

1. `x_stalk` of shape `[N, 2d, f]` stores the augmented state contiguously.
2. `x_curr = x_stalk[:, :d, :]` and `x_hist = x_stalk[:, d:, :]` are extracted by slicing.
3. The bilateral transform and Laplacian operate entirely on `x_curr` (d-dimensional).
4. The update assembles `cat([new_curr, x_curr], dim=1)` — shifting `x_curr` into the history slot.

## Model Variants

NSP provides five variants via the `NSPVariant` Enum:

| Variant                | Restriction Map Logic                                        |
|:-----------------------|:-------------------------------------------------------------|
| `DIAGONAL`             | Maps are diagonal matrices (vectors). Lightweight.           |
| `GENERAL`              | Maps are full $d \times d$ matrices. Maximum expressivity.   |
| `ORTHOGONAL`           | Maps via Cayley Transform ($W^T W = I$). Energy preservation.|
| `GENERAL_ATTENTION`    | `GENERAL` + attention: $F = I - \text{softmax}(\text{MLP})$.|
| `ORTHOGONAL_ATTENTION` | `ORTHOGONAL` + attention before Cayley.                      |

## Differences from NSD

| Property | NSD | NSP |
|:---------|:----|:----|
| Update order | First-order residual | Second-order wave recurrence |
| Node stalk storage | `[N, d, f]` | `[N, 2d, f]` (curr ++ hist) |
| Learnable step size | `alpha` (scalar) | None (coefficients fixed at 2/1) |
| Boundary condition | — | $\mathbf{X}^{(-1)} = \mathbf{0}$ (zero history at layer 0) |
| Variants | 3 (DIAGONAL, GENERAL, ORTHOGONAL) | 5 (adds two attention variants) |

## Quick Start

```python
from sheaf_mpnn.nsp import NSPModel, NSPVariant

model = NSPModel(
    in_channels=1433,  # raw input node features (e.g., Cora)
    out_channels=7,    # target classes
    stalk_dim=4,       # stalk dimension d
    hidden_dim=32,     # features per stalk channel f
    num_layers=2,
    variant=NSPVariant.GENERAL,
)
# x: [N, in_channels], edge_index: [2, E]
logits = model(x, edge_index)  # [N, out_channels]
```

## Package Structure

| File            | Responsibility                                                           |
|:----------------|:-------------------------------------------------------------------------|
| `nsp_layers.py` | `BaseNSPConv` and the five concrete variants; state-augmentation logic.  |
| `nsp_model.py`  | `NSPModel` wrapper, `NSPConfig` dataclass, `NSPVariant` Enum.            |
| `tests/`        | Deterministic unit tests (at repo root under `tests/nsp/`).              |

## Mathematical Properties & Testing

* **Output shape**: Each layer returns `[N, 2d, f]`; the model returns `[N, out_channels]`.
* **History update**: After one step, `x_stalk[:, d:, :]` equals the input current block,
  verifying $\tilde{\mathbf{x}}^{(t+1)}_{\text{hist}} = \mathbf{x}^{(t)}$.
* **Isolated-node recurrence**: Without edges the Laplacian is zero, so
  `new_curr = 2·x_curr − x_hist` and `new_hist = x_curr` exactly.
* **Gradient flow**: All parameters (`W1`, `W2`, `map_generator`) receive non-zero gradients.
* **Permutation invariance**: Reordering nodes yields identical outputs (up to $10^{-5}$).
* **Orthogonality**: The Cayley transform satisfies $W^T W = I$ within $10^{-5}$ tolerance.

### Running Tests

```bash
uv run pytest tests/nsp/
```
