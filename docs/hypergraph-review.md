# Hypergraph implementation review

This branch adds undirected HSNN and directed DHSNN implementations for review
before further hyperparameter tuning. Running successfully does not by itself
establish mathematical correctness.

## Suggested review order

1. `src/sheaf_mpnn/hypergraph_sheaf_conv.py` and
   `src/sheaf_mpnn/directed_hypergraph_sheaf_conv.py`: incidence propagation.
2. `src/sheaf_mpnn/nsd/hsnn_layers.py`, `hsnn_model.py`, `dhsnn_layers.py`,
   and `dhsnn_model.py`: restriction maps, normalization and model architecture.
3. `src/exp/datasets/hypergraph*.py`: data preparation and split handling.
4. Experiment registries, CLI entry points and `configs/hypergraphs/`.
5. Unit tests and independent reference tests under `tests/integration/`.

See [HSNN usage](hsnn.md), [DHSNN usage](dhsnn.md), and the
[DSHN reference correspondence report](dhsnn-official-audit.md).
## Running the core tests

```bash
python -m pytest --no-cov tests/test_hypergraph_sheaf_conv.py tests/test_directed_hypergraph_sheaf_conv.py tests/nsd/test_hsnn_layers.py tests/nsd/test_hsnn_model.py tests/nsd/test_dhsnn_layers.py tests/nsd/test_dhsnn_model.py
```

Please review the incidence convention, normalization, directed complex
transport, dataset protocols, and correspondence with the reference models.
Hyperparameter optimization is still in progress; no final benchmark or
complete correctness claim is made here.
