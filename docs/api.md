# API reference

Generated from the docstrings of the three packages:

- {py:mod}`sheaf_mpnn`: the sheaf neural network layers and models. Every
  public model, convolution and variant enum is re-exported at the top level
  and listed in `sheaf_mpnn.__all__`.
- {py:mod}`hetsheaf_mpnn`: heterogeneous mirrors of every family, with
  restriction maps conditioned on node and edge types.
- {py:mod}`exp`: the experiment runner behind the `sheaf` command line
  (configuration, dataset loaders, Lightning modules, registries, sweeps and
  the efficiency benchmark).

Undirected Hypergraph Sheaf Neural Networks expose `HSNNModel`, `HSNNVariant`,
`BaseHypergraphSheafConv` and four convolution variants through `sheaf_mpnn`.
See [HSNN usage and mathematical conventions](hsnn.md) for the incidence API;
these models use a separate hypergraph base. Run the audited Cora hypergraph
experiment with `sheaf run --preset hsnn_cora`.

```{toctree}
:maxdepth: 2

api/sheaf_mpnn/index
api/hetsheaf_mpnn/index
api/exp/index
hsnn
hsnn-implementation-zh
hsnn-cora-protocol-zh
hsnn-sweep-zh
```

## Directed hypergraph sheaves

See [DHSNN API and operator conventions](dhsnn.md) for the implemented magnetic message-passing layers, four map families, complex model, and audited Cora protocol.
