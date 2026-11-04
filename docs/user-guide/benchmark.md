# Efficiency benchmark

`sheaf benchmark` compares this library's `diagonal`, `orthogonal` and
`general` NSD variants against the original implementation,
[twitter-research/neural-sheaf-diffusion](https://github.com/twitter-research/neural-sheaf-diffusion),
on wall-clock time, peak GPU memory, parameter count and inference latency
under matched architectures. It is designed for a SLURM cluster (jobs are
submitted with `submitit`) and needs the `benchmark` extra.

One-time setup (an isolated Python 3.9 / PyTorch 1.11 environment for the
original code, which is vendored as a git submodule):

```bash
git submodule update --init external/neural-sheaf-diffusion
CUDA_TAG=cu113 bash scripts/benchmark/setup_original_env.sh   # CUDA_TAG=cpu without a GPU
```

Fill in the matched hyperparameters under
`configs/benchmark/bestconf/{ours,original}/<dataset>/<variant>.yaml` and the
`slurm:` block of the manifest, then:

```bash
sheaf benchmark submit  --manifest configs/benchmark/nsd_vs_original.yaml
sheaf benchmark collect --manifest configs/benchmark/nsd_vs_original.yaml
sheaf benchmark report  --manifest configs/benchmark/nsd_vs_original.yaml
```

`submit` is idempotent: re-running it only submits jobs without a result. Add
`--modes efficiency` or `--modes convergence` to work on one axis at a time.
