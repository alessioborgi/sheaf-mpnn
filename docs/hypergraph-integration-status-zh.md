# Hypergraph implementation

HSNN and DHSNN are implemented in src/sheaf_mpnn with four restriction-map families.
The 14 paper benchmarks have separate *-hypergraph presets.
Reference data and split fingerprints are packaged in src/exp/datasets.
Missing datasets are prepared automatically on first use and verified before training.
Search configurations live in configs/hypergraphs, with 250 trials per map family.

Local runs, checkpoints, downloaded datasets, temporary reference checkouts,
virtual environments and historical experiment reports have been removed.
Recreate the environment using uv sync before running the CLI.
Integration tests that need the optional upstream checkout skip when it is absent.
The extra upstream Pubmed, Questions, Dataset-3 and synthetic tasks remain outside
the currently implemented 14-benchmark catalog.