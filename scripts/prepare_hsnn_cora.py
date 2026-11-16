"""Compatibility imports for the packaged, audited Cora preparation."""
from exp.datasets.hypergraph_cora import *  # noqa: F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("exp.datasets.hypergraph_cora", run_name="__main__")
