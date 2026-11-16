"""Prepare hypergraphs using the installed dataset implementation."""
import runpy

if __name__ == "__main__":
    runpy.run_module("exp.datasets.hypergraph_prepare", run_name="__main__")
