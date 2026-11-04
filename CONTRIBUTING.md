# Contributing

Thank you for your interest in improving `sheaf_mpnn`. Bug reports, questions,
documentation fixes, new model families and new datasets are all welcome.

## Reporting bugs and asking questions

Open an issue on
[GitHub](https://github.com/alessioborgi/sheaf-mpnn/issues).
For bugs, include the command or a minimal script that reproduces the problem,
the full traceback, and the output of

```bash
python -c "import sys, torch, torch_geometric; print(sys.version, torch.__version__, torch_geometric.__version__)"
```

## Development setup

The project uses [`uv`](https://docs.astral.sh/uv/) and a lock file, so every
contributor gets the same environment:

```bash
git clone https://github.com/alessioborgi/sheaf-mpnn.git
cd sheaf-mpnn
uv sync
uv run prek install            # run the checks below on every commit
```

## Checks

CI runs these commands (the integration suite on a schedule, the rest on every
pull request):

```bash
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run ty check                # static type checking
uv run pytest                  # unit tests with coverage
uv run pytest -m integration   # slow tests: real datasets, short training runs
uv run --only-group docs sphinx-build -W docs docs/_build/html   # documentation
```

`uv run prek run --all-files` runs lint, formatting, type checking and the
hygiene hooks in one go.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Add tests for new behaviour; keep the unit suite fast (mark anything that
   downloads data or trains for long with `@pytest.mark.integration`).
3. Update the documentation (`docs/`, docstrings, README) and add an entry
   under "Unreleased" in `CHANGELOG.md`.
4. Make sure all checks above pass, then open the pull request.

## Code conventions

- Each model family is a sub-package of `src/sheaf_mpnn/` with
  `*_layers.py` (convolutions built on `BaseSheafConv`) and `*_model.py`
  (encoder, convolutions, decoder). Variants share logic through mixins rather
  than copied code. Heterogeneous mirrors live in `src/hetsheaf_mpnn/`.
- New model types are registered in `src/exp/registries/models.py` (factory)
  and `src/exp/config.py` (`ModelType`); new datasets in
  `src/exp/registries/datasets.py` and `src/exp/data.py`; new presets in
  `src/exp/registries/presets.py`.
- Public functions and classes have Google-style docstrings; they are rendered
  in the API reference.
- Comments and docstrings are ASCII only.

## Code of conduct

Everyone taking part in the project is expected to follow the
[Code of Conduct](https://github.com/alessioborgi/sheaf-mpnn/blob/main/CODE_OF_CONDUCT.md).

## License

By contributing you agree that your contributions are licensed under the
project's MIT License.
