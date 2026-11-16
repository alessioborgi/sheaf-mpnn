# Hypergraph data and split recheck — 2026-11-13

Reference: EmaMule/DirectionalSheafHypergraphs commit
`d94f81bc7d86bd8d41e416083063f3446c1ad796`.

All 14 benchmark artifacts passed numeric reference fingerprints. The 13
non-Cora datasets were downloaded/prepared again through `ensure_hypergraph`;
the existing Cora training artifact was fingerprint-checked, and Cora was
independently prepared again in the reference integration test.

The official loader functions were downloaded from the pinned commit and
executed by the integration tests. Comparison covers features (zero tolerance),
labels, node/hyperedge incidence, direction labels, hyperedge counts, and every
train/validation/test index in all ten splits. Incidences are sorted before
comparison; archive bytes and incidence ordering need not match upstream.

| Split source | Datasets | Result |
| --- | --- | --- |
| Official predefined masks | Roman-empire, Squirrel, Telegram, Chameleon, Cornell, Wisconsin, Amazon-ratings, Texas | All 8 passed, all 10 splits each |
| Official random split function, seed 0, ten sequential draws, 50/25/25 | email-EU, email-Enron, Citeseer, Cora, Molecular-1, Molecular-2 | All 6 passed, all 10 splits each |

Molecular labels/splits index hyperedges; the other twelve index nodes.
HSNN and DHSNN load the same frozen split artifacts. HSNN ignores direction
labels during propagation; DHSNN uses them. Ordinary graph presets are separate.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest tests/integration/test_hypergraph_reference.py -m integration --no-cov
# 13 passed; no skips
.venv/Scripts/python.exe -m pytest tests/integration/test_hsnn_cora_reference.py -m integration --no-cov
# 1 passed; no skips
.venv/Scripts/python.exe -m pytest tests/exp/test_hypergraph_download.py tests/exp/test_hypergraph_catalog.py --no-cov
# 5 passed; no skips
```

Reference tests require the downloaded source cache under
`tmp/hsnn-cora-source` and prepared datasets. They skip when caches are absent;
those skips must not be reported as successful data verification.

## Scope and remaining distinctions

- Exact split parity refers to the public implementation's default seed 0,
  default proportions and ten runs. Private paper-run split indices are not
  available for independent comparison. Changing the model seed locally does
  not regenerate the frozen split bank.
- Telegram and WebKB download through the same external dataset loader used
  upstream. Final numeric fingerprints reject changed external data or masks.
- This verifies the 14 requested table benchmarks, not every additional dataset
  mentioned elsewhere in the upstream repository.
- Training metadata in `hsnn_run.py` still describes all datasets as random
  50/25/25 splits. That description is inaccurate for predefined-mask datasets;
  actual loaded indices are correct, as verified above. No training-source
  changes were made during this audit, so existing search signatures are intact.
- Molecular evaluation now uses the upstream harmonic mean of macro precision
  and macro recall for checkpoint selection, sweep scores and final testing.
  Both HSNN and DHSNN passed two-epoch CLI training checks on real Molecular
  data. These short checks validate execution, not converged paper scores.

Downloaded data are retained as ignored runtime caches for subsequent runs.
