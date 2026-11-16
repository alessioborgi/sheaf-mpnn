"""Run all benchmark/model/map combinations with a small validation-only budget."""

import copy
import hashlib
import json
from pathlib import Path

from exp.config import ModelType
from exp.datasets.hypergraph import HYPERGRAPHS
from exp.hsnn_run import source_hashes
from exp.registries.presets import preset_registry
from exp.run import run


def main():
    root = Path("runs/hypergraph-verification")
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "status.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    signature = hashlib.sha256(
        json.dumps(source_hashes(), sort_keys=True).encode()
        + Path(__file__).read_bytes()
    ).hexdigest()
    for dataset in HYPERGRAPHS:
        for model in ("hsnn", "dhsnn"):
            for variant in ("diagonal", "orthogonal", "low_rank", "general"):
                key = f"{dataset}-{model}-{variant}"
                previous = report.get(key, {})
                if (
                    previous.get("status") == "passed"
                    and previous.get("source_signature") == signature
                ):
                    continue
                cfg = copy.deepcopy(preset_registry.get(dataset))
                cfg.model.type = ModelType(model)
                cfg.model.variant = variant
                cfg.model.stalk_dim = 2
                cfg.model.hidden_dim = 4
                cfg.model.rank = 1
                cfg.cv.folds = 1
                cfg.hypergraph.evaluate_test = False
                cfg.hardware.cuda = -1
                cfg.optim.epochs = 2
                cfg.optim.early_stopping = 2
                cfg.hypergraph.output = str(root / key)
                attempt = 1
                while Path(cfg.hypergraph.output).exists():
                    cfg.hypergraph.output = str(root / f"{key}-retry-{attempt}")
                    attempt += 1
                try:
                    values = run(cfg)
                    report[key] = {"status": "passed", "validation": values}
                except Exception as exc:
                    report[key] = {"status": "failed", "error": repr(exc)}
                report[key]["source_signature"] = signature
                temporary = report_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
                temporary.replace(report_path)
    if any(row.get("status") != "passed" for row in report.values()):
        raise SystemExit(
            "Some combinations failed; inspect runs/hypergraph-verification/status.json"
        )


if __name__ == "__main__":
    main()
