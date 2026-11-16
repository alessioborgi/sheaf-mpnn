"""Download only pinned upstream source files for optional reference tests."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = "d94f81bc7d86bd8d41e416083063f3446c1ad796"
FILES = (
    "LICENSE",
    "main.py",
    "models/sheafhgnn.py",
    "models/sheaf_utils/real_sheaf_builders.py",
    "models/sheaf_utils/real_sheaf_layers.py",
    "configs/base/SheafGeDiDiag.yaml",
    "configs/base/SheafGeDiDiagLight.yaml",
    "configs/base/SheafHyperGNNDiag.yaml",
    "models/sheafgedi.py",
    "models/mlp.py",
    "models/sheaf_utils/directed_sheaf_builders.py",
    "models/sheaf_utils/directed_sheaf_layers.py",
    "models/sheaf_utils/real_orthogonal.py",
    "models/sheaf_utils/utils.py",
    "models/complex_utils/complex_linear.py",
    "models/complex_utils/complex_norm.py",
    "models/complex_utils/complex_operators.py",
    "models/complex_utils/complex_sparse.py",
    "models/complex_utils/normalized_linear.py",
)


def main():
    root = Path("tmp/dshn-source") / COMMIT
    checksums = {}
    for relative in FILES:
        path = root / relative
        if not path.exists():
            url = f"https://raw.githubusercontent.com/EmaMule/DirectionalSheafHypergraphs/{COMMIT}/{relative}"
            with urlopen(url, timeout=60) as response:
                content = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        checksums[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(relative, checksums[relative], flush=True)
    (root / "manifest.json").write_text(
        json.dumps({"commit": COMMIT, "sha256": checksums}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
