#!/usr/bin/env python3
"""
Copy a **trimmed** MMAction2 tree sufficient for ASDMotion inference (``tools/test.py`` + ``mmaction``).

What is omitted (typical bulk / unused for ``test.py --dump`` only):

- ``.git``, ``docs/``, ``demo/``, ``tests/``, ``projects/``, ``.github/``, ``mmaction2.egg-info/``
- Everything under ``tools/`` except ``tools/test.py`` (drops ``tools/data/``, training scripts, etc.)
- ``configs/`` (ASDMotion uses a generated ``*_binary_config.py``; official configs are not required at runtime)

After export, install in your **mmlab** env::

    cd /path/to/exported_mmaction2
    pip install -r requirements/build.txt
    pip install -v -e .

Then set ``mmaction_path`` in ``config.yaml`` to this directory.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

def _copytree_compat(src: Path, dst: Path) -> None:
    """``shutil.copytree`` without ``dirs_exist_ok`` (Python 3.7 compatible)."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "mmaction2",
        help="Path to a full MMAction2 clone (default: ../mmaction2 next to this script).",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output directory (created; should not already contain mmaction/).",
    )
    args = ap.parse_args()
    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()

    if (dst / "mmaction").exists():
        raise SystemExit(f"Refusing to overwrite existing mmaction package at {dst / 'mmaction'}")

    dst.mkdir(parents=True, exist_ok=True)

    # Root files needed for pip install -e .
    for name in (
        "setup.py",
        "setup.cfg",
        "MANIFEST.in",
        "LICENSE",
        "README.md",
        "model-index.yml",
        "dataset-index.yml",
    ):
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)

    req_src = src / "requirements"
    if req_src.is_dir():
        _copytree_compat(req_src, dst / "requirements")

    mmaction_src = src / "mmaction"
    if not mmaction_src.is_dir():
        raise SystemExit(f"Missing mmaction package: {mmaction_src}")
    _copytree_compat(mmaction_src, dst / "mmaction")

    tools_dst = dst / "tools"
    tools_dst.mkdir(parents=True, exist_ok=True)
    test_py = src / "tools" / "test.py"
    if not test_py.is_file():
        raise SystemExit(f"Missing {test_py}")
    shutil.copy2(test_py, tools_dst / "test.py")

    print(f"Wrote minimal MMAction2 to: {dst}", file=sys.stderr)
    print("Next: pip install -r requirements/build.txt && pip install -v -e .", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
