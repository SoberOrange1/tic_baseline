#!/usr/bin/env python3
"""
Evaluate ASDMotion ``--dump`` / ``*_predictions.pkl`` against **ground-truth labels**.

**Label sources**

1. ``--labels-csv`` — column ``y`` or ``label`` (0/1), one row per window (dataset order).
2. ``--dataset-pkl`` only — if each ``annotations[i]['label']`` / ``gt`` is already 0/1.
3. ``--annotation-xlsx`` + ``--dataset-pkl`` — Excel holds **video time** (seconds from clip
   start) for SMM segments; script converts with ``--annotation-fps`` (default **30**),
   overlaps each sliding window ``[start, end)`` from the pickle, and sets **y=1** if any
   overlap with a positive row.

**Excel columns** (case-insensitive; spaces → ``_``):

* **Video id** — one of: ``video``, ``filename``, ``file``, ``clip``, ``name``, ``assessment``,
  ``video_id``. Row kept if the cell **matches** ``--video-stem`` (normalized: ``-``/`` `` → ``_``,
  lower-case) as substring / equality after normalization.
* **Start / end time** — e.g. ``start_time`` / ``timestamp (start)`` / ``onset`` and
  ``end_time`` / ``timestamp (end)`` / ``finish``. Values may be **numeric seconds** from clip
  start, or strings like ``HH:MM:SS`` / ``timedelta`` / ``datetime`` (converted when possible).
* **Optional label column** — ``label``, ``class``, ``movement``, ``behavior``, ``type``:
  if present, a row counts as **positive SMM interval** when value is ``1``, ``yes``, ``true``,
  ``smm``, ``stereotypical``, etc.; if absent, **every row** in the filtered sheet is treated
  as a positive SMM interval.

Example::

    python scripts/evaluate_predictions.py \\
        --predictions-pkl results/.../GN_..._predictions.pkl \\
        --dataset-pkl results/.../GN_..._dataset_200.pkl \\
        --annotation-xlsx DATA/annotation/tic_annotation_english.xlsx \\
        --annotation-fps 30 \\
        --video-stem GN_002_V2_20251105104429 \\
        --write-labels-csv /tmp/window_y.csv \\
        --threshold 0.85 \\
        --confusion-matrix-out cm.png

Requires: ``pandas``, ``openpyxl``, ``scikit-learn``; ``matplotlib`` for ``--confusion-matrix-out``.

For **K-fold MMAction2 training** labels + ``ann.pkl``, see ``scripts/build_mmaction_groupkfold_ann.py``.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asdmotion.detector.detector import _predictions_pkl_to_score_matrix  # noqa: E402
from asdmotion.training.excel_gt import (  # noqa: E402
    labels_from_excel_overlap,
    load_positive_intervals_from_excel,
    video_stem_from_frame_dir,
)


def _load_labels_csv(path: Path) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path)
    col = "y" if "y" in df.columns else "label" if "label" in df.columns else None
    if col is None:
        raise SystemExit(
            f"Labels CSV {path} needs a column named ``y`` or ``label`` (0/1 per window)."
        )
    y = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise SystemExit("Labels must be 0 or 1 only.")
    return y


def _labels_from_dataset_pkl(path: Path) -> Tuple[Optional[np.ndarray], str]:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or "annotations" not in bundle:
        raise SystemExit(f"Expected a dict with key 'annotations' in {path}")
    ann = bundle["annotations"]
    if not isinstance(ann, list) or not ann:
        raise SystemExit(f"Empty annotations in {path}")

    raw: list[int] = []
    for i, a in enumerate(ann):
        if not isinstance(a, dict):
            raise SystemExit(f"annotations[{i}] is not a dict")
        v = a.get("label", a.get("gt", -1))
        try:
            raw.append(int(v))
        except (TypeError, ValueError):
            raw.append(-1)

    arr = np.asarray(raw, dtype=int)
    if (arr == -1).all():
        return None, (
            "Every window has label/gt == -1. Use ``--annotation-xlsx`` + ``--dataset-pkl``, "
            "or ``--labels-csv``."
        )
    if np.isin(arr, [0, 1]).all():
        return arr, ""
    bad = sorted(set(arr.tolist()) - {0, 1, -1})
    raise SystemExit(
        f"annotations labels must be 0, 1, or -1 only; found other values: {bad}."
    )


def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray, confusion_matrix_out: Optional[Path]) -> None:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n=== Binary metrics (positive = SMM / class 1) ===")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall:    {rec:.4f}")
    print(f"f1:        {f1:.4f}")
    print("\nConfusion matrix [rows=true 0,1 | cols=pred 0,1]:")
    print(cm)
    print("\nclassification_report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["NoAction (0)", "Stereotypical (1)"],
            digits=4,
            zero_division=0,
        )
    )

    if confusion_matrix_out:
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise SystemExit("pip install matplotlib for --confusion-matrix-out") from e
        out = confusion_matrix_out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion matrix",
        )
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"\nSaved confusion matrix figure to {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-pkl", type=Path, required=True)
    ap.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Per-window ground truth (0/1); same order as dataset annotations.",
    )
    ap.add_argument(
        "--dataset-pkl",
        type=Path,
        default=None,
        help="*_dataset_*.pkl (window list). Required with --annotation-xlsx.",
    )
    ap.add_argument(
        "--annotation-xlsx",
        type=Path,
        default=None,
        help="tic-style annotation workbook; times converted with --annotation-fps.",
    )
    ap.add_argument(
        "--annotation-fps",
        type=float,
        default=30.0,
        help="FPS used to convert Excel times (seconds) → frame indices (default: 30).",
    )
    ap.add_argument(
        "--annotation-xlsx-sheet",
        type=str,
        default=None,
        help="Optional Excel sheet name (default: first sheet).",
    )
    ap.add_argument(
        "--video-stem",
        type=str,
        default=None,
        help="Video basename to filter Excel rows (e.g. GN_002_V2_20251105104429). "
        "Default: inferred from first annotation ``frame_dir``.",
    )
    ap.add_argument(
        "--write-labels-csv",
        type=Path,
        default=None,
        help="If set, write computed per-window y (0/1) for debugging.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Positive (SMM) score threshold; y_pred = 1 iff score[class 1] > threshold.",
    )
    ap.add_argument("--positive-class", type=int, default=1)
    ap.add_argument("--confusion-matrix-out", type=Path, default=None)
    args = ap.parse_args()

    if args.labels_csv is not None and args.annotation_xlsx is not None:
        raise SystemExit("Use either ``--labels-csv`` or ``--annotation-xlsx`` + ``--dataset-pkl``, not both.")
    if args.annotation_xlsx is not None and args.dataset_pkl is None:
        raise SystemExit("--annotation-xlsx requires --dataset-pkl (for window [start,end) list).")

    pkl_path = args.predictions_pkl.resolve()
    if not pkl_path.is_file():
        raise SystemExit(f"Not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    scores = _predictions_pkl_to_score_matrix(raw)
    if scores.shape[0] != 2:
        raise SystemExit(f"Expected 2 x N scores, got shape {scores.shape}")
    n = scores.shape[1]
    pos = scores[args.positive_class]
    neg = scores[1 - args.positive_class]

    print(f"Loaded {pkl_path}")
    print(f"Windows N = {n}")
    print(f"Positive-class (idx {args.positive_class}) score: min={pos.min():.4f} max={pos.max():.4f} mean={pos.mean():.4f}")
    print(f"Other-class score: min={neg.min():.4f} max={neg.max():.4f} mean={neg.mean():.4f}")
    pred_bin = (pos > args.threshold).astype(int)
    print(f"Predicted SMM fraction at threshold {args.threshold}: {pred_bin.mean():.4f}")

    y_true: Optional[np.ndarray] = None

    if args.labels_csv is not None:
        y_true = _load_labels_csv(args.labels_csv.resolve())
    elif args.annotation_xlsx is not None:
        with open(args.dataset_pkl.resolve(), "rb") as f:
            bundle = pickle.load(f)
        ann = bundle["annotations"]
        if not isinstance(ann, list) or len(ann) != n:
            raise SystemExit(
                f"Dataset window count {len(ann) if isinstance(ann, list) else 0} != predictions N={n}."
            )
        stem = args.video_stem or video_stem_from_frame_dir(str(ann[0].get("frame_dir", "")))
        if not stem:
            raise SystemExit("Could not infer --video-stem; pass it explicitly.")
        print(f"Using video stem for Excel filter: {stem!r} (fps={args.annotation_fps})")
        try:
            iv = load_positive_intervals_from_excel(
                args.annotation_xlsx.resolve(),
                video_stem=stem,
                fps=float(args.annotation_fps),
                sheet=args.annotation_xlsx_sheet,
                strict=True,
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e
        print(f"Loaded {len(iv)} positive interval(s) from {args.annotation_xlsx}")
        try:
            y_true = labels_from_excel_overlap(ann, iv)
        except (TypeError, ValueError) as e:
            raise SystemExit(str(e)) from e
        if args.write_labels_csv:
            import pandas as pd

            out = args.write_labels_csv.resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"y": y_true}).to_csv(out, index=False)
            print(f"Wrote per-window labels to {out}")
    elif args.dataset_pkl is not None:
        y_true, msg = _labels_from_dataset_pkl(args.dataset_pkl.resolve())
        if y_true is None:
            print(f"\n{msg}")
            return 0

    if y_true is None:
        print("\nNo labels: pass --labels-csv, or --dataset-pkl with 0/1 labels, "
              "or --annotation-xlsx + --dataset-pkl.")
        return 0

    if y_true.shape[0] != n:
        raise SystemExit(
            f"Label count {y_true.shape[0]} != prediction count {n}. "
            "Align labels to the same sliding-window order as the predictions."
        )

    try:
        _print_metrics(y_true, pred_bin, args.confusion_matrix_out)
    except ImportError as e:
        raise SystemExit("Install scikit-learn: pip install scikit-learn") from e
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
