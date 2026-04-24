#!/usr/bin/env python3
"""
Build merged ``PoseDataset`` pickle(s) + per-fold MMAction2 configs for **Group KFold** (video-level groups).

* **Windows / labels**: For each ``*_dataset_*.pkl``, assigns ``label`` in ``{0,1}`` from Excel tic intervals
  (same rules as ``evaluate_predictions.py``; see ``asdmotion.training.excel_gt``).
* **CV**: ``sklearn.model_selection.GroupKFold`` on **one group id per source video** (stem from
  ``frame_dir``, same convention as holistic). If only one video is present, falls back to
  shuffled ``KFold`` on windows (with a warning), matching ``holistic/training/train_three_stream.py``.

**One invocation** loads every ``*_dataset_*.pkl`` and the Excel file **once**, then writes **all**
folds (you do **not** need to re-run this script per fold). Training MMAction is still **one
``tools/train.py`` per fold** unless you only use a single split.

Writes under ``--out-dir``::

    fold00/ann.pkl
    fold00/binary_train_config.py
    fold01/...

Optional: ``--asdmotion-out-dir`` recursively collects every ``*_dataset_*.pkl`` under the executor
``-out`` / ``out_path`` tree (no ``--dataset-list`` needed).

Optional: ``--write-fold-manifests`` writes ``foldNN/fold_manifest.json`` (+ ``.txt``) listing **video stems**
in train vs val for each fold (same idea as ``tic_holistic/utils/post_analyze.py`` for the PyTorch pipeline).

Optional: ``--holistic-cv-splits-json`` — use **exact val videos per fold** exported by
``utils/post_analyze.py`` (``holistic_cv_splits_for_baseline.json`` next to ``training_config.json``).
Skips sklearn ``GroupKFold`` so MMAction folds match holistic CV **when** ``frame_dir`` stems align with
holistic ``video_folder`` strings (add optional ``val_video_stems`` per fold in JSON if names differ).

Optional: ``--save-labeled-cache`` saves a pickle of merged labeled windows; ``--from-labeled-cache``
replays only the GroupKFold split + file writes (no Excel / dataset I/O) when tuning ``--n-splits``
or the config template.

Then train with **your** full MMAction2 tree, e.g.::

    python tools/train.py fold00/binary_train_config.py

Paths in ``binary_train_config.py`` are absolute (``repr``) so you can run from any cwd.

Requires: ``pandas``, ``openpyxl``, ``scikit-learn``, ``numpy``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from asdmotion.training.excel_gt import (  # noqa: E402
    labeled_windows_from_dataset_pkl,
    video_stem_from_frame_dir,
)


def _paths_from_list_file(list_file: Path) -> List[Path]:
    raw = list_file.expanduser().resolve().read_text(encoding="utf-8")
    out: List[Path] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(Path(s).expanduser())
    return out


def _collect_dataset_pkls(
    paths: Sequence[Path],
    list_file: Path | None,
    glob_pattern: str | None,
    root: Path | None,
    asdmotion_out_dir: Path | None,
) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        rp = p.expanduser().resolve()
        if not rp.is_file():
            raise SystemExit(f"Not a file: {rp}")
        out.append(rp)
    if list_file is not None:
        lf = list_file.expanduser().resolve()
        if not lf.is_file():
            raise SystemExit(f"Not a file: {lf}")
        for q in _paths_from_list_file(lf):
            rq = q.expanduser().resolve()
            if not rq.is_file():
                raise SystemExit(f"Not a file (from --dataset-list): {rq}")
            out.append(rq)
    if glob_pattern:
        if root is None:
            raise SystemExit("--dataset-glob requires --dataset-root")
        r = root.expanduser().resolve()
        if not r.is_dir():
            raise SystemExit(f"--dataset-root is not a directory: {r}")
        out.extend(sorted(r.glob(glob_pattern)))
    if asdmotion_out_dir is not None:
        d = asdmotion_out_dir.expanduser().resolve()
        if not d.is_dir():
            raise SystemExit(f"--asdmotion-out-dir is not a directory: {d}")
        found = sorted({p.resolve() for p in d.rglob("*_dataset_*.pkl") if p.is_file()})
        if not found:
            raise SystemExit(
                f"No ``*_dataset_*.pkl`` under {d}. Run the executor with ``--skip-inference`` (or full run) "
                "so each video has ``.../asdmotion/<model>/*_dataset_<L>.pkl`` under this tree."
            )
        out.extend(found)
    uniq = sorted({str(p) for p in out})
    if not uniq:
        raise SystemExit(
            "No dataset pickles: pass ``--asdmotion-out-dir``, ``--dataset-pkl``, ``--dataset-list``, "
            "and/or ``--dataset-root`` + ``--dataset-glob``."
        )
    return [Path(s) for s in uniq]


def _write_fold_manifests(
    fold_dir: Path,
    fold_idx: int,
    k_folds: int,
    train_dirs: set,
    val_dirs: set,
) -> None:
    """Human + machine-readable list of video stems (from ``frame_dir``) in train1 vs test1 split."""
    train_stems = sorted({video_stem_from_frame_dir(str(fd)) for fd in train_dirs})
    val_stems = sorted({video_stem_from_frame_dir(str(fd)) for fd in val_dirs})
    payload = {
        "fold": fold_idx,
        "k_folds": k_folds,
        "val_video_stems": val_stems,
        "train_video_stems": train_stems,
        "n_val_videos": len(val_stems),
        "n_train_videos": len(train_stems),
    }
    jpath = fold_dir / "fold_manifest.json"
    jpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"fold={fold_idx:02d} / k_folds={k_folds}",
        "",
        f"# val videos ({len(val_stems)})",
        *[f"  {s}" for s in val_stems],
        "",
        f"# train videos ({len(train_stems)})",
        *[f"  {s}" for s in train_stems],
        "",
    ]
    (fold_dir / "fold_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def _dirs_for_holistic_val_keys(
    all_ann: List[dict],
    identifier: str,
    val_keys: set,
) -> Tuple[set, set]:
    """
    Partition ``frame_dir`` / ``filename`` ids into train vs val using holistic val name lists.

    A window is **val** if ``video_stem_from_frame_dir(id) in val_keys`` or ``id in val_keys``.
    """
    train_dirs: set = set()
    val_dirs: set = set()
    for ann in all_ann:
        fid = str(ann[identifier])
        stem = video_stem_from_frame_dir(fid)
        if stem in val_keys or fid in val_keys:
            val_dirs.add(fid)
        else:
            train_dirs.add(fid)
    return train_dirs, val_dirs


def _load_holistic_cv_folds(path: Path) -> List[dict]:
    raw = path.expanduser().resolve().read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or int(data.get("version", -1)) != 1:
        raise SystemExit(f"Expected holistic CV JSON version 1: {path}")
    folds = data.get("folds")
    if not isinstance(folds, list) or not folds:
        raise SystemExit(f"JSON has no 'folds' list: {path}")
    for i, fd in enumerate(folds):
        if not isinstance(fd, dict) or "fold" not in fd:
            raise SystemExit(f"Invalid fold entry at index {i} in {path}")
        has_v = bool(fd.get("val_video_folder")) or bool(fd.get("val_video_stems"))
        if not has_v:
            raise SystemExit(f"Fold {fd.get('fold')} needs val_video_folder and/or val_video_stems in {path}")
    return sorted(folds, key=lambda x: int(x["fold"]))


def _val_keys_for_fold(fold: dict) -> set:
    keys: set = set()
    for k in ("val_video_folder", "val_video_stems"):
        v = fold.get(k)
        if not v:
            continue
        if not isinstance(v, (list, tuple)):
            raise SystemExit(f"Fold {fold.get('fold')}: {k} must be a list")
        keys.update(str(x) for x in v)
    return keys


def _build_fold_bundle(
    all_ann: List[dict],
    train_dirs: set,
    val_dirs: set,
) -> dict:
    if train_dirs & val_dirs:
        raise RuntimeError("train/val frame_dir overlap (internal error).")
    return {
        "split": {
            "train1": sorted(train_dirs),
            "test1": sorted(val_dirs),
        },
        "annotations": all_ann,
    }


def _render_train_config(
    template_text: str,
    *,
    ann_path: Path,
    work_dir: Path,
    pretrained: Path | None,
    repeat_times: int,
) -> str:
    text = template_text.replace(
        'times=int("1")',
        f'times=int("{int(repeat_times)}")',
    )
    text = text.replace(
        'ann_file = "__ANN_FILE__"',
        f"ann_file = {repr(str(ann_path.resolve()))}",
    )
    text = text.replace(
        'work_dir = "__WORK_DIR__"',
        f"work_dir = {repr(str(work_dir.resolve()))}",
    )
    if pretrained is not None and pretrained.is_file():
        text = text.replace(
            'load_from = "__LOAD_FROM__"',
            f"load_from = {repr(str(pretrained.resolve()))}",
        )
    else:
        text = text.replace('load_from = "__LOAD_FROM__"', "load_from = None")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--annotation-xlsx",
        type=Path,
        default=None,
        help="Excel workbook (required unless ``--from-labeled-cache``).",
    )
    ap.add_argument("--annotation-fps", type=float, default=30.0)
    ap.add_argument("--annotation-xlsx-sheet", type=str, default=None)
    ap.add_argument(
        "--dataset-pkl",
        type=Path,
        nargs="*",
        default=(),
        help="One or more ``*_dataset_*.pkl`` paths (repeat flag).",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root directory for ``--dataset-glob``.",
    )
    ap.add_argument(
        "--dataset-glob",
        type=str,
        default=None,
        help="Glob relative to ``--dataset-root``, e.g. ``**/*_dataset_200.pkl``.",
    )
    ap.add_argument(
        "--dataset-list",
        type=Path,
        default=None,
        help="Text file: one ``*_dataset_*.pkl`` path per line; ``#`` starts a comment.",
    )
    ap.add_argument(
        "--asdmotion-out-dir",
        type=Path,
        default=None,
        help="Executor ``-out`` / ``out_path`` root: recursively collect all ``*_dataset_*.pkl`` "
        "(typical layout ``<out>/<video>/asdmotion/<model>/*.pkl``). Combines with other dataset sources.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Only used when falling back to KFold (single video).",
    )
    ap.add_argument(
        "--strict-excel",
        action="store_true",
        help="Require at least one Excel row per video; otherwise that video is all-label-0.",
    )
    ap.add_argument(
        "--pretrained-pkl",
        type=Path,
        default=None,
        help="Optional ``.pth`` for ``load_from`` in generated configs (e.g. shipped asdmotion weights).",
    )
    ap.add_argument(
        "--repeat-times",
        type=int,
        default=1,
        help="``RepeatDataset`` ``times=`` in train dataloader (default 1 for small K-fold sets).",
    )
    ap.add_argument(
        "--template",
        type=Path,
        default=ROOT / "resources" / "mmaction_template" / "binary_train_kfold_template.py",
        help="Config template with placeholders __ANN_FILE__, __WORK_DIR__, __LOAD_FROM__, __REPEAT_TIMES__.",
    )
    ap.add_argument(
        "--save-labeled-cache",
        action="store_true",
        help="After labeling, write ``--labeled-cache-out`` (default: <out-dir>/labeled_cache.pkl).",
    )
    ap.add_argument(
        "--labeled-cache-out",
        type=Path,
        default=None,
        help="Path for ``--save-labeled-cache`` (default: <out-dir>/labeled_cache.pkl).",
    )
    ap.add_argument(
        "--from-labeled-cache",
        type=Path,
        default=None,
        help="Skip Excel and all ``--dataset-pkl`` reads; split from this cache only (fast re-K-fold).",
    )
    ap.add_argument(
        "--write-fold-manifests",
        action="store_true",
        help="Per fold, write ``fold_manifest.json`` and ``fold_manifest.txt`` (video stems in val vs train).",
    )
    ap.add_argument(
        "--holistic-cv-splits-json",
        type=Path,
        default=None,
        help="``holistic_cv_splits_for_baseline.json`` from ``tic_holistic/utils/post_analyze.py`` "
        "(val videos per fold). Overrides sklearn GroupKFold; fold count comes from this file.",
    )
    args = ap.parse_args()

    if args.from_labeled_cache is None:
        if args.annotation_xlsx is None:
            raise SystemExit("Pass ``--annotation-xlsx`` unless using ``--from-labeled-cache``.")
        has_ds = (
            bool(args.dataset_pkl)
            or args.dataset_list is not None
            or (bool(args.dataset_glob) and args.dataset_root is not None)
            or args.asdmotion_out_dir is not None
        )
        if not has_ds:
            raise SystemExit(
                "Provide dataset pickles: ``--asdmotion-out-dir``, ``--dataset-pkl``, ``--dataset-list``, "
                "and/or ``--dataset-root`` + ``--dataset-glob``."
            )

    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tpl_path = args.template.expanduser().resolve()
    if not tpl_path.is_file():
        raise SystemExit(f"Template not found: {tpl_path}")
    template_text = tpl_path.read_text(encoding="utf-8")

    if args.from_labeled_cache is not None:
        cache_path = args.from_labeled_cache.expanduser().resolve()
        if not cache_path.is_file():
            raise SystemExit(f"Not found: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict) or cache.get("version") != 1:
            raise SystemExit(f"Invalid labeled cache (expected version=1 dict): {cache_path}")
        all_ann = cache["annotations"]
        groups = list(cache["groups"])
        identifier = str(cache["identifier"])
        if not isinstance(all_ann, list) or not all_ann:
            raise SystemExit("Cache has empty annotations.")
    else:
        xlsx = args.annotation_xlsx.expanduser().resolve()
        if not xlsx.is_file():
            raise SystemExit(f"Not found: {xlsx}")

        pkls = _collect_dataset_pkls(
            args.dataset_pkl,
            args.dataset_list,
            args.dataset_glob,
            args.dataset_root,
            args.asdmotion_out_dir,
        )

        all_ann = []
        groups = []
        stem_to_gid: Dict[str, int] = {}
        next_gid = 0

        for dp in pkls:
            labeled = labeled_windows_from_dataset_pkl(
                dp,
                xlsx,
                fps=float(args.annotation_fps),
                sheet=args.annotation_xlsx_sheet,
                strict_excel=args.strict_excel,
            )
            stem = video_stem_from_frame_dir(str(labeled[0].get("frame_dir", "")))
            if stem not in stem_to_gid:
                stem_to_gid[stem] = next_gid
                next_gid += 1
            gid = stem_to_gid[stem]
            for a in labeled:
                all_ann.append(a)
                groups.append(gid)

        identifier = "filename" if all_ann and "filename" in all_ann[0] else "frame_dir"

        if args.save_labeled_cache:
            cache_out = (
                args.labeled_cache_out.expanduser().resolve()
                if args.labeled_cache_out is not None
                else (out_root / "labeled_cache.pkl")
            )
            cache_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "annotations": all_ann,
                "groups": np.asarray(groups, dtype=np.int64),
                "identifier": identifier,
            }
            with open(cache_out, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Wrote labeled cache ({len(all_ann)} windows) -> {cache_out}")

    n = len(all_ann)
    g_arr = np.asarray(groups, dtype=np.int64)
    n_groups = len(np.unique(g_arr))

    holistic_path = args.holistic_cv_splits_json
    if holistic_path is not None:
        hpath = holistic_path.expanduser().resolve()
        if not hpath.is_file():
            raise SystemExit(f"Not found: {hpath}")
        holistic_folds = _load_holistic_cv_folds(hpath)
        k_json = len(holistic_folds)
        if int(args.n_splits) != k_json:
            warnings.warn(
                f"--n-splits={args.n_splits} ignored when using holistic CV JSON ({k_json} folds from {hpath.name}).",
                UserWarning,
                stacklevel=1,
            )
        split_dir_sets: List[Tuple[set, set]] = []
        for fd in holistic_folds:
            val_keys = _val_keys_for_fold(fd)
            train_dirs, val_dirs = _dirs_for_holistic_val_keys(all_ann, identifier, val_keys)
            if not val_dirs:
                raise SystemExit(
                    f"Holistic fold {fd.get('fold')}: no baseline windows matched val keys {sorted(val_keys)[:8]}… "
                    f"Check that PoseDataset ``frame_dir`` stems match holistic ``video_folder`` "
                    f"(or add ``val_video_stems`` in the JSON for this fold)."
                )
            if not train_dirs:
                raise SystemExit(f"Holistic fold {fd.get('fold')}: train set is empty (check val keys).")
            used_keys: set = set()
            for ann in all_ann:
                fid = str(ann[identifier])
                stem = video_stem_from_frame_dir(fid)
                if stem not in val_keys and fid not in val_keys:
                    continue
                if stem in val_keys:
                    used_keys.add(stem)
                if fid in val_keys:
                    used_keys.add(fid)
            unused_keys = val_keys - used_keys
            if unused_keys:
                warnings.warn(
                    f"Holistic fold {fd.get('fold')}: val keys never matched any window: "
                    f"{sorted(unused_keys)[:24]}",
                    UserWarning,
                    stacklevel=1,
                )
            split_dir_sets.append((train_dirs, val_dirs))
        k_eff = k_json
    else:
        split_dir_sets = []
        from sklearn.model_selection import GroupKFold, KFold

        k_req = max(2, int(args.n_splits))
        if n < 2:
            raise SystemExit(f"Need at least 2 windows for CV; found {n}.")

        if n_groups < 2:
            warnings.warn(
                "Only one video group; using KFold on windows (not video-level GroupKFold). "
                "Add more videos for leakage-free CV.",
                UserWarning,
                stacklevel=1,
            )
            k_eff = max(2, min(k_req, n))
            if k_eff != k_req:
                warnings.warn(f"Adjusted n_splits from {k_req} to {k_eff} (n_samples={n}).", UserWarning, stacklevel=1)
            splitter = KFold(n_splits=k_eff, shuffle=True, random_state=int(args.seed))
            splits = list(splitter.split(np.arange(n)))
        else:
            k_eff = max(2, min(k_req, n_groups))
            if k_eff != k_req:
                warnings.warn(
                    f"Adjusted n_splits from {k_req} to {k_eff} (n_groups={n_groups}).",
                    UserWarning,
                    stacklevel=1,
                )
            gkf = GroupKFold(n_splits=k_eff)
            splits = list(gkf.split(np.zeros(n), groups=g_arr))

    if holistic_path is None:
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_dirs = {all_ann[i][identifier] for i in train_idx}
            val_dirs = {all_ann[i][identifier] for i in val_idx}
            split_dir_sets.append((train_dirs, val_dirs))

    for fold_idx, (train_dirs, val_dirs) in enumerate(split_dir_sets):
        n_tr = sum(1 for a in all_ann if str(a[identifier]) in train_dirs)
        n_va = sum(1 for a in all_ann if str(a[identifier]) in val_dirs)
        bundle = _build_fold_bundle(all_ann, train_dirs, val_dirs)
        fold_dir = out_root / f"fold{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ann_path = fold_dir / "ann.pkl"
        with open(ann_path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        cfg_path = fold_dir / "binary_train_config.py"
        cfg_body = _render_train_config(
            template_text,
            ann_path=ann_path,
            work_dir=fold_dir,
            pretrained=args.pretrained_pkl,
            repeat_times=args.repeat_times,
        )
        cfg_path.write_text(cfg_body, encoding="utf-8")
        print(
            f"fold{fold_idx:02d}: windows train={n_tr} val={n_va} "
            f"videos train={len(train_dirs)} val={len(val_dirs)} -> {ann_path}"
        )
        if args.write_fold_manifests:
            _write_fold_manifests(fold_dir, fold_idx, len(split_dir_sets), train_dirs, val_dirs)
            print(f"  manifest -> {fold_dir / 'fold_manifest.json'}")

    print(f"\nWrote {len(split_dir_sets)} folds under {out_root}")
    print("Train with: python tools/train.py <fold_dir>/binary_train_config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
