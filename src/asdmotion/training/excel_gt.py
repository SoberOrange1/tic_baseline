"""Excel tic annotations → frame intervals and per-window 0/1 labels (no OpenPose / MMAction imports)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def norm_token(s: str) -> str:
    t = str(s).strip().lower().replace("-", "_")
    t = re.sub(r"\s+", "_", t)
    return t


def video_stem_from_frame_dir(frame_dir: str) -> str:
    if "_" in frame_dir:
        head, tail = frame_dir.rsplit("_", 1)
        if tail.isdigit():
            return head
    return frame_dir


def pick_column(df, candidates: Sequence[str], *, contains: bool = False) -> Optional[str]:
    cols = {c: norm_token(c) for c in df.columns}
    cand_n = [norm_token(x) for x in candidates]
    for c, cn in cols.items():
        for cand in cand_n:
            if cn == cand or (contains and cand in cn):
                return c
    return None


def series_to_seconds(ser) -> np.ndarray:
    import pandas as pd

    s = ser
    if pd.api.types.is_timedelta64_dtype(s.dtype):
        return s.dt.total_seconds().to_numpy(dtype=np.float64)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.99:
        return num.to_numpy(dtype=np.float64)
    td = pd.to_timedelta(s, errors="coerce")
    if td.notna().mean() > 0.99:
        return td.dt.total_seconds().to_numpy(dtype=np.float64)
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        base = s.min()
        return (s - base).dt.total_seconds().to_numpy(dtype=np.float64)
    raise ValueError(
        f"Could not parse time column {getattr(ser, 'name', '?')!r} to seconds. "
        "Use numeric seconds from clip start or HH:MM:SS / timedelta strings."
    )


def row_is_positive_smm(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True
    if isinstance(val, (int, np.integer)):
        return int(val) == 1
    t = str(val).strip().lower()
    if t in ("0", "no", "false", "none", "nan", ""):
        return False
    if t in ("1", "yes", "true", "y"):
        return True
    return any(k in t for k in ("smm", "stereo", "repet", "motor", "vocal", "phonic", "rrb"))


def load_positive_intervals_from_excel(
    path: Path,
    *,
    video_stem: str,
    fps: float,
    sheet: Optional[str] = None,
    strict: bool = True,
) -> List[Tuple[int, int]]:
    """Return half-open frame intervals ``[gs, ge)`` overlapping positive tic rows for this video."""
    import pandas as pd

    df = pd.read_excel(path, sheet_name=sheet or 0, engine="openpyxl")
    df.columns = [norm_token(c) for c in df.columns]

    vcol = pick_column(
        df,
        ("video", "filename", "file", "clip", "name", "assessment", "video_id", "id"),
        contains=True,
    )
    if vcol is None:
        raise ValueError(
            f"Could not find a video column in {path}. Expected something like "
            "``video``, ``filename``, ``assessment``, …"
        )
    scol = pick_column(
        df,
        ("start_time", "begin_time", "t_start", "onset", "begin", "start"),
        contains=True,
    )
    ecol = pick_column(
        df,
        ("end_time", "finish_time", "t_end", "offset", "finish", "end"),
        contains=True,
    )
    if scol is None or ecol is None:
        raise ValueError(
            f"Could not find start/end time columns in {path}. "
            "Need e.g. ``Timestamp (start)`` / ``start_time`` + matching end column."
        )

    stem_n = norm_token(video_stem)
    mask = df[vcol].astype(str).map(lambda x: stem_n in norm_token(x) or norm_token(x) in stem_n)
    sub = df.loc[mask].copy()
    if sub.empty:
        if strict:
            raise ValueError(
                f"No Excel rows matched video stem {video_stem!r} (column {vcol!r}). "
                "Check spelling vs dataset ``frame_dir`` stem."
            )
        return []

    lcol = pick_column(
        sub,
        ("label", "class", "movement", "behavior", "type", "annotation", "smm"),
        contains=True,
    )

    starts = series_to_seconds(sub[scol])
    ends = series_to_seconds(sub[ecol])
    intervals: List[Tuple[int, int]] = []
    for i in range(len(sub)):
        if lcol is not None and not row_is_positive_smm(sub.iloc[i][lcol]):
            continue
        ts, te = float(starts[i]), float(ends[i])
        if te < ts:
            ts, te = te, ts
        gs = int(np.floor(ts * fps + 1e-9))
        ge_ex = int(np.ceil(te * fps - 1e-9))
        if ge_ex <= gs:
            ge_ex = gs + 1
        intervals.append((gs, ge_ex))
    return intervals


def labels_from_excel_overlap(
    annotations: List[dict],
    intervals: List[Tuple[int, int]],
) -> np.ndarray:
    """y[i] = 1 iff window ``annotations[i]`` ``[start,end)`` overlaps any GT half-open interval."""
    y = np.zeros(len(annotations), dtype=int)
    for i, a in enumerate(annotations):
        if not isinstance(a, dict):
            raise TypeError(f"annotations[{i}] is not a dict")
        ws, we = int(a["start"]), int(a["end"])
        for gs, ge in intervals:
            if ws < ge and we > gs:
                y[i] = 1
                break
    return y


def labeled_windows_from_dataset_pkl(
    dataset_pkl: Path,
    annotation_xlsx: Path,
    *,
    fps: float,
    sheet: Optional[str] = None,
    strict_excel: bool = False,
) -> List[dict]:
    """
    Load one ``*_dataset_*.pkl``, assign ``label`` in ``{0,1}`` from Excel intervals, return ``annotations`` copies.

    If ``strict_excel`` is False and no sheet row matches the video stem, every window is labeled 0.
    """
    import pickle

    with open(dataset_pkl, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or "annotations" not in bundle:
        raise ValueError(f"Expected dict with 'annotations' in {dataset_pkl}")
    ann = bundle["annotations"]
    if not isinstance(ann, list) or not ann:
        raise ValueError(f"Empty annotations in {dataset_pkl}")
    stem = video_stem_from_frame_dir(str(ann[0].get("frame_dir", "")))
    if not stem:
        raise ValueError(f"Could not infer video stem from frame_dir in {dataset_pkl}")
    intervals = load_positive_intervals_from_excel(
        annotation_xlsx,
        video_stem=stem,
        fps=float(fps),
        sheet=sheet,
        strict=strict_excel,
    )
    y = labels_from_excel_overlap(ann, intervals)
    out: List[dict] = []
    for i, a in enumerate(ann):
        d = dict(a)
        d["label"] = int(y[i])
        out.append(d)
    return out
