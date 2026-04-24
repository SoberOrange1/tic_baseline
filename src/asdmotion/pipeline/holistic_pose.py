"""
Build ASDMotion / PoseC3D skeleton pickles from MediaPipe Holistic ``*_landmarks.json``.

When ``holistic_landmarks_json`` is set on :class:`asdmotion.detector.preprocess.VideoTransformer`,
OpenPose is skipped and pose is taken from ``pose_landmarks`` (33 points per frame), mapped to
COCO-17 (same topology as :class:`asdmotion.pipeline.skeleton_layout.COCO_LAYOUT`).

Holistic ``x``, ``y`` are normalized to [0, 1]; they are scaled by video width/height from
``get_video_properties`` to pixel coordinates, matching the OpenPose-based pipeline.
"""
from __future__ import annotations

import bisect
import json
import logging
from os import path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

POSE_NUM_JOINTS = 33


def resolve_holistic_landmarks_json(holistic_output_root: str, video_path: str) -> str:
    """
    Find ``*_landmarks.json`` for ``video_path`` under a tic_holistic-style export tree, e.g.::

        {root}/GN-002/GN_002_V1_20251103055634/GN-002_GN_002_V1_20251103055634_landmarks.json

    Rule: parent directory of the JSON file must be named exactly ``Path(video_path).stem``
    (anywhere under ``holistic_output_root``).
    """
    root = Path(holistic_output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"holistic_output_root is not a directory: {root}")
    stem = Path(video_path).stem
    candidates = sorted(
        p.resolve()
        for p in root.rglob("*_landmarks.json")
        if p.is_file() and p.parent.name == stem
    )
    if not candidates:
        raise FileNotFoundError(
            f"No ``*_landmarks.json`` under {root!r} whose parent folder is named {stem!r} "
            f"(expected …/{{group}}/{stem}/*_landmarks.json). Check Holistic export layout vs video basename."
        )
    if len(candidates) > 1:
        logger.warning(
            "Multiple Holistic JSONs for stem %r under %s; using %s",
            stem,
            root,
            candidates[0],
        )
    return str(candidates[0])

# COCO-17 order: nose, leye, reye, lear, rear, lsho, rsho, lelb, relb, lwri, rwri,
# lhip, rhip, lkne, rkne, lank, rank  -> MediaPipe Pose landmark indices.
MP33_TO_COCO17 = np.array(
    [
        0,  # nose
        2,  # left eye (approx center)
        5,  # right eye
        7,  # left ear
        8,  # right ear
        11,
        12,  # shoulders
        13,
        14,  # elbows
        15,
        16,  # wrists
        23,
        24,  # hips
        25,
        26,  # knees
        27,
        28,  # ankles
    ],
    dtype=np.int64,
)


def _pose_list_to_matrix(pose_landmarks: Any) -> np.ndarray:
    """``(33, 4)`` — x, y, z, visibility (visibility defaults to 0 if absent)."""
    out = np.zeros((POSE_NUM_JOINTS, 4), dtype=np.float32)
    if not pose_landmarks or not isinstance(pose_landmarks, list):
        return out
    for i, p in enumerate(pose_landmarks[:POSE_NUM_JOINTS]):
        if not isinstance(p, dict):
            continue
        out[i, 0] = float(p.get("x", 0.0))
        out[i, 1] = float(p.get("y", 0.0))
        out[i, 2] = float(p.get("z", 0.0))
        v = p.get("visibility", None)
        out[i, 3] = float(v) if v is not None else 0.0
    return out


def _load_sorted_frames(json_path: str) -> Tuple[List[int], np.ndarray]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of frames at {json_path!r}")
    rows: List[Tuple[int, np.ndarray]] = []
    for fr in data:
        if not isinstance(fr, dict):
            continue
        fi = int(fr.get("frame_index", len(rows)))
        mat = _pose_list_to_matrix(fr.get("pose_landmarks"))
        rows.append((fi, mat))
    rows.sort(key=lambda x: x[0])
    if not rows:
        raise ValueError(f"No frames in {json_path!r}")
    fis = [r[0] for r in rows]
    pose = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
    return fis, pose


def _nearest_pose_row(sorted_fis: List[int], pose_l_33_4: np.ndarray, query: int) -> np.ndarray:
    if not sorted_fis:
        return np.zeros((POSE_NUM_JOINTS, 4), dtype=np.float32)
    pos = bisect.bisect_left(sorted_fis, query)
    cand: List[int] = []
    if pos < len(sorted_fis):
        cand.append(pos)
    if pos > 0:
        cand.append(pos - 1)
    best = min(cand, key=lambda i: abs(sorted_fis[i] - query))
    return pose_l_33_4[best].astype(np.float32)


def _resolution_wh(resolution: Any) -> Tuple[float, float]:
    """Width/height in pixels. Accepts list, tuple, numpy array, or OmegaConf ListConfig."""
    if resolution is None:
        return 1920.0, 1080.0
    try:
        flat = np.asarray(resolution, dtype=np.float64).reshape(-1)
        if flat.size >= 2:
            return float(flat[0]), float(flat[1])
    except (TypeError, ValueError):
        pass
    raise TypeError(f"resolution must be a length-2 (w, h) sequence; got {resolution!r}")


def _native_wh_tuple(resolution: Any) -> Tuple[int, int]:
    """``(W, H)`` as plain ``int`` for pickle/MMAction (no OmegaConf in ``*_raw.pkl`` / dataset)."""
    w, h = _resolution_wh(resolution)
    return int(round(w)), int(round(h))


def build_skeleton_from_holistic_json(
    json_path: str,
    *,
    video_path: str,
    resolution: Any,
    fps: Optional[float],
    frame_count: Optional[int],
    name: str,
) -> Dict[str, Any]:
    """
    Return a skeleton dict compatible with :class:`asdmotion.pipeline.splitter.Splitter`
    (same keys as :meth:`asdmotion.pipeline.openpose_executor.OpenposeInitializer.to_poseC3D`).
    """
    json_path = osp.abspath(json_path)
    if not osp.isfile(json_path):
        raise FileNotFoundError(f"Holistic JSON not found: {json_path}")

    sorted_fis, pose_l_33_4 = _load_sorted_frames(json_path)
    w_px, h_px = _resolution_wh(resolution)
    img_wh = _native_wh_tuple(resolution)

    if frame_count is None or int(frame_count) <= 0:
        # Fall back to span of landmark file or row count
        frame_count = int(max(sorted_fis) + 1) if sorted_fis else pose_l_33_4.shape[0]
        logger.warning(
            "frame_count missing from video probe; using %s from Holistic JSON span/count",
            frame_count,
        )
    F = int(frame_count)
    if fps is None or fps <= 0:
        fps = 30.0
        logger.warning("fps missing; defaulting to 30")

    # (F, 33, 4) aligned to video frame indices 0 .. F-1
    dense = np.zeros((F, POSE_NUM_JOINTS, 4), dtype=np.float32)
    for t in range(F):
        dense[t] = _nearest_pose_row(sorted_fis, pose_l_33_4, t)

    # COCO-17 xy in pixels, scores from visibility
    xy = np.zeros((1, F, 17, 2), dtype=np.float32)
    sc = np.zeros((1, F, 17), dtype=np.float32)
    for coco_i, mp_i in enumerate(MP33_TO_COCO17):
        xn = dense[:, mp_i, 0]
        yn = dense[:, mp_i, 1]
        xy[0, :, coco_i, 0] = xn * w_px
        xy[0, :, coco_i, 1] = yn * h_px
        sc[0, :, coco_i] = dense[:, mp_i, 3]

    length_seconds = float(F / max(fps, 1e-6))
    video_basename = name
    result: Dict[str, Any] = {
        "keypoint": xy,
        "keypoint_score": sc,
        "frame_dir": video_basename,
        "video_path": str(video_path),
        "img_shape": img_wh,
        "original_shape": tuple(img_wh),
        "fps": float(fps),
        "length_seconds": length_seconds,
        "frame_count": F,
        "adjust": (0, 0),
        "total_frames": F,
    }
    return result
