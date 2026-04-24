import time
from argparse import ArgumentParser
from os import path as osp
from pathlib import Path

from omegaconf import OmegaConf

from asdmotion.detector.detector import Predictor
from asdmotion.detector.preprocess import VideoTransformer
from asdmotion.logger import LogManager
from asdmotion.pipeline.holistic_pose import resolve_holistic_landmarks_json
from asdmotion.utils import load_config

logger = LogManager.APP_LOGGER

_VIDEO_SUFFIXES = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg"}
)


def _collect_videos_under_root(root: str, *, recursive: bool) -> list[str]:
    r = Path(root).expanduser().resolve()
    if not r.is_dir():
        raise FileNotFoundError(f"Not a directory: {r}")
    it = r.rglob("*") if recursive else r.iterdir()
    found = [str(p.resolve()) for p in it if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES]
    if not found:
        raise ValueError(
            f"No video files ({', '.join(sorted(_VIDEO_SUFFIXES))}) under {r} "
            f"(recursive={recursive})."
        )
    return sorted(found)


def predict_video(vt, vpath, p=None):
    v = osp.splitext(osp.basename(vpath))[0]
    logger.info(f'Starting video creation: {vpath}\n\tResults will be saved to {osp.join(vt.work_dir, v)}')
    s = time.time()
    vinf = vt.prepare_environment(vpath)
    if p is not None:
        p.annotate_video(vinf)
    else:
        logger.info(
            'Skipping inference / CSV outputs (``*_predictions.pkl``, scores, annotations). '
            'Dataset pickle and MMAction cfg are under %s',
            vinf.get('dataset_path', '?'),
        )
    t = time.time()
    delta = t - s
    logger.info(f'Total {int(delta // 3600):02d}:{int((delta % 3600) // 60):02d}:{delta % 60:05.2f} for {v}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-cfg", "--cfg", type=str)
    parser.add_argument("-video", "--video_path")
    parser.add_argument("-out", "--out_path")
    parser.add_argument(
        "-gpu",
        "--gpu",
        type=int,
        default=0,
        help="GPU index for MMAction2 / OpenPose / child detector (>=0). Use -1 to force CPU inference.",
    )
    # Do not use a short flag like -holistic-json: argparse treats leading -h as --help.
    parser.add_argument(
        "--holistic-json",
        "--holistic_landmarks_json",
        dest="holistic_landmarks_json",
        type=str,
        default=None,
        help="Single Holistic ``*_landmarks.json`` path (same for all videos in this run). OpenPose skipped.",
    )
    parser.add_argument(
        "--holistic-output-root",
        type=str,
        default=None,
        help="Holistic export root (tic_holistic ``DATA/output`` layout). For each video, picks the JSON "
        "whose parent folder name equals the video basename (stem). Overrides ``--holistic-json`` when set.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only run pose + sliding windows: write ``*_dataset_*.pkl`` and MMAction cfg under "
        "``out_path``; do not run MMAction test or write predictions / score CSVs.",
    )
    parser.add_argument(
        "--video-list",
        type=str,
        default=None,
        help="Text file: one video path per line (# comments). Lower priority than ``--videos-root``.",
    )
    parser.add_argument(
        "--videos-root",
        type=str,
        default=None,
        help="Directory tree to scan for **video** files only (common extensions). Recursive by default; "
        "overrides cfg ``video_path`` and ``--video-list``. Not related to Holistic JSON paths.",
    )
    parser.add_argument(
        "--videos-root-non-recursive",
        action="store_true",
        help="With ``--videos-root``, only scan immediate children (no subfolders).",
    )
    args = parser.parse_args()

    args_dict = {k: v for k, v in vars(args).items() if k != "cfg"}
    args_dict.pop("skip_inference", None)
    videos_root = args_dict.pop("videos_root", None)
    videos_root_non_recursive = args_dict.pop("videos_root_non_recursive", None)
    video_list_path = args_dict.pop("video_list", None)
    holistic_output_root_cli = args_dict.pop("holistic_output_root", None)
    if not args_dict.get("holistic_landmarks_json"):
        args_dict.pop("holistic_landmarks_json", None)
    cfg = OmegaConf.merge(load_config(args.cfg), OmegaConf.create(args_dict))

    work_dir = cfg.out_path

    if videos_root:
        video_paths = _collect_videos_under_root(
            videos_root,
            recursive=not bool(videos_root_non_recursive),
        )
        logger.info("Collected %s video(s) under %s", len(video_paths), videos_root)
    elif video_list_path:
        if not osp.isfile(video_list_path):
            raise FileNotFoundError(f"Video list not found: {video_list_path}")
        video_paths = []
        with open(video_list_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                video_paths.append(s)
        if not video_paths:
            raise ValueError(f"No video paths in {video_list_path}")
    else:
        video_path = cfg.video_path
        if not video_path:
            raise ValueError(
                "Set ``video_path`` in cfg, or pass ``--videos-root``, or ``--video-list``."
            )
        if not osp.exists(video_path):
            raise FileNotFoundError(f"Video path {video_path} does not exist.")
        video_paths = [video_path]

    for vp in video_paths:
        if not osp.exists(vp):
            raise FileNotFoundError(f"Video path does not exist: {vp}")

    logger.info(
        "Dataset-only / full run on %s video(s); results under %s",
        len(video_paths),
        work_dir,
    )
    holistic_json = getattr(cfg, "holistic_landmarks_json", None) or getattr(
        cfg, "holistic_json", None
    )
    holistic_output_root = holistic_output_root_cli or getattr(cfg, "holistic_output_root", None)
    if holistic_output_root:
        holistic_output_root = str(Path(holistic_output_root).expanduser().resolve())
        logger.info(
            "Per-video Holistic JSON under holistic_output_root=%r (parent dir name == video stem).",
            holistic_output_root,
        )
        vt_init_holistic = None
    else:
        vt_init_holistic = holistic_json

    vt = VideoTransformer(
        work_dir,
        cfg.model_name,
        cfg.open_pose_path,
        cfg.child_detection,
        cfg.sequence_length,
        cfg.step_size,
        cfg.gpu,
        cfg.num_person_in,
        cfg.num_person_out,
        holistic_landmarks_json=vt_init_holistic,
    )
    if args.skip_inference:
        p = None
        logger.info("skip-inference: not loading MMAction / not writing predictions.")
    else:
        p = Predictor(
            work_dir,
            cfg.model_name,
            cfg.classification_threshold,
            ["NoAction", "Stereotypical"],
            cfg.mmlab_python_path,
            cfg.mmaction_path,
            cfg.gpu,
        )
    for video_path in video_paths:
        logger.info("Processing: %s", video_path)
        if holistic_output_root:
            jp = resolve_holistic_landmarks_json(holistic_output_root, video_path)
            vt.set_holistic_landmarks_json(jp)
            logger.info("Holistic JSON (resolved): %s", jp)
        predict_video(vt=vt, p=p, vpath=video_path)
