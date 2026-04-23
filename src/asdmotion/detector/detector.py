import copy
import os
import shlex
import subprocess
import sys
from os import path as osp
from typing import Any, List

import numpy as np
import pandas as pd

from asdmotion.pipeline.aggregator import aggregate
from asdmotion.logger import LogManager
from asdmotion.utils import RESOURCES_ROOT, read_pkl

MODELS_DIR = osp.join(RESOURCES_ROOT, 'models')
logger = LogManager.APP_LOGGER


def _safe_int_gpu_id(gpu_id):
    if gpu_id is None:
        return None
    try:
        return int(gpu_id)
    except (TypeError, ValueError):
        return None


def _mmaction_subprocess_env_for_gpu(gpu_id) -> dict:
    """
    MMEngine ``tools/test.py`` has no ``--gpu-ids``; pin the device via env.

    * ``gpu_id >= 0`` — use that physical GPU index for the subprocess.
    * ``gpu_id < 0`` — hide GPUs (``CUDA_VISIBLE_DEVICES=``) so PyTorch stays on CPU.
    * ``gpu_id is None`` — inherit parent process env (no override).
    """
    env = copy.deepcopy(os.environ)
    if gpu_id is None:
        return env
    try:
        gid = int(gpu_id)
    except (TypeError, ValueError):
        return env
    if gid >= 0:
        env['CUDA_VISIBLE_DEVICES'] = str(gid)
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def _predictions_pkl_to_score_matrix(dumped: Any) -> np.ndarray:
    """
    Normalize MMAction2 / MMEngine ``DumpResults`` pickles to shape ``(2, N)``:
    row ``i`` = class ``i`` score per sliding window (same layout as legacy ``--out`` pickles).

    Legacy format: list of length-2 score vectors per window (then ``np.asarray(dumped).T``).
    MMEngine format: list of ``ActionDataSample`` (or dicts) with ``pred_score`` length 2.
    """
    if isinstance(dumped, np.ndarray):
        a = np.asarray(dumped, dtype=np.float64)
        if a.ndim == 2 and a.shape[0] == 2:
            return a
        if a.ndim == 2 and a.shape[1] == 2:
            return a.T
        raise ValueError(f'Unexpected ndarray shape for predictions: {a.shape}')

    if not isinstance(dumped, (list, tuple)) or len(dumped) == 0:
        raise ValueError(f'Empty or invalid predictions dump: {type(dumped)!r}')

    el0 = dumped[0]
    if isinstance(el0, (list, tuple, np.ndarray)) and not hasattr(el0, 'pred_score'):
        m = np.stack(
            [np.asarray(x, dtype=np.float64).reshape(-1)[:2] for x in dumped],
            axis=0,
        )
        return m.T

    rows: List[np.ndarray] = []
    for item in dumped:
        ps = None
        if hasattr(item, 'pred_score'):
            ps = getattr(item, 'pred_score', None)
        elif isinstance(item, dict):
            ps = item.get('pred_score')
            if ps is None and 'pred_scores' in item:
                ps = item['pred_scores']
        if ps is None:
            continue
        if hasattr(ps, 'detach'):
            ps = ps.detach().cpu().numpy()
        vec = np.asarray(ps, dtype=np.float64).reshape(-1)
        if vec.size >= 2:
            rows.append(vec[:2])
    if not rows:
        raise ValueError(
            'Could not parse predictions pickle (expected legacy list-of-pairs '
            'or MMEngine DumpResults with pred_score).'
        )
    return np.stack(rows, axis=0).T


def _dump_predictions_readable(path: str) -> bool:
    """True if ``path`` exists, is non-empty, and ``read_pkl`` succeeds."""
    try:
        if not osp.isfile(path) or osp.getsize(path) <= 0:
            return False
        read_pkl(path)
        return True
    except Exception:
        return False


class Predictor:
    def __init__(self, work_dir, model_name, binary_threshold, labels, mmlab_python, mmaction_root, gpu_id=None):
        self.work_dir = work_dir
        self.model_name = model_name
        self.binary_model_path = osp.join(MODELS_DIR, self.model_name)
        self.threshold = binary_threshold

        self.labels = labels
        self.mmlab_python = mmlab_python
        self.mmaction_root = mmaction_root
        self.va_columns = ['video', 'video_full_name', 'video_path', 'start_time', 'end_time', 'start_frame', 'end_frame', 'movement', 'calc_date', 'annotator']
        self.gpu_id = gpu_id

    def _predict(self, cfg_path, model_path, out_path):
        if not osp.exists(out_path):
            out_exec = f'\\{out_path}' if out_path.startswith('\\\\') else out_path
            test_script = osp.join(self.mmaction_root, 'tools', 'test.py')
            py = self.mmlab_python
            # MMEngine-era MMAction2: ``tools/test.py`` uses ``--dump`` (not ``--out``) and
            # has no ``--gpu-ids``; pin the device via ``CUDA_VISIBLE_DEVICES``.
            env = _mmaction_subprocess_env_for_gpu(self.gpu_id)
            _gid = _safe_int_gpu_id(self.gpu_id)
            if py and osp.isfile(py):
                argv = [py, test_script, cfg_path, model_path, '--dump', out_exec]
                if _gid is not None and _gid >= 0:
                    logger.info('MMAction2 subprocess: CUDA_VISIBLE_DEVICES=%s', _gid)
                elif _gid is not None and _gid < 0:
                    logger.info('MMAction2 subprocess: forcing CPU (CUDA_VISIBLE_DEVICES empty)')
                logger.info('Executing MMAction2 test: %s', ' '.join(f'"{a}"' if ' ' in str(a) else str(a) for a in argv))
                try:
                    subprocess.check_call(argv, universal_newlines=True, env=env)
                except subprocess.CalledProcessError as e:
                    # Some stacks SIGABRT during interpreter teardown after a successful ``--dump``
                    # (e.g. ``free(): invalid pointer``) even though the pickle is complete.
                    if _dump_predictions_readable(out_exec):
                        logger.warning(
                            'MMAction2 subprocess failed with %s but dump file is readable; continuing.',
                            e,
                        )
                    else:
                        raise
            elif sys.platform == 'win32':
                gpu_prefix = ''
                if _gid is not None and _gid >= 0:
                    gpu_prefix = f'set CUDA_VISIBLE_DEVICES={_gid}&& '
                elif _gid is not None and _gid < 0:
                    gpu_prefix = 'set CUDA_VISIBLE_DEVICES=&& '
                cmd = (
                    f'{gpu_prefix}python "{test_script}" "{cfg_path}" "{model_path}" '
                    f'--dump "{out_exec}"'
                )
                cmd = f'{osp.join(RESOURCES_ROOT, "run_in_env.bat")} {cmd}'.replace('\\', '/')
                logger.info(f'Executing: {cmd}')
                try:
                    subprocess.check_call(cmd, universal_newlines=True, shell=True)
                except subprocess.CalledProcessError as e:
                    if _dump_predictions_readable(out_exec):
                        logger.warning(
                            'MMAction2 subprocess failed with %s but dump file is readable; continuing.',
                            e,
                        )
                    else:
                        raise
            else:
                raise FileNotFoundError(
                    f'MMAction Python not found at mmlab_python_path={py!r} (file must exist). '
                    'On Linux/macOS, set ``mmlab_python_path`` in config.yaml to the interpreter '
                    'that has MMAction2 installed, e.g. ``/path/to/miniconda3/envs/mmlab/bin/python``. '
                    'Do not rely on run_in_env.bat (Windows only).'
                )
            logger.info('Prediction complete successfully.')
        else:
            logger.info(f'Prediction exists: {out_path}')
        scores = read_pkl(out_path)
        return _predictions_pkl_to_score_matrix(scores)

    def _detect_stereotypical_movements(self, video_info):
        dataset = read_pkl(video_info['dataset_path'])['annotations']
        basename, fullname, path, fps = video_info['name'], video_info['fullname'], video_info['video_path'], video_info['properties']['fps']
        logger.info(f'Binary classification in progress')
        cfg_path, model_path, out_path = video_info['binary_cfg_path'], self.binary_model_path, video_info['predictions_path']
        binary_scores = self._predict(cfg_path, model_path, out_path)
        pos_score = binary_scores[1]

        df = pd.DataFrame(columns=self.va_columns + ['stereotypical_score'])
        for d, score in zip(dataset, pos_score):
            s, t = d['start'], d['end']
            df.loc[df.shape[0]] = [basename, fullname, path, s / fps, t / fps, s, t, -1, pd.Timestamp.now(), self.model_name, score]
        return df

    def _model_predictions(self, video_info):
        logger.info(f'Collecting ASDMotion predictions for {video_info["name"]}')
        scores_path = video_info['scores_path']
        if osp.exists(scores_path):
            df = pd.read_csv(scores_path)
        else:
            df = self._detect_stereotypical_movements(video_info)
            df.to_csv(scores_path, index=False)
        agg = aggregate(df, self.threshold)
        agg['source'] = self.model_name
        return agg

    def conclude(self, _df, video_info):
        df = _df[_df['movement'] == 'Stereotypical'].copy()
        fps = video_info['properties']['fps']
        video_length_seconds = video_info['properties']['length']
        video_length_minute = video_length_seconds / 60
        video_frame_count = video_info['properties']['frame_count']
        valid_frames = video_info['properties']['valid_frames']
        last_valid_frame = video_info['properties']['last_valid_frame']
        df['segment_frames'] = df['end_frame'] - df['start_frame']
        df['relative_segment_frames'] = df['segment_frames'] / valid_frames
        df['segment_length_minute'] = df['segment_frames'] / (fps * 60)
        grp = df.groupby('video').agg({'segment_length_minute': 'sum', 'relative_segment_frames': 'sum', 'movement': 'count'}).reset_index()
        grp.columns = ['video', 'smm_length_minute', 'smm_proportion', 'smm_count']
        grp['fps'] = fps
        grp['video_length_minute'] = video_length_minute
        grp['video_frame_count'] = video_frame_count
        grp['valid_frames'] = valid_frames
        grp['last_valid_frame'] = last_valid_frame
        grp['smm/min'] = grp['smm_count'] / grp['video_length_minute']
        grp['assessment'] = grp['video'].apply(lambda v: '_'.join(v.split('_')[:-2]))
        return grp

    def annotate_video(self, video_info):
        df = self._model_predictions(video_info).sort_values(by=['video', 'start_time'])
        conc = self.conclude(df, video_info)
        df.to_csv(video_info['annotations_path'], index=False)
        conc.to_csv(video_info['conclusion_path'], index=False)
        return df
