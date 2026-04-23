from os import path as osp

import numpy as np

from asdmotion.child_detector.child_detector import ChildDetector
from asdmotion.logger import LogManager
from asdmotion.pipeline.holistic_pose import build_skeleton_from_holistic_json
from asdmotion.pipeline.openpose_executor import OpenposeInitializer
from asdmotion.pipeline.skeleton_layout import BODY_25_LAYOUT, COCO_LAYOUT
from asdmotion.pipeline.splitter import Splitter
from pathlib import Path

from asdmotion.utils import read_pkl, write_pkl, get_video_properties, init_directories, create_config, save_config, RESOURCES_ROOT

CFG_DIR = osp.join(RESOURCES_ROOT, 'mmaction_template')
logger = LogManager.APP_LOGGER


def _warn_if_numpy2_breaks_mmaction_pickles(dataset_path: str) -> None:
    """MMAction is spawned under ``mmlab_python_path``; NumPy 2 pickles need ``numpy._core`` (missing on NumPy 1)."""
    major = int(np.__version__.split(".", 1)[0])
    if major >= 2:
        logger.warning(
            "NumPy %s writes annotation pickles that many older MMAction/OpenMMLab stacks "
            "cannot load (ModuleNotFoundError: numpy._core). Use NumPy 1.x in this env: "
            "pip install 'numpy>=1.21,<2', then remove %s and related *_raw.pkl / per-video "
            "*.pkl caches and re-run.",
            np.__version__,
            dataset_path,
        )


class VideoTransformer:
    def __init__(
        self,
        work_dir,
        binary_model_name,
        openpose_root,
        detect_child,
        sequence_length,
        step_size,
        gpu_id,
        num_person_in,
        num_person_out,
        holistic_landmarks_json=None,
    ):
        self.default_cfgs = {
            'binary': osp.join(CFG_DIR, 'binary_cfg_template.py'),
        }
        self.work_dir = osp.join(work_dir)
        self.gpu_id = gpu_id
        # OpenPose / YOLO: only use a CUDA index when gpu is a non-negative int (OmegaConf may wrap values).
        try:
            _gid = int(gpu_id) if gpu_id is not None else None
        except (TypeError, ValueError):
            _gid = None
        _cuda_idx = _gid if _gid is not None and _gid >= 0 else None
        self.holistic_landmarks_json = None
        if holistic_landmarks_json:
            p = str(holistic_landmarks_json).strip()
            if p:
                self.holistic_landmarks_json = osp.abspath(p)
        self.initializer = OpenposeInitializer(
            sequence_length=sequence_length,
            num_person_in=num_person_in,
            num_person_out=num_person_out,
            open_pose_path=openpose_root,
            as_img_dir=True,
            gpu_id=_cuda_idx,
        )
        self.binary_model_name, self.detect_child, self.sequence_length, self.step_size = binary_model_name, detect_child, sequence_length, step_size
        if self.detect_child:
            child_dev = 'cpu' if _cuda_idx is None else f'cuda:{_cuda_idx}'
            self.child_detector = ChildDetector(device=child_dev)

    def _create_skeleton(self, video_info):
        video_path = video_info['video_path']
        skeleton_path = video_info['skeleton_path']
        raw_path = video_info['raw_skeleton_path']
        if osp.exists(skeleton_path):
            logger.info(f'Skeleton already exists: {skeleton_path}')
            skeleton = read_pkl(skeleton_path)
        else:
            # Prefer Holistic when configured so an old OpenPose *_raw.pkl is not reused by mistake.
            if self.holistic_landmarks_json and osp.isfile(self.holistic_landmarks_json):
                logger.info(
                    'Building COCO-17 skeleton from Holistic JSON (OpenPose skipped): %s',
                    self.holistic_landmarks_json,
                )
                skeleton = build_skeleton_from_holistic_json(
                    self.holistic_landmarks_json,
                    video_path=video_path,
                    resolution=video_info['properties']['resolution'],
                    fps=video_info['properties']['fps'],
                    frame_count=video_info['properties']['frame_count'],
                    name=video_info['name'],
                )
                write_pkl(skeleton, raw_path)
            elif osp.exists(raw_path):
                logger.info(f'Raw skeleton already exists: {raw_path}')
                skeleton = read_pkl(raw_path)
            else:
                logger.info(f'Initializing new skeleton via OpenPose: {skeleton_path}')
                skeleton_json = self.initializer.prepare_skeleton(video_path)
                skeleton = self.initializer.to_poseC3D(skeleton_json,
                                                       in_layout=BODY_25_LAYOUT, out_layout=COCO_LAYOUT)
                write_pkl(skeleton, raw_path)
            if self.detect_child:
                detections_path = video_info['detections_path']
                if osp.exists(detections_path):
                    logger.info(f'Detections already exists: {detections_path}')
                    detections = read_pkl(detections_path)
                else:
                    logger.info(f'Child detection in process: {video_path} , {skeleton_path}')
                    detections = self.child_detector.detect(video_path)
                    write_pkl(detections, detections_path)
                logger.info(f'Child detection - skeleton match in process: {video_path} , {skeleton_path}')
                skeleton = self.child_detector.match_skeleton(skeleton, detections, tolerance=200)
            else:
                T = video_info['properties']['frame_count']
                # Person index 0 from OpenPose for all frames when child detection is off.
                skeleton['child_ids'] = np.zeros(T, dtype=np.int64)
                skeleton['child_detected'] = np.zeros(T)
                skeleton['child_bbox'] = np.zeros((T, 4))
        cids = skeleton['child_ids']
        if np.all(cids == -1):
            raise ValueError(f'No children detected in {video_info["name"]}')
        valid_frames = cids[cids != -1].shape[0]
        last_valid_frame = int(np.where(cids != -1)[0][-1])
        video_info['properties']['valid_frames'] = valid_frames
        video_info['properties']['last_valid_frame'] = last_valid_frame
        write_pkl(skeleton, skeleton_path)
        return skeleton

    def prepare_dataset(self, video_info):
        basename = video_info['name']
        dataset_output = video_info['dataset_path']

        logger.info(f'Creating new skeleton for {basename}')
        skeleton = self._create_skeleton(video_info)
        logger.info('Writing Dataset.')
        dataset = Splitter(skeleton, sequence_length=self.sequence_length, step_size=self.step_size, min_length=self.step_size*2).collect()
        out = {
            'split': {'test1': [f'{x["frame_dir"]}' for x in dataset]},
            'annotations': dataset
        }
        _warn_if_numpy2_breaks_mmaction_pickles(dataset_output)
        write_pkl(out, dataset_output)
        logger.info('Data initialized successfully.')

    def init_cfg(self, video_info, name, ann_file, model_type):
        logger.info(f'Initializing cfg for {name}')
        with open(self.default_cfgs['binary'], encoding='utf-8') as f:
            text = f.read()
        ann_posix = Path(ann_file).expanduser().resolve().as_posix()
        work_posix = Path(video_info['jordi_dir']).expanduser().resolve().as_posix()
        text = text.replace('__ANN_FILE__', ann_posix)
        text = text.replace('__WORK_DIR__', work_posix)
        with open(video_info[f'{model_type}_cfg_path'], 'w', encoding='utf-8') as f:
            f.write(text)
    def prepare_environment(self, video_path):
        fullname = osp.basename(video_path)
        name, ext = osp.splitext(fullname)
        work_dir = osp.join(self.work_dir, name)
        jordi_dir = osp.join(work_dir, 'asdmotion')
        model_dir = osp.join(jordi_dir, self.binary_model_name)
        resolution, fps, frame_count, length = get_video_properties(video_path)
        video_info = {
            'name': name,
            'fullname': fullname,
            'video_path': video_path,
            'work_dir': work_dir,
            'jordi_dir': jordi_dir,
            'skeleton_path': osp.join(jordi_dir, f'{name}.pkl'),
            'raw_skeleton_path': osp.join(jordi_dir, f'{name}_raw.pkl'),
            'dataset_path': osp.join(model_dir, f'{name}_dataset_{self.sequence_length}.pkl'),
            'binary_cfg_path':  osp.join(model_dir, f'{name}_binary_config.py'),
            'annotations_path': osp.join(model_dir, f'{name}_annotations.csv'),
            'conclusion_path': osp.join(model_dir, f'{name}_conclusion.csv'),
            'predictions_path': osp.join(model_dir, f'{name}_predictions.pkl'),
            'scores_path': osp.join(model_dir, f'{name}_scores.csv'),
            'self_path': osp.join(model_dir, f'{name}_exec_info.yaml'),
            'properties': {
                'resolution': resolution,
                'fps': fps,
                'frame_count': frame_count,
                'length': length
            }
        }
        if self.detect_child:
            video_info['child_detect'] = True
            video_info['detections_path'] = osp.join(work_dir, f'{name}_detections.pkl')
        init_directories(work_dir, jordi_dir, model_dir)
        video_info = create_config(video_info)
        self.init_cfg(video_info, name, video_info['dataset_path'], 'binary')
        self.prepare_dataset(video_info)
        save_config(video_info, video_info['self_path'])
        return video_info
