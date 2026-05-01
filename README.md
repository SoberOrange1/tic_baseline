# ASDMotion (`baseline`)

This tree implements **ASDMotion** (Barami *et al.*, 2024, [JAMA Network Open](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2823635); [bioRxiv](https://doi.org/10.1101/2024.03.02.582828)) for skeleton-based stereotypical movement detection, with PoseC3D / training aligned to [TalBarami/mmaction2](https://github.com/TalBarami/mmaction2). The `baseline` package under **tic_holistic** adds inference wiring and integrations such as MediaPipe Holistic JSON.

## Quick Run Guide (Inference + Training)

This section keeps the current project workflow and command style, but uses relative paths.

Assume your current directory is:

```bash
cd baseline
```

## Configuration (what to edit)

There are two config layers in this baseline workflow:

1. **ASDMotion executor config**: `resources/configs/config.yaml`
2. **MMAction training config**: generated per fold (`results/kfold_out/foldXX/binary_train_config.py`)

### A) `resources/configs/config.yaml` (executor / inference)

Main keys you usually need to adjust:

- `sequence_length`, `step_size`: window length/stride for ASDMotion dataset generation.
- `classification_threshold`: threshold for binary prediction at inference time.
- `holistic_landmarks_json`: optional default holistic JSON path for single-video inference
  (CLI `--holistic-json` overrides this).
- `open_pose_path`, `mmaction_path`, `mmlab_python_path`: environment/tool paths.
- `gpu`: GPU index for subprocesses (`-1` to force CPU path where supported).

Recommended practice:

- Keep this file stable for environment defaults (paths, GPU).
- Override per-run values via CLI whenever possible (`--holistic-json`, `--videos-root`, `-out`, etc.).

### B) MMAction train config (`binary_train_config.py`)

These files are generated in Step 2 for each fold. For repeatable changes, edit the template first:

- Template file: `resources/mmaction_template/binary_train_kfold_template.py`

Common training knobs in the template:

- `train_cfg.max_epochs`
- `optim_wrapper.optimizer.lr`, `momentum`, `weight_decay`
- `train_dataloader.batch_size`, `num_workers`
- `param_scheduler` milestones / gamma
- `model.cls_head.loss_cls.class_weight`
- `default_hooks.checkpoint` (`interval`, `save_best`)
- `val_evaluator` metric type

After you modify the template, re-run:

- `scripts/build_mmaction_groupkfold_ann.py`

to regenerate all fold configs consistently.

Quick tip:

- If you only hot-fix one fold once, you can edit that fold’s `binary_train_config.py` directly.
- If you want consistent behavior for all folds and future runs, edit the template and regenerate.

---

## 1) Single-video inference and statistics

### Step 1: Run ASDMotion detector/inference

```bash
python -m asdmotion.detector.executor \
  -cfg resources/configs/config.yaml \
  -video ../videos_compressed/GN-002/GN_002_V2_20251105104429.mp4 \
  -out results \
  --holistic-json ../DATA/output/GN-002/GN_002_V2_20251105104429/GN-002_GN_002_V2_20251105104429_landmarks.json
```

This writes prediction outputs under:

- `results/<video_stem>/asdmotion/asdmotion.pth/`

### Step 2: Evaluate predictions against Excel labels

```bash
python scripts/evaluate_predictions.py \
  --predictions-pkl results/GN_002_V2_20251105104429/asdmotion/asdmotion.pth/GN_002_V2_20251105104429_predictions.pkl \
  --dataset-pkl results/GN_002_V2_20251105104429/asdmotion/asdmotion.pth/GN_002_V2_20251105104429_dataset_200.pkl \
  --annotation-xlsx ../DATA/annotation/tic_annotation_english.xlsx \
  --annotation-fps 30 \
  --video-stem GN_002_V2
```

---

## 2) Baseline training (MMAction2, K-fold)

The full training flow has 3 stages:

1. Build ASDMotion dataset assets from videos (skip per-video inference outputs)
2. Build MMAction K-fold annotations/configs
3. Train each fold in MMAction2

### Step 1: Build ASDMotion preprocessing outputs for all videos

```bash
python -m asdmotion.detector.executor \
  -cfg resources/configs/config.yaml \
  -out results/kfold_out \
  --videos-root ../videos_compressed \
  --holistic-output-root ../DATA/output \
  --skip-inference
```

### Step 2: Build fold annotations and per-fold MMAction configs

```bash
python scripts/build_mmaction_groupkfold_ann.py \
  --annotation-xlsx ../DATA/annotation/tic_annotation_english.xlsx \
  --annotation-fps 30 \
  --asdmotion-out-dir results/kfold_out \
  --out-dir results/kfold_out \
  --save-labeled-cache \
  --holistic-cv-splits-json ../results/BLB_16/holistic_cv_splits_for_baseline.json
```

After this, each fold directory contains generated files such as:

- `results/kfold_out/fold00/binary_train_config.py`
- `results/kfold_out/fold01/binary_train_config.py`
- ...

### Step 3: Train each fold in MMAction2

Activate your MMAction environment first (example):

```bash
conda activate open-mmlab
```

Then run training fold by fold:

```bash
cd mmaction2
python tools/train.py ../results/kfold_out/fold00/binary_train_config.py
python tools/train.py ../results/kfold_out/fold01/binary_train_config.py
python tools/train.py ../results/kfold_out/fold02/binary_train_config.py
python tools/train.py ../results/kfold_out/fold03/binary_train_config.py
```

---

## Notes

- Keep `annotation_fps` consistent with how frame indices were produced (commonly 30).
- If you use holistic-aligned folds, always pass `--holistic-cv-splits-json` in Step 2.
- Training/evaluation paths in generated configs are relative to how you launch commands; use absolute paths if your runtime environment changes frequently.
