<p align="center">
  <img src="/resources/logo_jordi.webp" alt="ASDMotion" width="500"/>
</p>


# ASDMotion

1. [Abstract](#abstract)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Citation](#citation)


## Abstract
This repository holds the codebase, dataset and models for the paper:

**Automated Analysis of Stereotypical Movements in Videos of Children With Autism Spectrum Disorder**, Barami Tal, Manelis-Baram Liora, Kaiser Hadas, Dinstein Ilan; 2024. [JAMA Open](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2823635)

Stereotypical motor movements (SMMs) are a form of restricted and repetitive behaviors (RRBs) that are prevalent in individuals with Autism Spectrum Disorders (ASD). Previous studies attempting to quantify SMMs in ASD have relied on coarse and subjective reports or on manual annotation of video recordings. Here, we demonstrate the utility of a novel open-source AI algorithm that can analyze video recordings of children and automatically identify segments containing stereotypical movements.


## Requirements
This code was tested using:
1. Windows 10/11 or Linux (GPU servers are supported)
2. Python 3.9–3.10
3. PyTorch 1.13+ (CPU wheels by default in `requirements.txt`; use CUDA wheels on GPU hosts)
4. CUDA 11.7 (historical); CUDA **12.x** driver with **cu121** PyTorch wheels is supported on cloud Linux

Other OS/Python distributions are expected to work.

### GPU inference (Linux, CUDA 12.2 driver)

Inference is slow when the **MMAction2** interpreter (`mmlab_python_path`) uses a **CPU-only** PyTorch build. Use a **GPU** build there and keep `gpu: 0` (or `-gpu 0`) in config / CLI. Use **`gpu: -1`** or **`-gpu -1`** to force CPU for the MMAction subprocess.

Use **two** conda environments: one for this repo (`asdmotion`) and one for OpenMMLab / MMAction2 (`mmlab`), both with matching **NumPy 1.x** (see `requirements.txt`) so annotation pickles load correctly.

**1) `asdmotion` environment (preprocess, Holistic / OpenPose orchestration)**

```bash
conda create -n asdmotion python=3.10 -y
conda activate asdmotion
cd /path/to/tic_holistic/baseline
pip install -r requirements.txt
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

**2) `mmlab` environment (MMAction2 `tools/test.py`)**

Install PyTorch with CUDA first, then MMEngine / MMCV built for that CUDA, then install the **same** MMAction2 tree you point to with `mmaction_path` (this repo ships one under `baseline/mmaction2`).

```bash
conda create -n mmlab python=3.10 -y
conda activate mmlab
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install "mmengine>=0.10.0,<1.1"
# MMCV must match torch + CUDA; for torch 2.1 + cu121:
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
# If that index URL fails, pick **torch + CUDA** from: https://mmcv.readthedocs.io/en/latest/get_started/installation.html
cd /path/to/tic_holistic/baseline/mmaction2
pip install -r requirements/build.txt
pip install -v -e .
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
```

If `torch.cuda.is_available()` is `False`, the PyTorch / driver pairing is wrong (re-check the `cu121` install and `nvidia-smi`).

**3) Configuration on the server**

In `resources/configs/config.yaml` (or your copy), set Linux paths, for example:

- `mmaction_path`: absolute path to `baseline/mmaction2`
- `mmlab_python_path`: absolute path to `mmlab` env’s `python` (e.g. `/home/ubuntu/miniconda3/envs/mmlab/bin/python`)
- `open_pose_path`: only needed if you do **not** use Holistic JSON; with `--holistic-json`, OpenPose is skipped
- `gpu`: `0` for the first GPU (or another physical index)

Then run:

```bash
conda activate asdmotion
python -m asdmotion.detector.executor -cfg resources/configs/config.yaml -video /path/to/video.mp4 -out /path/to/results --holistic-json /path/to/landmarks.json -gpu 0
```

Re-run once after pulling this repo so regenerated `*_binary_config.py` picks up `cudnn_benchmark=True` in the template (delete old `*_binary_config.py` under `results/.../asdmotion/...` if you want a clean regen).

### Minimal MMAction2 for inference (no full repo upload)

You do **not** need `docs/`, `demo/`, `tests/`, `tools/data/`, or the full `tools/` tree on the server. ASDMotion only shells out to **`tools/test.py`** with a **generated** `*_binary_config.py` (not the upstream `configs/` tree).

**Option A — recommended:** shallow clone on the server (no huge `.git` history if you use depth 1):

```bash
git clone --depth 1 https://github.com/open-mmlab/mmaction2.git
cd mmaction2 && pip install -r requirements/build.txt && pip install -v -e .
```

Use the same **commit / major.minor** as your local fork if you rely on API quirks.

**Option B — smallest tarball from your existing tree:** run the exporter (drops dataset-prep scripts and almost all of `tools/`):

```bash
cd baseline
python scripts/export_mmaction2_minimal.py --dst ./mmaction2_minimal
# upload ./mmaction2_minimal, then on server:
cd mmaction2_minimal && pip install -r requirements/build.txt && pip install -v -e .
```

Point `mmaction_path` at that directory. The script keeps the full **`mmaction/`** Python package (registry imports need it); that is the bulk of the code and is required.

**Option C:** `pip install mmaction2` from PyPI in the `mmlab` env, then set `mmaction_path` to the site-packages path where `tools/test.py` lives (fragile across versions; prefer A or B).

## Installation
### Prepare new environment:
```console
> conda create -n asdmotion python==3.9
```
### Install required packages:
```console
> pip install -r requirements.txt
```

### Install OpenPose:
This repository utilizes OpenPose to extract the skeletal representation of individuals per video frame.
The OpenPose demo version is sufficient for this task. For installation, [follow the instructions.](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md)

### Install MMAction2:
The SMM identification model is based on PoseC3D, a skeleton-based action recognition model. 
For our setup, it is sufficient to clone the repository and install its dependencies on a **separate** environment.
[Follow the instructions (Forked repository)](https://github.com/TalBarami/mmaction2)


### Install the child detector (Optional)
The child detector allows more accurate detections when the video contains multiple adults and one child.
```console
> git clone https://github.com/TalBarami/Child-Detector.git
> cd Child-Detector
> python setup.py develop
```

### Downloads

[ASDMotion Dataset (No annotations).](https://drive.google.com/file/d/1MiNIhlf4mL-vRW1ub2TP3nCYzfMW0bYt/view?usp=drive_link)

[Annotated Dataset for Training.](https://drive.google.com/file/d/13t1tO4ZxTKmQG-w6fTy3hHQy8gX1bopl/view?usp=sharing)

[Checkpoint weights for inference.](https://drive.google.com/file/d/1PuPXu6pfBYjz0G6NvWOEUQ_RvedvinAE/view?usp=drive_link)

## Training
The entire training pipeline is managed through a forked repository of [MMAction2](https://github.com/TalBarami/mmaction2/tree/master/configs/skeleton/posec3d).
To train ASDPose, you need to follow the MMAction2 installation process. Once MMAction2 is installed, you can initiate the training by executing the following command:
```console
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example:
```console
python tools/train.py configs/skeleton/posec3d/asdmotion.py \
    --work-dir work_dirs/asdmotion \
    --validate --seed 0 --deterministic
```

For your convenience, we provide an example configuration file within the MMAction2 repository, which can be found at `/configs/skeleton/posec3d/asdmotion.py`. 
This configuration file is tailored specifically for training ASDPose and includes all the necessary parameters and settings to get you started efficiently.

## Inference
You can test the model on a pre-defined skeleton dataset using the [MMAction2](https://github.com/TalBarami/mmaction2/tree/master/configs/skeleton/posec3d) repository. To do this, execute the following command:

```console
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

In addition to testing, we provide tools for extracting a skeletal representation from a video and creating a dataset file. This dataset corresponds to a sliding window that iterates over the video and classifies each segment as either containing a stereotypical motor movement (SMM) or not.

We offer both graphical user interface (GUI) and command-line interface (CMD) applications for this task.

To use the GUI, execute:
```console
> python src/asdmotion/app/main_app.py
```
Ensure that the configuration file located at `/resources/configs/config.yaml` contains the correct paths to OpenPose and MMAction2.

To use the CMD application, install the package in editable mode once from the repo root, then run the CLI module:
```console
> pip install -e .
> python -m asdmotion.detector.executor -cfg "<path_to_config_file>" -video "<path_to_video_file>" -out "<path_to_outputs_directory>"
```
Alternatively, set `PYTHONPATH` to the `src` folder (e.g. `set PYTHONPATH=src` on Windows) if you do not use editable install.

### MediaPipe Holistic landmarks (no OpenPose)

If you already have a tic_holistic-style ``*_landmarks.json`` with ``pose_landmarks`` per frame, you can skip OpenPose: set ``holistic_landmarks_json`` in ``resources/configs/config.yaml`` to that file path, or pass ``--holistic-json`` (or ``--holistic_landmarks_json``) on the CLI (CLI overrides YAML). Do **not** use ``-holistic-json``: a single-dash flag starting with ``-h`` is parsed as ``--help``. Pose is mapped **BlazePose 33 → COCO 17** in pixel space using the probe resolution of ``-video``. OpenPose is not invoked while this path is set (``open_pose_path`` is still required in YAML for compatibility but unused for skeleton extraction).

```console
> python -m asdmotion.detector.executor -cfg "<path_to_config_file>" -video "<path_to_video_file>" -out "<path_to_outputs_directory>" --holistic-json "<path_to_session_landmarks.json>"
```

### Configuration File:
Each execution of ASDMotion relies on a set of customizable configurations, which can be specified as follows:

```yaml
sequence_length: Length of each sequence to be predicted by PoseC3D. Default is 200.
step_size: Step size of the sliding window that processes the entire video. Default is 30.
model_name: Name of the model inside the resources/models directory. Default is 'asdmotion'.
classification_threshold: Threshold to classify an action as either SMM or not. Default is 0.85.
child_detection: Utilizes the YOLOv5 child detection module to detect the child in each video frame. Default is true.
num_person_in: Maximum number of people in each video frame. Default is 5.
num_person_out: Maximum number of people in each skeleton sequence. Default is 5.
open_pose_path: Path to the OpenPose root directory.
holistic_landmarks_json: Optional path to Holistic ``*_landmarks.json``; when set, OpenPose is skipped (see above).
mmaction_path: Path to the MMAction2 root directory.
mmlab_python_path: Path to the OpenMMLab Python executable.
```

If you are using this repository for the first time, ensure to update the configuration file with the appropriate paths and settings.

### Outputs:

Upon execution, a directory named after the input video will be created. Inside this directory, you will find the following structure:

```yaml
├── asdmotion
│   ├── asdmotion.pth
│   │   ├──  <video_name>_annotations.csv - A table with start time, end time, movement type, and stereotypical score of each segment.
│   │   ├──  <video_name>_conclusion.csv - Summarizes the annotations table with the total length of SMMs, the proportion of SMMs, the number of SMM segments, and the number of SMMs per minute.
│   │   ├──  <video_name>_exec_info.yaml - Configuration file containing execution information.
│   │   ├──  <video_name>_binary_config.py - Configuration file used to execute PoseC3D.
│   │   ├──  <video_name>_predictions.pkl & <video_name>_scores.pkl - Per-sequence scores produced by PoseC3D for each sequence of <sequence_length> length while iterating over the entire video with step size <step_size>.
│   │   └──  <video_name>_dataset_<sequence_length>.pkl - Skeleton sequences that were fed to PoseC3D.
│   ├── <video_name>raw.pkl - The skeleton sequence produced by OpenPose.
│   └── <video_name>.pkl - The skeleton sequence after the matching process with the child detection module.
└── <video_name>_detections.pkl - Child detection outputs produced by the child detection module (optional).
```

An example of a video segment where SMM is observed, along with the signal produced by the model:
<p align="center">
  <img src="/resources/sample.gif" alt="Example" width="500"/>
</p>

## Citation
If you find this project useful in your research, please consider citing:
```BibTeX
@article {Barami2024.03.02.582828,
	author = {Tal Barami and Liora Manelis-Baram and Hadas Kaiser and Michal Ilan and Aviv Slobodkin and Ofri Hadashi and Dor Hadad and Danel Waissengreen and Tanya Nitzan and Idan Menashe and Analya Michaelovsky and Michal Begin and Ditza A. Zachor and Yair Sadaka and Judah Koler and Dikla Zagdon and Gal Meiri and Omri Azencot and Andrei Sharf and Ilan Dinstein},
	title = {Automated identification and quantification of stereotypical movements from video recordings of children with ASD},
	elocation-id = {2024.03.02.582828},
	year = {2024},
	doi = {10.1101/2024.03.02.582828},
	URL = {https://www.biorxiv.org/content/early/2024/03/06/2024.03.02.582828},
	eprint = {https://www.biorxiv.org/content/early/2024/03/06/2024.03.02.582828.full.pdf},
	journal = {bioRxiv}
}
```
