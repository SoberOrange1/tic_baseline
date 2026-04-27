# PoseC3D binary head — **training** bundle (Group KFold ann from ``build_mmaction_groupkfold_ann.py``).
#
# Placeholders (replaced by ``build_mmaction_groupkfold_ann.py``): ``__ANN_FILE__``, ``__WORK_DIR__``,
# ``__LOAD_FROM__`` (string token; becomes ``None`` or a checkpoint path). ``times=int("1")`` → repeat count.
#
# ``HolisticTicValMetric`` lives in ``mmaction.evaluation.metrics.holistic_tic_val_metric`` so
# ``Config.fromfile`` (mmengine **lazy** import) does not execute ``@METRICS.register_module()`` inside
# this file (that raises ``RuntimeError``). ``custom_imports`` loads the module and registers the metric.
# Requires **scikit-learn** in the MMAction / open-mmlab environment.
import sys

# mmengine 0.8.x: ``custom_imports`` must be a mapping (``imports=...``), not a bare list.
custom_imports = dict(
    imports=["mmaction.evaluation.metrics.holistic_tic_val_metric"],
    allow_failed_imports=False,
)

default_scope = "mmaction"

default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20, ignore_last=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="holistic_val/f1_tic",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffers=dict(type="SyncBuffersHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(
        mp_start_method=("spawn" if sys.platform == "win32" else "fork"),
        opencv_num_threads=0,
    ),
    dist_cfg=dict(backend="gloo"),
)

log_processor = dict(type="LogProcessor", window_size=20, by_epoch=True)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)

log_level = "INFO"
resume = False

model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="ResNet3dSlowOnly",
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2,),
        stage_blocks=(3, 4, 6),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1),
    ),
    cls_head=dict(
        type="I3DHead",
        in_channels=512,
        num_classes=2,
        loss_cls=dict(type="CrossEntropyLoss", class_weight=[0.75, 0.25]),
        spatial_type="avg",
        dropout_ratio=0.5,
        average_clips="prob",
    ),
    train_cfg=None,
    test_cfg=None,
)

dataset_type = "PoseDataset"
ann_file = "__ANN_FILE__"
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type="UniformSampleFrames", clip_len=48),
    dict(type="PoseDecode"),
    dict(type="PoseCompact", hw_ratio=1.0, allow_imgpad=True),
    dict(type="Resize", scale=(-1, 64)),
    dict(type="RandomResizedCrop", area_range=(0.56, 1.0)),
    dict(type="Resize", scale=(48, 48), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type="GeneratePoseTarget",
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
    ),
    dict(type="FormatShape", input_format="NCTHW_Heatmap"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="UniformSampleFrames", clip_len=48, num_clips=1, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="PoseCompact", hw_ratio=1.0, allow_imgpad=True),
    dict(type="Resize", scale=(-1, 56)),
    dict(type="CenterCrop", crop_size=56),
    dict(
        type="GeneratePoseTarget",
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
    ),
    dict(type="FormatShape", input_format="NCTHW_Heatmap"),
    dict(type="PackActionInputs"),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=int("1"),
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            split="train1",
            pipeline=train_pipeline,
        ),
    ),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split="test1",
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split="test1",
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(type="HolisticTicValMetric")
test_evaluator = []

train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=12, val_begin=1, val_interval=1
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[9, 11],
        gamma=0.1,
    )
]

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0003),
    clip_grad=dict(max_norm=40, norm_type=2),
)

work_dir = "__WORK_DIR__"
load_from = "__LOAD_FROM__"
