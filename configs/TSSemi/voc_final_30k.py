_base_ = [
    'default_config.py'
]


model_wrapper = dict(
    type='NegativeLoss',
    teacher="${model}",
    student="${model}",
    n_cls=20,
)

custom_hooks = [
    dict(type="TSUpdateHook", update_interval=1, burnIn_stage=0),
    dict(type="PesudoSummaryHook", log_interval=200, burnIn_stage=0),
    dict(type="DistBatchSamplerSeedHook"),
    # dict(type="GradientsPrintHook")
]
find_unused_parameters = True

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[36])

# lr_config = dict(policy='step', step=[9])

# runner = dict(
#     type='EpochBasedRunner', max_epochs=12)
runner = dict(type="IterBasedRunner", max_iters=30000)
evaluation = dict(interval=5000, metric='bbox')

train_cfg = dict(
    print_pesudo_summary=True,
    check_geo_trans_bboxes=False,
    is_neg_loss=True,
    is_iou_loss=True,
    is_recall=True,
    is_ignore_ubreliable=True,
    pesudo_thr=0.9,
    unreliable_thr=0.7,

    unsup_loss_weight=0.5,
    # contrast_loss_weight=0.5
    feat_dim=128,

    theta1=0.1,
    theta2=0.7,
    alpha=0.9,
    topk=2,
    neg_center=0.001,

    region_bg_iou_thr=0.3,
    region_bg_max_num=10,
    region_fg_max_num=80,
    region_bg_nms_cfg=dict(type='nms', iou_threshold=0.5),
    region_bg_score_thr=0.5,
)

# -------------------------dataset------------------------
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# dataset settings
dataset_type = 'CocoDataset'
img_prefix = '/home/liu/datasets/voc/VOCdevkit'
ann_prefix = '/home/liu/datasets/voc/cocofmt/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


weak_pipeline = [
    dict(
        type='RandResize',
        img_scale=[(1000, 300), (1000, 900)],
        keep_ratio=True,
        record=True),
    dict(type='RandFlip', flip_ratio=0.5, record=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='ToDataContainer', fields=[dict(key='ori_bboxes')]),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', "ori_bboxes"],
         meta_keys=(
             "filename",
             "ori_shape",
             "img_shape",
             "img_norm_cfg",
             "pad_shape",
             "scale_factor",
             "tag",
             "transform_matrix",
         ),),
]


albu_train_transforms = [
    dict(type='ColorJitter', brightness=0.2,
         contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    dict(type='ToGray', p=0.2),
    dict(type='GaussianBlur', sigma_limit=(0.1, 2.0), p=0.2),
]

strong_pipeline = [
    dict(
        type='RandResize',
        img_scale=[(1000, 300), (1000, 900)],
        keep_ratio=True,
        record=True),
    dict(type='RandFlip', flip_ratio=0.5, record=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomErasing', p=0.7, scale=(
        0.05, 0.2), ratio=(0.3, 3.3), value="random"),
    dict(type='RandomErasing', p=0.5, scale=(
        0.02, 0.2), ratio=(0.1, 6), value="random"),
    dict(type='RandomErasing', p=0.3, scale=(
        0.02, 0.2), ratio=(0.05, 8), value="random"),

    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type='ToDataContainer', fields=[dict(key='ori_bboxes')]),

    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "ori_bboxes"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]

sup_pipline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadOriAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    # dict(type="PseudoSamples", with_bbox=True),
    dict(type="ExtraAttrs", tag="sup"),
    dict(
        type="MultiBranch", strong=strong_pipeline, weak=weak_pipeline
    ),
]


unsup_pipline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadOriAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    # dict(type="PseudoSamples", with_bbox=True),
    dict(type="ExtraAttrs", tag="unsup"),
    dict(
        type="MultiBranch", strong=strong_pipeline, weak=weak_pipeline
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

sup = dict(type=dataset_type,
           ann_file=ann_prefix + 'voc07_trainval.json',
           img_prefix=img_prefix,
           pipeline=sup_pipline,
           classes=CLASSES)
unsup = dict(type=dataset_type,
             ann_file=ann_prefix + 'voc12_trainval.json',
             img_prefix=img_prefix,
             pipeline=unsup_pipline,
             classes=CLASSES)

unsup_test = dict(type=dataset_type,
                  ann_file=ann_prefix + 'voc12_trainval.json',
                  img_prefix=img_prefix,
                  pipeline=test_pipeline,
                  classes=CLASSES)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type='SemiDataset',
        labeled=sup,
        unlabeled=unsup
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_prefix + 'voc07_test.json',
        img_prefix=img_prefix,
        pipeline=test_pipeline,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=ann_prefix + 'voc07_test.json',
        img_prefix=img_prefix,
        pipeline=test_pipeline,
        classes=CLASSES),
    sample_ratio=[1, 2],
    # epoch_length=2500,
    epoch_length=2500,
)


# -------------------------model------------------------
model = dict(
    type='PretrainFasterRCNN',
    init_cfg=dict(type='Pretrained',
                  checkpoint='/home/liu/ytx/SS-OD/voc_iou_pretrained.pth'),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='RoiFeaturesHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCFeatBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=0.7),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=0.5),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.7, min_score=0.05),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
