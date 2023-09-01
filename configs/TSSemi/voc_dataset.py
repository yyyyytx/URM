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
        img_scale=[(1000, 300), (1000, 600)],

        multiscale_mode="range",
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
        multiscale_mode="range",

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
        type='RandResize',
        img_scale=[(1000, 300), (1000, 900)],
        multiscale_mode="range",

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
    samples_per_gpu=5,
    workers_per_gpu=5,
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
    sample_ratio=[1, 4],
    epoch_length=2500
)
