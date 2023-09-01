# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/liu/datasets/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


weak_pipeline = [
    dict(type='RandResize', img_scale=(1333, 800), keep_ratio=True, record=True),
    dict(type='RandFlip', flip_ratio=0.5, record=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels'],
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

strong_pipeline = [
    dict(type='RandResize', img_scale=(1333, 800), keep_ratio=True, record=True),
    dict(type='RandFlip', flip_ratio=0.5, record=True),
    dict(
        type='RandColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
        p=0.8
    ),
    dict(type='RandGrayscale', p=0.2),
    dict(type='RandGaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(
        type="RandErase",
        n_iterations=(1, 5),
        size=[0, 0.2],
        squared=True,
    ),

    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
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

sup_pipline=[
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    # dict(type="PseudoSamples", with_bbox=True),
    dict(type="ExtraAttrs", tag="sup"),
    dict(
        type="MultiBranch", strong=strong_pipeline, weak=weak_pipeline
    ),
]

unsup_pipline=[
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
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
        img_scale=(1333, 800),
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



sup=dict(type=dataset_type,
         ann_file=data_root + 'annotations/semi_supervised/instances_train2017.1@10.json',
         img_prefix=data_root + 'train2017/',
         pipeline=sup_pipline)
unsup=dict(type=dataset_type,
           ann_file=data_root + 'annotations/semi_supervised/instances_train2017.1@10-unlabeled.json',
           img_prefix=data_root + 'train2017/',
           pipeline=unsup_pipline)

unsup_test=dict(type=dataset_type,
               ann_file=data_root + 'annotations/semi_supervised/instances_train2017.1@10-unlabeled.json',
               img_prefix=data_root + 'train2017/',
               pipeline=test_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='SemiDataset',
        labeled=sup,
        unlabeled=unsup
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    sample_ratio = [1, 1],
    # epoch_length =
    # 106458 / 2
    epoch_length=10000
    # epoch_length=200
)
evaluation = dict(interval=2, metric='bbox')
