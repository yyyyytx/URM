_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

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
    dict(type='RandResize', img_scale=(1000, 600), keep_ratio=True, record=True),
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

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussianBlur', sigma_limit=[0, 3], p=1.0),
            dict(type='MedianBlur',  p=1.0),
            dict(type='Sharpen', alpha=[0.0, 1.0], lightness=[0.75, 1.5], p=1.0),
            dict(type='GaussNoise', var_limit=[0.0, 0.05], per_channel=True, p=1.0),
            dict(type='InvertImg', p=1.0),
            dict(
                type='RGBShift',
                r_shift_limit=[-10, 10],
                g_shift_limit=[-10, 10],
                b_shift_limit=[-10, 10],
                p=1.0),
            dict(
                type='MultiplicativeNoise',
                multiplier=[0.5, 1.5],
                elementwise=True,
                p=1.0),
            dict(type='ColorJitter',
                 brightness=[0.5, 1.5],
                 contrast=[0.5, 1.5],
                 saturation=[0.5, 1.5],
                 hue=0.2),
        ]),
]

strong_pipeline = [
    dict(type='RandResize', img_scale=(1000, 600), keep_ratio=True, record=True),
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
    dict(
        type="RandErase",
        n_iterations=(3, 5),
        size=[0.05, 0.2],
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



sup=dict(type=dataset_type,
         ann_file=ann_prefix + 'voc07_trainval.json',
         img_prefix=img_prefix,
         pipeline=sup_pipline,
         classes=CLASSES)
unsup=dict(type=dataset_type,
           ann_file=ann_prefix + 'voc12_trainval.json',
           img_prefix=img_prefix,
           pipeline=unsup_pipline,
           classes=CLASSES)

unsup_test=dict(type=dataset_type,
               ann_file=ann_prefix + 'voc12_trainval.json',
               img_prefix=img_prefix,
               pipeline=test_pipeline,
                classes=CLASSES)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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
    sample_ratio = [1, 1],
    # epoch_length =
    epoch_length=10032

    # epoch_length=200
)
evaluation = dict(interval=2, metric='bbox')
# fp16 = dict(loss_scale=512.)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[9])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=12)  # actual epoch = 4 * 3 = 12