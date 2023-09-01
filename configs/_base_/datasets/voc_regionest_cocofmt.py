
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
    # dict(type='RandResize', img_scale=(1000, 600), keep_ratio=True, record=True),
    dict(
        type='RandResize',
        img_scale=[(1000, 480), (1000, 512), (1000, 544), (1000, 576),
                   (1000, 608), (1000, 640), (1000, 672), (1000, 704),
                   (1000, 736), (1000, 768), (1000, 800)],
        multiscale_mode='value',
        # img_scale=[(1333, 500), (1333, 800)],
        # keep_ratio=True,
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
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='GaussianBlur', sigma_limit=[0, 3], p=1.0),
    #         dict(type='MedianBlur',  p=1.0),
    #         dict(type='Sharpen', alpha=[0.0, 1.0],
    #              lightness=[0.75, 1.5], p=1.0),
    #         dict(type='GaussNoise', var_limit=[
    #              0.0, 0.05], per_channel=True, p=1.0),
    #         dict(type='InvertImg', p=1.0),
    #         # dict(
    #         #     type='RGBShift',
    #         #     r_shift_limit=[-10, 10],
    #         #     g_shift_limit=[-10, 10],
    #         #     b_shift_limit=[-10, 10],
    #         #     p=1.0),
    #         dict(
    #             type='MultiplicativeNoise',
    #             multiplier=[0.5, 1.5],
    #             elementwise=True,
    #             p=1.0),
    #         dict(type='ColorJitter',
    #              brightness=[0.5, 1.5],
    #              contrast=[0.5, 1.5],
    #              saturation=[0.5, 1.5],
    #              hue=0.2),
    #     ]),
    dict(type='ColorJitter', brightness=0.2,
         contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    dict(type='ToGray', p=0.2),
    dict(type='GaussianBlur', sigma_limit=(0.1, 2.0), p=0.2),
]

image_size = (1024, 1024)
strong_pipeline = [
    # dict(type='RandResize', img_scale=(1000, 600), keep_ratio=True, record=True),
    dict(
        type='RandResize',
        img_scale=[(1000, 480), (1000, 512), (1000, 544), (1000, 576),
                   (1000, 608), (1000, 640), (1000, 672), (1000, 704),
                   (1000, 736), (1000, 768), (1000, 800)],
        multiscale_mode='value',
        # img_scale=[(1333, 500), (1333, 800)],
        # keep_ratio=True,
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
    # dict(
    # type="RandErase",
    # n_iterations=(3, 5),
    # size=[0.05, 0.2],
    # squared=True,
    # ),
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

sup_center = dict(type=dataset_type,
                  ann_file=ann_prefix + 'voc07_trainval.json',
                  img_prefix=img_prefix,
                  pipeline=[
                      dict(type="LoadImageFromFile"),
                      dict(type="LoadAnnotations", with_bbox=True),
                      dict(type='Resize', img_scale=(
                          1000, 600), keep_ratio=True),
                      dict(type='Normalize', **img_norm_cfg),
                      dict(type='Pad', size_divisor=32),
                      dict(type='DefaultFormatBundle'),
                      dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
                           meta_keys=("filename",
                                      "img_shape",
                                      "img_norm_cfg",
                                      "pad_shape",
                                      "scale_factor")),
                  ],
                  classes=CLASSES
                  )


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
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
    sample_ratio=[1, 3],
    # epoch_length =
    epoch_length=2500
    # epoch_length=7500
    # epoch_length=200
)
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[36])

# lr_config = dict(policy='step', step=[9])

runner = dict(
    type='EpochBasedRunner', max_epochs=36)

# lr_config = dict(policy='step', step=[24, 32])

# runner = dict(
# type='EpochBasedRunner', max_epochs=36)
