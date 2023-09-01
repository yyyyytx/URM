_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn_ori.py',
    '../../_base_/default_runtime.py'
]
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

# dataset settings
dataset_type = 'CocoDataset'
img_prefix = '/home/liu/datasets/voc/VOCdevkit'
ann_prefix = '/home/liu/datasets/voc/cocofmt/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_prefix + 'voc07_trainval.json',
            img_prefix=img_prefix,
            pipeline=train_pipeline,
            classes=CLASSES)),
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
        classes=CLASSES))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='TSEpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12
