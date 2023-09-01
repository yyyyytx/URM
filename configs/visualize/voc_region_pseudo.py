_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_region_est.py',
    '../_base_/default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))


model_wrapper = dict(
    type='NegativeLoss',
    teacher="${model}",
    student="${model}",
    n_cls=20,
)

train_cfg = dict(
    print_pesudo_summary=True,
    check_geo_trans_bboxes=False,
    pesudo_thr=0.7,
    unsup_loss_weight=2.0,

    neg_queue_len=65536,
    pos_queue_len=100,
    region_bg_iou_thr=0.3,
    region_bg_max_num=10,
    region_fg_max_num=80,
    region_bg_nms_cfg=dict(type='nms', iou_threshold=0.5),
    region_bg_score_thr=0.5,
)

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# dataset settings
dataset_type = 'CocoDataset'
img_prefix = '/home/liu/datasets/voc/VOCdevkit'
ann_prefix = '/home/liu/datasets/voc/cocofmt/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)



test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]

unsup_test=dict(type=dataset_type,
               ann_file=ann_prefix + 'voc07_trainval.json',
               img_prefix=img_prefix,
               pipeline=test_pipeline,
                classes=CLASSES)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    test=dict(
        type=dataset_type,
        ann_file=ann_prefix + 'voc07_test.json',
        img_prefix=img_prefix,
        pipeline=test_pipeline,
        classes=CLASSES)
)