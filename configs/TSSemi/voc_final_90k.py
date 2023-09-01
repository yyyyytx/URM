_base_ = [
    'voc_dataset.py',
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
    dict(type="PesudoSummaryHook", log_interval=50, burnIn_stage=0),
    # dict(type="DistBatchSamplerSeedHook"),
    # dict(type="GradientsPrintHook")
]
# find_unused_parameters = True

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[90000])

# lr_config = dict(policy='step', step=[9])

runner = dict(type="IterBasedRunner", max_iters=90000)
checkpoint_config = dict(by_epoch=False, interval=5000,
                         max_keep_ckpts=1, create_symlink=False)
evaluation = dict(interval=5000, metric='bbox')
train_cfg = dict(
    print_pesudo_summary=False,
    check_geo_trans_bboxes=False,
    is_neg_loss=True,
    is_iou_loss=True,
    is_recall=True,
    pesudo_thr=0.9,
    unreliable_thr=0.3,
    is_ignore_ubreliable=False,

    unsup_loss_weight=1.0,

    theta1=0.2,
    theta2=0.8,
    alpha=0.8,
    topk=2,
    neg_center=0.001,
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
