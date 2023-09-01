_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_region_est.py',
    '../_base_/datasets/voc_regionest_cocofmt.py',
    '../_base_/default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

model_wrapper = dict(
    type='RegionEstimation',
    teacher="${model}",
    student="${model}",
    n_cls=20,
)

custom_hooks = [
    dict(type="TSUpdateHook", update_interval=1, burnIn_stage=0),
    dict(type="PesudoSummaryHook", log_interval=100, burnIn_stage=0),
    dict(type="DistBatchSamplerSeedHook"),
]
find_unused_parameters = True

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

    unreliable=False,
)