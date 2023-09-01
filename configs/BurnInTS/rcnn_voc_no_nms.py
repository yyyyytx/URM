_base_ = [
    '../_base_/datasets/voc_burnin_cocofmt.py',
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

model_wrapper = dict(
    type='BurnInTSModel',
    teacher="${model}",
    student="${model}",
    n_cls=20,
)

custom_hooks = [
    dict(type="TSUpdateHook", update_interval=1, burnIn_stage=0),
    dict(type="PesudoSummaryHook", log_interval=50, burnIn_stage=0),
    dict(type="DistBatchSamplerSeedHook"),
]
find_unused_parameters = True

train_cfg = dict(
    print_pesudo_summary=True,
    check_geo_trans_bboxes=False,
    pesudo_thr=0.7,
    unsup_loss_weight=0.6,

    is_no_nms=True,
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.9,
        neg_iou_thr=0.1,
        min_pos_iou=0.9,
        match_low_quality=False,
        ignore_iof_thr=-1),
)