_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_coco_10.py',
    '../_base_/datasets/coco_10_burnin.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]


model = dict(roi_head=dict(bbox_head=dict(num_classes=80)))

model_wrapper = dict(
    type='BurnInTSModel',
    teacher="${model}",
    student="${model}"
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
    unsup_loss_weight=1.0,
    rpn_filter_with_thr=False,
    filter_unsup_regions=False,
    filter_unsup_positive=False,
    filter_unsup_negative=False,
    n_cls=80
)