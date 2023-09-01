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
    dict(type="PesudoSummaryHook", log_interval=100, burnIn_stage=0),
    dict(type="DistBatchSamplerSeedHook"),
]
find_unused_parameters = True

train_cfg = dict(
    print_pesudo_summary=False,
    pesudo_summary_iou_thrs=[0.25, 0.5, 0.75],
    unsup_sample_measure_iou_thrs=[0.5],
    check_geo_trans_bboxes=True,
    pesudo_thr=0.7,
    unsup_loss_weight=2.0,
    rpn_filter_with_thr=False,
    filter_unsup_regions=False,
    filter_unsup_positive=False,
    filter_unsup_negative=False,
)