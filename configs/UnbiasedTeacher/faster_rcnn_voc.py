_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc_UnbiasedTeacher.py',
    '../_base_/schedules/voc_schedule_20e.py',
    '../_base_/default_runtime.py'
]


model = dict(roi_head=dict(bbox_head=dict(num_classes=20,
                                          loss_cls=dict(type='FocalLoss',
                                                        use_sigmoid=True,
                                                        gamma=2.0,
                                                        alpha=0.25,
                                                        loss_weight=1.0))))

model_wrapper = dict(
    type='UnbiasedTSModel',
    teacher="${model}",
    student="${model}",
)

custom_hooks = [
    dict(type="TSUpdateHook", update_interval=1, burnIn_stage=6),
    dict(type="PesudoSummaryHook", log_interval=200, burnIn_stage=6),
    dict(type="DistBatchSamplerSeedHook"),
]
find_unused_parameters = True

train_cfg = dict(
    print_pesudo_summary=True,
    pesudo_summary_iou_thrs=[0.25, 0.5, 0.75],
    check_geo_trans_bboxes=False,
    pesudo_thr=0.7,
    unsup_loss_weight=2.0,
    rpn_filter_with_thr=True,
)