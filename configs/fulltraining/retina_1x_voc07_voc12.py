_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/voc_schedule_1x.py',
    '../_base_/default_runtime.py'
]



model = dict(bbox_head=dict(num_classes=20))

