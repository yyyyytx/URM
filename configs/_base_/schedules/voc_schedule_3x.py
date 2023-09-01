_base_ = [
    'schedule_1x.py',
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch 9 * 3 = 27   11 * 3 = 33
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 32])
# lr_config = dict(policy='step', step=[9, 11])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=36)