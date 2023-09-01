checkpoint_config = dict(interval=12, max_keep_ckpts=1)
# checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
# dist_params = dict(backend='gloo')
log_level = 'INFO'
load_from = None
resume_from = None
auto_resume = True
workflow = [('train', 1)]
fp16 = dict(loss_scale=512.)
