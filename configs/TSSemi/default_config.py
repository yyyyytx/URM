
# checkpoint_config = dict(interval=4000, max_keep_ckpts=3)
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
# fp16 = dict(loss_scale="dynamic", cumulative_iters=4)
p16 = dict(loss_scale="dynamic")
