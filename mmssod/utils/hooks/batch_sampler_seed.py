# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class DistBatchSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        if hasattr(runner.data_loader.batch_sampler, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.data_loader.batch_sampler.set_epoch(runner.epoch)

