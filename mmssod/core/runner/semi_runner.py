# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
Support save ema model
"""
import os.path as osp
import platform
import shutil

import mmcv
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint

from mmcv.parallel import is_module_wrapper
from mmcv.runner import IterBasedRunner, EpochBasedRunner
import time


@RUNNERS.register_module()
class SemiIterBasedRunner(IterBasedRunner):
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        self.call_hook('before_train_iter')
        data_batch = next(data_loader)
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1


