from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n
import torchvision

@HOOKS.register_module()
class TSUpdateHook(Hook):
    def __init__(self, update_interval=1, burnIn_stage=4, ema_decay_factor=0.1, ema_decay_interval=None, **kwargs):
        self.update_interval = update_interval
        self.burnIn_stage = burnIn_stage
        self.is_semi_train = False
        self.ema_decay_interval = ema_decay_interval
        self.ema_decay_factor = ema_decay_factor

    def before_run(self, runner):
        cur_epoch = runner.epoch
        if self.is_semi_train == False:
            if (cur_epoch >= self.burnIn_stage):
                self.is_semi_train = True
                model = runner.model
                if is_module_wrapper(model):
                    model = model.module
                model.switch_semi_train()

    def before_epoch(self, runner):
        cur_epoch = runner.epoch
        print("cur epoch:", cur_epoch)
        if self.is_semi_train == False:
            if (cur_epoch == self.burnIn_stage):
                model = runner.model
                if is_module_wrapper(model):
                    model = model.module
                model.start_semi_train()
            if (cur_epoch >= self.burnIn_stage):
                self.is_semi_train = True


    def after_train_iter(self, runner):
        if self.is_semi_train is True:
            model = runner.model
            if is_module_wrapper(model):
                model = model.module
            if self.every_n_iters(runner, self.update_interval):
                model.momentum_update()
                self.update_ema(runner)


    def update_ema(self, runner):
        curr_step = runner.iter
        if self.ema_decay_interval is None:
            return
        if curr_step == self.ema_decay_interval:
            model = runner.model
            if is_module_wrapper(model):
                model = model.module
            model.momentum = 1 - model.momentum * self.ema_decay_factor

            print('ema decay:', model.momentum)
