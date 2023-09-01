from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
import torchvision

@HOOKS.register_module()
class PesudoSummaryHook(Hook):
    def __init__(self, log_interval=200, burnIn_stage=4, **kwargs):
        self.log_interval = log_interval
        self.is_log_pesudo_summary = False
        self.burnIn_stage = burnIn_stage

    def before_epoch(self, runner):
        if (runner.epoch >= self.burnIn_stage):
            self.is_log_pesudo_summary = True

    def after_train_iter(self, runner):
        if self.is_log_pesudo_summary == True:
            if self.every_n_iters(runner, self.log_interval):
                model = runner.model
                if is_module_wrapper(model):
                    model = model.module
                model.log_recall_precisions(epoch_num=runner.epoch, iter_num=runner.iter)
