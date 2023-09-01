from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n
import torchvision

@HOOKS.register_module()
class DebugHook(Hook):

    def before_epoch(self, runner):
        cur_epoch = runner.epoch
        print('epoch:', cur_epoch , ' start')

    def after_epoch(self, runner):
        cur_epoch = runner.epoch
        print('epoch:', cur_epoch , ' end')

