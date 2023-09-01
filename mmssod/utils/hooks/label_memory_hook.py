from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n
import torchvision

@HOOKS.register_module()
class LabelMemoryHook(Hook):
    def __init__(self, dataloader):
        self.loader = dataloader
        self.name_list = []



