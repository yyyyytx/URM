from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner import get_dist_info
import mmcv
import time
from mmssod.utils.structure_utils import dict_split


@HOOKS.register_module()
class InitialCentersHook(Hook):

    # def _split(self, imgs, img_metas, **kwargs):
    #     print(kwargs)
    def after_epoch(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model._save_center()

