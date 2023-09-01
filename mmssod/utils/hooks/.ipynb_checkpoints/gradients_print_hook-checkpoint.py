from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n
import torchvision

@HOOKS.register_module()
class GradientsPrintHook(Hook):

    def after_train_iter(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        # runner.logger.info("teacher gradients:")
        # for name, parms in model.teacher.named_parameters():
        #     logger_str = '-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        #           ' -->grad_value:', parms.grad
        #     runner.logger.info(logger_str)

        runner.logger.info("student gradients:")
        for name, parms in model.student.named_parameters():
            logger_str = '-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                         ' -->grad_value:', parms.grad
            runner.logger.info(logger_str)
