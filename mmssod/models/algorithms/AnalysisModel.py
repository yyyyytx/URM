import torch
from mmdet.models import DETECTORS
from .BurnInTS import BurnInTSModel
from multiprocessing import Pool
from mmssod.utils.structure_utils import dict_split

@DETECTORS.register_module()
class AnalysisModel(BurnInTSModel):
    def __init__(self, teacher: dict, student: dict, train_cfg=None, test_cfg=None, n_cls=20):
        super().__init__(teacher, student, train_cfg, test_cfg)
        print("train_cfg:",train_cfg)

