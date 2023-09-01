from mmcv.utils import Registry
from mmdet.datasets import DATASETS, ConcatDataset, build_dataset

@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    '''Wrapper for semi-supervised'''

    def __init__(self, labeled: dict, unlabeled: dict, **kwargs):
        super().__init__([build_dataset(labeled), build_dataset(unlabeled)], **kwargs)

    @property
    def labeled(self):
        return self.datasets[0]

    @property
    def unlabeled(self):
        return self.datasets[1]




