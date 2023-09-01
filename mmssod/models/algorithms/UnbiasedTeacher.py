from mmdet.models import DETECTORS
from .BurnInTS import BurnInTSModel
from ...utils.structure_utils import weighted_loss

@DETECTORS.register_module()
class UnbiasedTSModel(BurnInTSModel):
    '''Base arch for teacher-student model with burn-in stage'''

    def __init__(self, teacher: dict, student: dict, train_cfg=None, test_cfg=None):
        super().__init__(teacher, student, train_cfg, test_cfg)
        print('building Unbiased Model')


    def _compute_unsup_loss(self, sup_data, unsup_data):
        strong_unsup = self._gen_pseudo_labels(unsup_data)

        losses = dict()
        sup_loss = self.student.forward_train(**sup_data)
        sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
        losses.update(sup_loss)

        unsup_loss = self.student.forward_train(**strong_unsup)
        unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
        unsup_loss.pop("unsup_loss_rpn_bbox")
        unsup_loss.pop("unsup_loss_bbox")
        losses.update(unsup_loss)

        # for k, v in losses.items():
        #     if k == "unsup_loss_rpn_bbox" or k == "unsup_loss_bbox":
        #         losses[k] = v * 0.0

        losses = weighted_loss(losses, self.unsup_loss_weight)

        return losses
