from .RegionEstimation import RegionEstimation
import torch
from mmdet.models import DETECTORS
from .BurnInTS import BurnInTSModel
from mmssod.utils.structure_utils import dict_split
from torch import nn
from mmssod.utils.gather import gather_same_shape_tensors, gather_diff_shape_tensors
# import torch.distributed as dist
from mmssod.models.utils.tensor_utils import norm_tensor
# from torch.profiler import profile, record_function, ProfilerActivity
from ...utils.structure_utils import weighted_loss, weighted_all_loss
from mmdet.core import build_sampler, build_assigner
import torch.distributed as dist
from mmdet.core import multiclass_nms
from mmssod.models.utils.eval_utils import cal_bboxes_overlaps, cal_bboxes_all_overlaps
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import time

@DETECTORS.register_module()
class URM(BurnInTSModel):
    def __init__(self, teacher: dict, student: dict, train_cfg=None, test_cfg=None, n_cls=20):
        super().__init__(teacher, student, train_cfg, test_cfg, n_cls)
        print("train_cfg:", train_cfg)

        self.train_cfg = train_cfg


        # self.unreliable_thr = train_cfg.get("unreliable_thr", 0.3)
        # self.neg_thr = train_cfg.get("neg_thr", 0.0005)
        # self.is_neg_loss = train_cfg.get("is_neg_loss", False)
        # self.is_iou_loss = train_cfg.get("is_iou_loss", False)
        # self.contrast_loss_weight = train_cfg.get("contrast_loss_weight", 1.0)
        #
        # self.is_ignore_ubreliable = train_cfg.get("ignore_ubreliable", False)
        # self.is_recall = train_cfg.get("is_recall", False)
        # self.is_soft_label_filter = train_cfg.get("is_soft_label_filter", False)
        #
        # self.alpha = train_cfg.get("alpha", 0.8)
        # self.theta1 = train_cfg.get("theta1", 0.1)
        # self.theta2 = train_cfg.get("theta2", 0.8)
        # self.topk = train_cfg.get("topk", 3)
        # self.neg_center = train_cfg.get("neg_center", 0.001)
        # self.neg_loss_weight = train_cfg.get("neg_loss_weight", 1.0)
        #
        # self.ema_decay = 0.999
        # self.ema_iteration = 0



        self.unreliable_thr = train_cfg.get("unreliable_thr", 0.3)
        self.neg_thr = train_cfg.get("neg_thr", 0.0005)
        self.is_neg_loss = train_cfg.get("is_neg_loss", False)
        self.is_iou_loss = train_cfg.get("is_iou_loss", False)
        self.contrast_loss_weight = train_cfg.get("contrast_loss_weight", 1.0)

        self.is_ignore_ubreliable = train_cfg.get("ignore_ubreliable", False)
        self.is_recall = train_cfg.get("is_recall", False)
        self.is_soft_label_filter = train_cfg.get("is_soft_label_filter", False)


        self.alpha = train_cfg.get("alpha", 0.8)
        self.theta1 = train_cfg.get("theta1", 0.1)
        self.theta2 = train_cfg.get("theta2", 0.8)
        self.topk = train_cfg.get("topk", 3)
        self.neg_center = train_cfg.get("neg_center", 0.001)

    def forward_train(self, imgs, img_metas, gt_bboxes, gt_labels, **kwargs):
        losses = dict()
        #----------------label data------------------
        sup_loss = self._compute_student_sup_negtive_loss(imgs, gt_bboxes, gt_labels, img_metas)
        # print(sup_loss)
        losses.update(sup_loss)

        #-------------------create pesudo labels--------------------
        weak_unsup = {'img':kwargs['img_unlabeled'],
                      'img_metas':kwargs['img_metas_unlabeled'],
                      'gt_bboxes':kwargs['gt_bboxes_unlabeled'],
                      'gt_labels':kwargs['gt_labels_unlabeled']}
        strong_unsup = {'img':kwargs['img_unlabeled_1'],
                        'img_metas':kwargs['img_metas_unlabeled_1'],
                        'gt_bboxes':kwargs['gt_bboxes_unlabeled_1'],
                        'gt_labels':kwargs['gt_labels_unlabeled_1']}
        strong_unsup, weak_unsup = self._gen_unreliable_pseudo_labels(weak_unsup, strong_unsup)



        unsup_loss = self._compute_student_unsup_negative_loss(weak_unsup, strong_unsup)
        losses.update(unsup_loss)


        # self._update_class_centers(weak_unsup, is_sup=False)
        return losses



    def _gen_unreliable_pseudo_labels(self, weak_unsup, strong_unsup):
        self.teacher.eval()
        # self.parse_informativeness(weak_unsup)

        strong_unsup.update({"gt_bboxes_true": None})
        strong_unsup.update({"gt_labels_true": None})

        with torch.no_grad():
            # extract feats
            feat = self.teacher.extract_feat(weak_unsup['img'])
            # extract proposal regions
            proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, weak_unsup['img_metas'])
            # extract det results from the detector
            det_bboxes, det_labels = self.teacher.roi_head.simple_test_bboxes(
                feat, weak_unsup['img_metas'], proposal_list, self.teacher.test_cfg.rcnn, rescale=False)

            result_bboxes = [bbox[:, :4] for bbox in det_bboxes]
            result_scores = [bbox[:, 4] for bbox in det_bboxes]

        self.student.eval()
        s_feat = self.student.extract_feat(weak_unsup['img'])
        self.student.train()

        # filter bboxes using thr
        reliable_bboxes = [bboxes[scores >= self.pesudo_thr] for (bboxes, scores) in zip(result_bboxes, result_scores)]
        reliable_labels = [labels[scores >= self.pesudo_thr] for (labels, scores) in zip(det_labels, result_scores)]
        reliable_scores = [scores1[scores2 >= self.pesudo_thr] for (scores1, scores2) in
                           zip(result_scores, result_scores)]
        gt_bboxes = reliable_bboxes
        gt_labels = reliable_labels
        gt_scores = reliable_scores
        unreliable_bboxes = [bboxes[(scores >= self.unreliable_thr) & (scores < self.pesudo_thr)] for (bboxes, scores)
                             in zip(result_bboxes, result_scores)]
        unreliable_labels = [labels[(scores >= self.unreliable_thr) & (scores < self.pesudo_thr)] for (labels, scores)
                             in zip(det_labels, result_scores)]
        unreliable_scores = [scores1[(scores2 >= self.unreliable_thr) & (scores2 < self.pesudo_thr)] for
                             (scores1, scores2)
                             in zip(result_scores, result_scores)]



        if self.is_recall is True:
            recall_bboxes = unreliable_bboxes
            recall_labels = unreliable_labels
            recall_scores = unreliable_scores

            filter_mask, mix_pro_list = self._t_s_filter_bboxes(feat, s_feat, recall_bboxes)
            recall_bboxes = [bboxes[mask] for (bboxes, mask) in zip(recall_bboxes, filter_mask)]
            recall_labels = [labels[mask].type(torch.long) for (labels, mask) in zip(recall_labels, filter_mask)]
            recall_scores = [scores[mask] for (scores, mask) in zip(recall_scores, filter_mask)]


            unreliable_bboxes = [bboxes[~mask] for (bboxes, mask) in zip(unreliable_bboxes, filter_mask)]
            unreliable_labels = [labels[~mask].type(torch.long) for (labels, mask) in
                                 zip(unreliable_labels, filter_mask)]
            unreliable_scores = [scores[~mask] for (scores, mask) in zip(unreliable_scores, filter_mask)]
            unreliabel_mix_pro = [pro[~mask] for (pro, mask) in zip(mix_pro_list, filter_mask)]
            strong_unsup.update({"unreliable_mix_pro": unreliabel_mix_pro})


            gt_bboxes = [torch.cat([bboxes1, bboxes2], dim=0) for (bboxes1, bboxes2) in
                         zip(reliable_bboxes, recall_bboxes)]
            gt_labels = [torch.cat([labels1, labels2]) for (labels1, labels2) in
                         zip(reliable_labels, recall_labels)]
            gt_scores = [torch.cat([scores1, scores2]) for (scores1, scores2) in
                         zip(reliable_scores, recall_scores)]



        strong_unsup.update({"gt_soft_labels": None})

        # transform det bbox
        M = self._extract_transform_matrix(weak_unsup, strong_unsup)
        gt_bboxes = self._transform_bbox(
            gt_bboxes,
            M,
            [meta["img_shape"] for meta in strong_unsup["img_metas"]],
        )

        strong_unsup.update({"gt_bboxes": gt_bboxes})
        strong_unsup.update({"gt_labels": gt_labels})
        strong_unsup.update({"gt_scores": gt_scores})

        # print([soft[labels] for soft, labels in zip(gt_soft_labels, gt_labels)])

        # transform det bbox
        M = self._extract_transform_matrix(weak_unsup, strong_unsup)
        unreliable_bboxes = self._transform_bbox(
            unreliable_bboxes,
            M,
            [meta["img_shape"] for meta in strong_unsup["img_metas"]],
        )
        strong_unsup.update({"unreliable_bboxes": unreliable_bboxes})
        strong_unsup.update({"unreliable_labels": unreliable_labels})
        strong_unsup.update({"unreliable_scores": unreliable_scores})
        strong_unsup.update({"trans_m": M})

        return strong_unsup, weak_unsup

    def _t_s_filter_bboxes(self, t_feat, s_feat, gt_bboxes):
        from mmdet.core import bbox2roi
        import torch.nn.functional as F
        rois = bbox2roi(gt_bboxes)
        num_proposals_per_img = [len(p) for p in gt_bboxes]

        t_bbox_results = self.teacher.roi_head._bbox_forward(t_feat, rois)
        t_cls_score = F.softmax(t_bbox_results['cls_score'], dim=-1)

        ind = 0
        t_score_list = []
        for i in range(len(num_proposals_per_img)):
            t_score_list.append(t_cls_score[ind:ind + num_proposals_per_img[i]])
            ind += num_proposals_per_img[i]

        s_bbox_results = self.student.roi_head._bbox_forward(s_feat, rois)
        s_cls_score = F.softmax(s_bbox_results['cls_score'], dim=-1)
        s_score_list = []
        ind = 0
        for i in range(len(num_proposals_per_img)):
            s_score_list.append(s_cls_score[ind:ind + num_proposals_per_img[i]])
            ind += num_proposals_per_img[i]

        def mix2(pro1, pro2, alpha):
            pro = alpha * pro1 + (1 - alpha) * pro2
            pro = pro / pro.sum(dim=1, keepdim=True)
            return pro

        mask_list = []
        mix_pro_list = []
        for i in range(len(num_proposals_per_img)):
            # score_mask = t_score_list[i] > 0.1
            # score_mask = torch.sum(score_mask, dim=1) > 1
            # print(score_mask)
            # print(t_score_list)

            # mix_pro = mix(t_score_list[i], s_score_list[i])
            mix_pro = mix2(t_score_list[i], s_score_list[i], self.alpha)
            bg_mask = mix_pro[:, self.n_cls] < self.theta1

            # top_thr = torch.sort(mix_pro, dim=1, descending=True).values[:, :2]
            # top_thr = torch.sort(mix_pro, dim=1, descending=True).values[:, :3]
            # top_ind = torch.sort(mix_pro, dim=1, descending=True).indices[:, :3]
            topk = torch.topk(mix_pro, k=3, dim=1)
            top_thr = topk.values[:, :3]
            top_ind = topk.indices[:, :3]

            bg_top_thr_list = []
            for j in range(len(top_thr)):
                # print(top_thr[j], top_ind[j])
                bg_top_thr = top_thr[j][top_ind[j] != self.n_cls]
                bg_top_thr_list.append(bg_top_thr[:2])
                # print(bg_top_thr)
            if len(bg_top_thr_list) != 0:
                top_thr = torch.stack(bg_top_thr_list)
            else:
                top_thr = top_thr[:, :2]
            obj_mask = (top_thr[:, 0] - top_thr[:, 1]) > self.theta2
            # print(mix_pro)
            # print(mask)
            mask_list.append(bg_mask & obj_mask)
            mix_pro_list.append(mix_pro)
            # mix_pro = mix_pro ** (1 / 0.5)
            # mix_pro = mix_pro / mix_pro.sum(dim=1, keepdim=True)
            # mix_soft_labels.append(mix_pro)
        return mask_list, mix_pro_list

    def _compute_student_sup_negtive_loss(self, img, gt_bboxes, gt_labels, img_metas):
        self.student.train()
        # print(sup_data)
        feats = self.student.extract_feat(img)
        losses = dict()
        proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                  self.student.test_cfg.rpn)
        rpn_losses, proposal_list = self.student.rpn_head.forward_train(
            feats,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=None,
            proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)

        if self.is_iou_loss is True:
            roi_losses = self.student.roi_head.iou_forward_train(feats,
                                                                 img_metas,
                                                                 proposal_list,
                                                                 gt_bboxes,
                                                                 gt_labels)
        else:
            roi_losses = self.student.roi_head.forward_train(feats, img_metas, proposal_list,
                                                             gt_bboxes, gt_labels)
        losses.update(roi_losses)

        losses = {"sup_" + k: v for k, v in losses.items()}

        return losses

    def _compute_student_unsup_negative_loss(self, weak_unsup, strong_unsup):
        # with torch.no_grad():
        #     weak_feat = self.teacher.extract_feat(weak_unsup["img"])
        strong_feat = self.student.extract_feat(strong_unsup["img"])

        self.student.train()
        losses = dict()

        roi_losses = self._compute_student_unsup_losses(weak_unsup, strong_unsup)
        roi_losses = {"unsup_" + k: v for k, v in roi_losses.items()}
        roi_losses = weighted_all_loss(roi_losses, self.unsup_loss_weight)
        losses.update(roi_losses)

        if self.is_neg_loss is True:
            neg_loss, roi_masks, bboxes_scores = self.student.roi_head._unreliable_neg_loss(x=strong_feat,
                                                                                            bboxes=strong_unsup["unreliable_bboxes"],
                                                                                            bboxes_scores=strong_unsup["unreliable_mix_pro"],
                                                                                            n_cls=self.n_cls,
                                                                                            top_thr=self.topk,
                                                                                            thr=self.neg_center)
            losses.update(neg_loss)
        return self._check_losses_item(losses, strong_feat[0].device)

    def _compute_student_unsup_losses(self, weak_unsup, strong_unsup, proposals=None):
        feat = self.student.extract_feat(strong_unsup["img"])
        t_feat = self.teacher.extract_feat(strong_unsup["img"])
        losses = dict()

        if self.is_ignore_ubreliable is True:
            ignore_bboxes = self._cal_ignore_unreliable_bboxes(strong_unsup["gt_bboxes"], strong_unsup["unreliable_bboxes"])
        else:
            ignore_bboxes = None
        # student RPN forward and loss
        if self.student.with_rpn:
            proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                      self.student.test_cfg.rpn)

            rpn_losses, proposal_list = self._compute_student_rpn_losses(feat,
                                                                         strong_unsup["img_metas"],
                                                                         strong_unsup["gt_bboxes"],
                                                                         gt_bboxes_ignore=ignore_bboxes,
                                                                         proposal_cfg=proposal_cfg,
                                                                         gt_bboxes_true=strong_unsup["gt_bboxes_true"])
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self._compute_student_soft_rcnn_losses(feat,
                                                            t_feat,
                                                            strong_unsup["img_metas"],
                                                            proposal_list,
                                                            strong_unsup["gt_bboxes"],
                                                            strong_unsup["gt_labels"],
                                                            strong_unsup['gt_soft_labels'],
                                                            gt_bboxes_ignore=ignore_bboxes,
                                                            gt_bboxes_true=strong_unsup["gt_bboxes_true"],
                                                            gt_labels_true=strong_unsup["gt_labels_true"])
        losses.update(roi_losses)

        return losses

    def _compute_student_soft_rcnn_losses(self,
                                          x,
                                          t_feat,
                                          img_metas,
                                          proposal_list,
                                          gt_bboxes,
                                          gt_labels,
                                          gt_soft_labels,
                                          gt_bboxes_ignore=None,
                                          gt_masks=None,
                                          gt_bboxes_true=None,
                                          gt_labels_true=None,
                                          **kwargs):
        if self.student.roi_head.with_bbox or self.student.roi_head.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.student.roi_head.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                # print("assign_result:", assign_result)
                sampling_result = self.student.roi_head.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # print("sample result:", sampling_result)
                if self.filter_unsup_regions:
                    sampling_result = self._filter_unsup_region(sampling_result,
                                                                gt_bboxes_true[i])
                if self.print_pesudo_summary:
                    self._add_unsup_sampling_bboxes(sampling_result.pos_bboxes.clone(),
                                                    sampling_result.neg_bboxes.clone(),
                                                    gt_bboxes_true[i].detach(),
                                                    img_metas[i])

                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.is_iou_loss is True:
            bbox_results, rois, bbox_weights = self.student.roi_head.iou_weighted_loss(x,
                                                                                       sampling_results,
                                                                                       gt_bboxes,
                                                                                       gt_labels,
                                                                                       img_metas,
                                                                                       t_feat,
                                                                                       self.teacher)

        else:
            bbox_results = self.student.roi_head._bbox_forward_train(x, sampling_results,
                                                                     gt_bboxes, gt_labels,
                                                                     img_metas)
        losses.update(bbox_results['loss_bbox'])

        return losses



    @torch.no_grad()
    def _cal_ignore_unreliable_bboxes(self, gt_bboxes, unreliable_bboxes):
        iou_results = [self.iou_calculator(gts, unrlia, mode='iof') for gts, unrlia in
                       zip(gt_bboxes, unreliable_bboxes)]

        masks = [overlaps.max(dim=0).values < 0.3 if overlaps.shape[0] != 0 else torch.ones(overlaps.shape[1],
                                                                                            dtype=torch.bool).to(
            overlaps.device)
                 for overlaps in iou_results]

        result_bbox = [unralia[mask] for unralia, mask in zip(unreliable_bboxes, masks)]
        return result_bbox
