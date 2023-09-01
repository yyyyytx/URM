from mmdet.models import DETECTORS, BaseDetector, build_detector
from typing import Dict
# from abc import ABCMeta
import torch
from mmssod.utils.structure_utils import dict_split
from mmssod.models.utils.bbox_utils import Transform2D
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core.evaluation.mean_ap import tpfp_default
from mmcv import Config, DictAction

import numpy as np
np.set_printoptions(suppress=True, threshold=10000)
from mmdet.utils import get_root_logger
from ..utils.eval_utils import get_cls_results, cal_recall_precisions, cal_unsup_sampling_overlaps, cal_bboxes_overlaps
from ...utils.structure_utils import weighted_loss
from mmdet.core.bbox.samplers import SamplingResult
from mmdet.utils.profiling import profile_time
import math
from mmssod.utils.gather import gather_same_shape_tensors
import torch.distributed as dist
import mmcv
import os.path as osp
import pickle
from torch.profiler import profile, record_function, ProfilerActivity
from mmdet.core import build_sampler, build_assigner

@DETECTORS.register_module()
class BurnInTSModel(BaseDetector):
    '''Base arch for teacher-student model with burn-in stage'''

    def __init__(self, teacher: dict, student: dict, train_cfg=None, test_cfg=None, n_cls=20):
        # super(self, BurnInTSModel).__init__()

        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        print('building Model')

        ## initialize teacher student models
        self.teacher = build_detector(teacher)
        self.student = build_detector(student)
        self.submodules = ['teacher', 'student']

        self.momentum = 0.999
        self.unsup_loss_weight = train_cfg.get("unsup_loss_weight", 1.0)

        self.run_model = self.teacher
        self.run_model_name = self.submodules[0]

        self.is_semi_train = False
        # self.n_cls = self.train_cfg.n_cls
        self.n_cls = n_cls
        ## get debug setting
        self.print_pesudo_summary = train_cfg.get("print_pesudo_summary", False)
        self.pesudo_summary_iou_thrs = train_cfg.get("pesudo_summary_iou_thrs", [0.5])
        self.unsup_sample_measure_iou_thrs = train_cfg.get("unsup_sample_measure_iou_thrs", [0.5])

        self.measure_det_bboxes_list = []
        self.measure_gts_list = []

        self.pos_proposals_list = []
        self.neg_proposals_list = []
        self.gts_proposals_list = []
        self.pos_roi_weight_list = []
        self.neg_roi_weight_list = []
        self.pos_roi_labels_list = []
        self.neg_roi_labels_list = []
        self.img_meta_list = []

        # self.neg_mask_list = []
        self.neg_unreliable_bboxes_list = []
        self.neg_unreliable_mask_list = []



        assigner_cfg = train_cfg.get("assigner", None)
        self.is_assigner_recreate = False
        if assigner_cfg is not None:
            print('create assigner')
            self.is_assigner_recreate = True
            self.rcnn_bbox_unsup_assigner = build_assigner(train_cfg.assigner)
            self.rcnn_bbox_sup_assigner = build_assigner(self.student.train_cfg.rcnn.assigner)





        self.check_geo_trans_bboxes = train_cfg.get("check_geo_trans_bboxes", False)
        self.pesudo_thr = train_cfg.get("pesudo_thr", 0.7)
        self.is_no_nms = train_cfg.get("is_no_nms", False)


        self.filter_unsup_regions = train_cfg.get("filter_unsup_regions", False)
        self.filter_unsup_positive = train_cfg.get("filter_unsup_positive", False)
        self.filter_unsup_negative = train_cfg.get("filter_unsup_negative", False)
        self.train_semi_with_gts = train_cfg.get("train_semi_with_gts", False)
        self.control_pos_neg_ratio = train_cfg.get("control_pos_neg_ratio", -1)
        self.filter_incorrect = train_cfg.get("filter_incorrect", False)

    def switch_semi_train(self):
        self.is_semi_train = True

    def start_semi_train(self):
        print('start semi train')
        self.is_semi_train = True
        for param_t, param_s in zip(self.teacher.parameters(),
                                    self.student.parameters()):
            param_s.data.copy_(param_t.data)
            param_t.requires_grad = False
            param_s.requires_grad = True

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the target network."""
        for param_t, param_s in zip(self.teacher.parameters(),
                                    self.student.parameters()):
            param_t.data = param_t.data * self.momentum + \
                           param_s.data * (1. - self.momentum)

    def forward_train(self, imgs, img_metas, **kwargs):
        kwargs.update({"img": imgs})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})

        #   data_groups : dict {"sup"：sup_data , "unsup":unsup_data}
        #   sup/unsup_data : dict
        #       "gt_bboxes", "gt_labels", "img", "img_metas", "tag"
        data_groups = dict_split(kwargs, "tag")

        losses = dict()
        if self.is_assigner_recreate is True:
            self._switch_sup_train()

        sup_loss = self.student.forward_train(**data_groups["sup"])
        sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
        losses.update(sup_loss)

        if self.is_assigner_recreate is True:
            self._switch_unsup_train()
        strong_unsup = self._gen_pseudo_labels(data_groups["unsup"])
        unsup_loss = self._compute_student_unsup_losses(strong_unsup)
        unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
        losses.update(unsup_loss)

        losses = weighted_loss(losses, self.unsup_loss_weight)
        return losses

    def _compute_student_unsup_losses(self, strong_unsup, proposals=None):
        feat = self.student.extract_feat(strong_unsup["img"])
        losses = dict()
        # student RPN forward and loss
        if self.student.with_rpn:
            proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                      self.student.test_cfg.rpn)

            rpn_losses, proposal_list = self._compute_student_rpn_losses(feat,
                                                                         strong_unsup["img_metas"],
                                                                         strong_unsup["gt_bboxes"],
                                                                         gt_bboxes_ignore=None,
                                                                         proposal_cfg=proposal_cfg,
                                                                         gt_bboxes_true=strong_unsup["gt_bboxes_true"])
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self._compute_student_rcnn_losses(feat,
                                                       strong_unsup["img_metas"],
                                                       proposal_list,
                                                       strong_unsup["gt_bboxes"],
                                                       strong_unsup["gt_labels"],
                                                       gt_bboxes_true=strong_unsup["gt_bboxes_true"],
                                                       gt_labels_true=strong_unsup["gt_labels_true"])
        losses.update(roi_losses)

        # return self._check_losses_item(losses, feat[0].device)
        return losses

    def _compute_student_rpn_losses(self,
                                    feats,
                                    img_metas,
                                    gt_bboxes,
                                    gt_bboxes_ignore=None,
                                    proposal_cfg=None,
                                    gt_bboxes_true=None):
        rpn_outs = self.student.rpn_head(feats)

        loss_inputs = rpn_outs + (gt_bboxes, img_metas)
        if self.filter_unsup_regions == True:
            losses = self.student.rpn_head.filter_loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,
                filter_neg=self.filter_unsup_negative,
                filter_pos=self.filter_unsup_positive,
                gt_bboxes_true=gt_bboxes_true
            )
        else:
            losses = self.student.rpn_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        proposal_list = self.student.rpn_head.get_bboxes(
            *rpn_outs, img_metas=img_metas, cfg=proposal_cfg)
        return losses, proposal_list

    def _compute_student_rcnn_losses(self,
                                     x,
                                     img_metas,
                                     proposal_list,
                                     gt_bboxes,
                                     gt_labels,
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
        if self.student.roi_head.with_bbox:
            bbox_results = self.student.roi_head._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.student.roi_head.with_mask:
            mask_results = self.student.roi_head._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _check_losses_item(self, losses, device):
        # prevent the empty pesudo errors
        items_list = ["unsup_loss_rpn_cls", "unsup_loss_rpn_bbox", "unsup_loss_cls", "unsup_loss_bbox", "unsup_acc"]
        for item in items_list:
            if item not in losses.keys():
                losses[item] = torch.tensor(0.).to(device)
        return losses

    def _gen_pseudo_labels(self, unsup_data):
        self.teacher.eval()
        weak_unsup, strong_unsup = self._split_unsup_data(unsup_data)

        # extract feats
        feat = self.teacher.extract_feat(weak_unsup['img'])

        # extract proposal regions
        proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, weak_unsup['img_metas'])

        if self.is_no_nms == True:
            test_cfg1 = {'score_thr': 0.5,
                        'nms': None,
                        'max_per_img': 100}
            test_cfg1 = Config(test_cfg1)
            det_bboxes1, det_labels1 = self.teacher.roi_head.simple_test_bboxes(
                feat, weak_unsup['img_metas'], proposal_list, test_cfg1, rescale=False)

            test_cfg2 = {'score_thr': 0.7,
                        'nms': {'type': 'nms', 'iou_threshold': 0.5},
                        # 'nms': None,
                        'max_per_img': 100}
            test_cfg2 = Config(test_cfg2)
            det_bboxes2, det_labels2 = self.teacher.roi_head.simple_test_bboxes(
                feat, weak_unsup['img_metas'], proposal_list, test_cfg2, rescale=False)
            # res =  [cal_bboxes_overlaps(bboxes1[:, :4], bboxes2[:,:4]) for (bboxes1, bboxes2) in zip(det_bboxes1, det_bboxes2)]
            # for (overlaps, inds), bboxes1, bboxes2 in zip(res, det_bboxes1, det_bboxes2):
            #     mask = overlaps > 0.5
            #     if len(det_bboxes2[0]) == 0:
            #         res_bboxes = det_bboxes2
            #         res_labels = det_labels2
            #
            #     else:
            #         selected = det_labels1[0][mask] == det_labels2[0][inds][mask]
            #         res_bboxes = [det_bboxes1[0][mask][selected]]
            #         res_labels = [det_labels1[0][mask][selected]]
            result_bboxes = []
            result_labels = []
            result_scores = []
            for bboxes1, labels1, bboxes2, labels2 in zip(det_bboxes1, det_labels1, det_bboxes2, det_labels2):
                overlaps, inds = cal_bboxes_overlaps(bboxes1[:, :4], bboxes2[:, :4])
                mask = overlaps > 0.5
                if len(bboxes2) == 0:
                    res_bboxes = bboxes2
                    res_labels = labels2
                else:
                    selected = labels1[mask] == labels2[inds][mask]
                    res_bboxes = bboxes1[mask][selected]
                    res_labels = labels1[mask][selected]

                result_bboxes.append(res_bboxes[:, :4])
                result_labels.append(res_labels)
                result_scores.append(res_bboxes[:, 4])

        else:
            # extract det results from the detector
            det_bboxes, det_labels = self.teacher.roi_head.simple_test_bboxes(
                feat, weak_unsup['img_metas'], proposal_list, self.teacher.test_cfg.rcnn, rescale=False)

            result_bboxes = [bbox[:, :4] for bbox in det_bboxes]
            result_scores = [bbox[:, 4] for bbox in det_bboxes]

            # filter bboxes using thr
            result_bboxes = [bboxes[scores >= self.pesudo_thr] for (bboxes, scores) in zip(result_bboxes, result_scores)]
            result_labels = [labels[scores >= self.pesudo_thr] for (labels, scores) in zip(det_labels, result_scores)]
            result_scores = [scores1[scores2 >= self.pesudo_thr] for (scores1, scores2) in zip(result_scores, result_scores)]
        # transform det bbox
        M = self._extract_transform_matrix(weak_unsup, strong_unsup)
        result_bboxes = self._transform_bbox(
            result_bboxes,
            M,
            [meta["img_shape"] for meta in strong_unsup["img_metas"]],
        )

        if self.check_geo_trans_bboxes:
            trans_bboxes = self._transform_bbox(
                weak_unsup["gt_bboxes"],
                M,
                [meta["img_shape"] for meta in strong_unsup["img_metas"]],
            )
            self._check_geo_trans_bboxes(trans_bboxes, strong_unsup["gt_bboxes"])

        strong_unsup.update({"gt_bboxes_true" : [bboxes.clone() for bboxes in strong_unsup["gt_bboxes"]]})
        strong_unsup.update({"gt_labels_true" : [labels.clone() for labels in strong_unsup["gt_labels"]]})

        if self.print_pesudo_summary:
            self._add_summary_bboxes([{"bboxes":torch.cat([bboxes.clone(), torch.unsqueeze(scores.clone(), dim=1)], dim=1).detach().cpu().numpy(), "labels":labels.clone().detach().cpu().numpy()} for (bboxes, scores, labels) in zip(result_bboxes, result_scores, result_labels)],
                                     [{"bboxes":bboxes.detach().cpu().numpy(), "labels":labels.detach().cpu().numpy()} for (bboxes, labels) in zip(strong_unsup["gt_bboxes"], strong_unsup["gt_labels"])])

        # replace with the pesudo labels/bboxes
        strong_unsup.update({"gt_bboxes" : result_bboxes})
        strong_unsup.update({"gt_labels" : result_labels})
        strong_unsup.update({"trans_m": M})

        return strong_unsup


    def _split_unsup_data(self, unsup_data):
        weak_unsup_data = {k : v[::2] for k, v in unsup_data.items()}
        strong_unsup_data = {k : v[1::2] for k, v in unsup_data.items()}
        return weak_unsup_data, strong_unsup_data

    def _extract_transform_matrix(self, weak_unsup_data, strong_unsup_data):
        M_w = [meta['transform_matrix'] for meta in weak_unsup_data['img_metas']]
        M_s = [meta['transform_matrix'] for meta in strong_unsup_data['img_metas']]
        # print(M_w, M_s)
        M = [torch.from_numpy(m2) @ torch.from_numpy(m1).inverse() for (m1, m2) in zip(M_w, M_s)]
        return M

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    def extract_feat(self, imgs):
        feat = self.run_model.extract_feat(imgs)
        return feat

    def simple_test(self, img, img_metas, **kwargs):
        return self.teacher.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.aug_test(imgs, img_metas, **kwargs)


    ## debug functions
    def _control_region_ratio(self, sampling_result):
        pos_region_num = len(sampling_result.pos_bboxes)
        neg_region_num = math.ceil(pos_region_num * self.control_pos_neg_ratio)
        if neg_region_num > len(sampling_result.neg_bboxes):
            return sampling_result
        neg_bboxes = sampling_result.neg_bboxes
        select_ind = torch.randperm(len(neg_bboxes))[:neg_region_num]
        sampling_result.neg_bboxes = neg_bboxes[select_ind]
        return sampling_result

    def _filte_incorrect(self, det_bboxes, det_labels, det_socres, gt_bboxes, gt_labels):
        results = [cal_bboxes_overlaps(det_bbox, gt_bbox) for (det_bbox, gt_bbox) in zip(det_bboxes, gt_bboxes)]
        filter_bboxes = []
        filter_labels = []
        filter_socres = []
        for i in range(len(results)):
            # print(results, i)
            overlaps, gts_inds = results[i]
            if len(overlaps) == 0:
                filter_bboxes.append(det_bboxes[i])
                filter_labels.append(det_labels[i])
                filter_socres.append(det_socres[i])
                continue
            # det_bbox = det_bboxes[i][overlaps >= 0.7]
            overlaps_inds = overlaps >= 0.5
            correct_inds = det_labels[i] == gt_labels[i][gts_inds]
            correct_inds = overlaps_inds & correct_inds

            filter_bboxes.append(det_bboxes[i][correct_inds])
            filter_labels.append(det_labels[i][correct_inds])
            filter_socres.append(det_socres[i][correct_inds])
        return filter_bboxes, filter_labels, filter_socres

    def _filter_unsup_region(self, sampling_result, gts_bboxes):
        pos_overlaps, neg_overlaps = cal_unsup_sampling_overlaps(sampling_result.pos_bboxes,
                                                                 sampling_result.neg_bboxes,
                                                                 gts_bboxes)
        pos_bboxes, neg_bboxes = sampling_result.pos_bboxes, sampling_result.neg_bboxes
        if self.filter_unsup_positive:
            # filter pesudo positive regions which are backgrouds
            pos_bboxes = pos_bboxes[pos_overlaps > 0.5]
            sampling_result.pos_gt_labels = sampling_result.pos_gt_labels[pos_overlaps > 0.5]
            sampling_result.pos_gt_bboxes = sampling_result.pos_gt_bboxes[pos_overlaps > 0.5]
        if self.filter_unsup_negative:
            # filter pesudo negative regions which are foregrounds
            neg_bboxes = neg_bboxes[neg_overlaps < 0.5]
        sampling_result.pos_bboxes = pos_bboxes
        sampling_result.neg_bboxes = neg_bboxes
        return sampling_result

    def _add_summary_bboxes(self, pesudo_results, gt_results):
        # print("pesudo_results:",pesudo_results)
        # print("gt_resutls:", gt_results)

        self.measure_det_bboxes_list.extend(pesudo_results)
        self.measure_gts_list.extend(gt_results)


    def _add_unsup_sampling_bboxes(self,
                                   pos_bboxes,
                                   neg_bboxes,
                                   gts_bboxes,
                                   img_meta,
                                   pos_weights=None,
                                   neg_weights=None,
                                   pos_roi_labels=None,
                                   neg_roi_labels=None,
                                   ):
        self.pos_proposals_list.append(pos_bboxes)
        self.neg_proposals_list.append(neg_bboxes)
        self.gts_proposals_list.append(gts_bboxes)
        self.img_meta_list.append(img_meta)

        if pos_weights is not None:
            self.pos_roi_weight_list.append(pos_weights)
            self.neg_roi_weight_list.append(neg_weights)
            self.pos_roi_labels_list.append(pos_roi_labels)
            self.neg_roi_labels_list.append(neg_roi_labels)

    def _add_neg_loss_info(self,
                           unreliable_mask,
                           unreliable_bboxes):
        self.neg_unreliable_mask_list.append(unreliable_mask)
        self.neg_unreliable_bboxes_list.append(unreliable_bboxes)

    def log_recall_precisions(self, epoch_num=0, iter_num=0):
        # logger = get_root_logger(log_level="INFO")
        # print("dump pesudo infos")
        rank = dist.get_rank()
        work_dir = self.train_cfg.work_dir
        summary_path = osp.join(work_dir, 'pesudo_infos')
        mmcv.mkdir_or_exist(osp.abspath(summary_path))
        info_path = osp.join(summary_path, 'iter_%d_rank_%d.pkl'%(iter_num, rank))
        with open(info_path, 'wb') as f:
            pickle.dump({"det_bboxes":self.measure_det_bboxes_list,
                         "gts_bboxes":self.measure_gts_list}, f)


        # print("unsup regions infos")
        summary_path = osp.join(work_dir, 'proposal_regions_infos')
        mmcv.mkdir_or_exist(osp.abspath(summary_path))
        info_path = osp.join(summary_path, 'iter_%d_rank_%d.pkl' % (iter_num, rank))
        with open(info_path, 'wb') as f:
            # if len(self.pos_roi_weight_list) != 0:
            pickle.dump({"pos_regions": self.pos_proposals_list,
                         "neg_regions": self.neg_proposals_list,
                         "gts_regions": self.gts_proposals_list,
                         "pos_weights": self.pos_roi_weight_list,
                         "neg_weights": self.neg_roi_weight_list,
                         "pos_labels": self.pos_roi_labels_list,
                         "neg_labels": self.neg_roi_labels_list,
                         "img_metas": self.img_meta_list
                         }, f)
            # else:
            #     pickle.dump({"pos_regions": self.pos_proposals_list,
            #                  "neg_regions": self.neg_proposals_list,
            #                  "gts_regions": self.gts_proposals_list}, f)

        summary_path = osp.join(work_dir, 'neg_unreliable_infos')
        mmcv.mkdir_or_exist(osp.abspath(summary_path))
        info_path = osp.join(summary_path, 'iter_%d_rank_%d.pkl' % (iter_num, rank))
        with open(info_path, 'wb') as f:
            # if len(self.pos_roi_weight_list) != 0:
            pickle.dump({"unraliable_bboxes": self.neg_unreliable_bboxes_list,
                         "unraliable_mask": self.neg_unreliable_mask_list,
                         "gts_bboxes": self.measure_gts_list,
                         }, f)

        self.measure_det_bboxes_list = []
        self.measure_gts_list = []
        self.pos_proposals_list = []
        self.neg_proposals_list = []
        self.gts_proposals_list = []
        self.pos_roi_weight_list = []
        self.neg_roi_weight_list = []
        self.pos_roi_labels_list = []
        self.neg_roi_labels_list = []
        self.img_meta_list = []
        self.neg_unreliable_bboxes_list = []
        self.neg_unreliable_mask_list = []


    def _check_geo_trans_bboxes(self, trans_bboxes, strong_gt):
        e_m = [torch.abs(m_trans - m_gt) > 2.0 for (m_trans, m_gt) in zip(trans_bboxes, strong_gt)]
        for e in e_m:
            aa = torch.sum(e).item()
            if aa != 0:
                print('error pesudo bboxes')
                print("gt:", strong_gt)
                print("pesudo:", trans_bboxes)
                raise 'check pesudo bboxes fails'

    def _switch_sup_train(self):
        self.student.roi_head.bbox_assigner = self.rcnn_bbox_sup_assigner

    def _switch_unsup_train(self):
        self.student.roi_head.bbox_assigner = self.rcnn_bbox_unsup_assigner


