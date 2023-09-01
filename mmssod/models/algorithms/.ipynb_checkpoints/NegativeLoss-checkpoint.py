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

@DETECTORS.register_module()
class NegativeLoss(BurnInTSModel):
    def __init__(self, teacher: dict, student: dict, train_cfg=None, test_cfg=None, n_cls=20):
        super().__init__(teacher, student, train_cfg, test_cfg, n_cls)
        print("train_cfg:",train_cfg)

        self.train_cfg = train_cfg

        # queues settings
        # self.feat_dim = train_cfg.get("feat_dim", 1024)
        # self.bbox_feat_dim = train_cfg.get("bbox_feat_dim", 1024)
        # self.pos_queue_len = train_cfg.get("pos_queue_len", 100)
        # self.neg_queue_len = train_cfg.get("neg_queue_len", 65536)
        # self.region_bg_max_num = train_cfg.get("region_bg_max_num", 10)
        # self.region_fg_max_num = train_cfg.get("region_fg_max_num", 80)

        # self.cls_count = torch.zeros(n_cls).to(self.device)
        # self.register_buffer('cls_count', torch.zeros(1, n_cls))
        # self._init_cls_count()
        # store the keys of pos/neg regions
        # self.register_buffer('pos_queue', torch.zeros(self.n_cls ,self.bbox_feat_dim, self.pos_queue_len))
        # self.register_buffer('pos_queue_ptr', torch.zeros((self.n_cls, 1), dtype=torch.long))
        # self.register_buffer('neg_queue', torch.zeros(self.bbox_feat_dim, self.neg_queue_len))
        # self.register_buffer('neg_queue_ptr', torch.zeros(1, dtype=torch.long))
        
        # self.register_buffer('projector_pos_queue', torch.zeros(self.n_cls, self.feat_dim, self.pos_queue_len))
        # self.register_buffer('projector_pos_queue_ptr', torch.zeros((self.n_cls, 1), dtype=torch.long))
        # self.register_buffer('projector_neg_queue', torch.zeros(self.feat_dim, self.neg_queue_len))
        # self.register_buffer('projector_neg_queue_ptr', torch.zeros(1, dtype=torch.long))

        # self.register_buffer('center_features', torch.zeros(self.n_cls+1, self.feat_dim))
        # self._init_center_features()
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


        self.ema_decay = 0.999
        self.ema_iteration = 0




    def forward_train(self, imgs, img_metas, **kwargs):
        kwargs.update({"img": imgs})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")

        losses = dict()

        sup_loss = self._compute_student_sup_negtive_loss(data_groups["sup"])
        # self._accumulate_train_number_per_category(data_groups["sup"]["gt_labels"])
        losses.update(sup_loss)

        # self._update_reliable_queues(data_groups["sup"])
        # self._update_class_centers(data_groups["sup"])

        weak_unsup, strong_unsup = self._split_unsup_data(data_groups["unsup"])

        strong_unsup, weak_unsup = self._gen_unreliable_pseudo_labels(weak_unsup, strong_unsup)

        # self._accumulate_train_number_per_category(strong_unsup['gt_labels'])

        unsup_loss = self._compute_student_unsup_negative_loss(weak_unsup, strong_unsup)
        losses.update(unsup_loss)

        self.ema_iteration += 1

        # self._update_class_centers(weak_unsup, is_sup=False)
        return losses

    # def _init_cls_count(self):
    #     labeled_count =[306, 353, 486, 290, 505, 229, 1250, 376, 798, 259, 215, 510, 362, 339, 4690, 514, 257, 248, 297, 324]
    #     for i in range(self.n_cls):
    #         self.cls_count[0][i] = labeled_count[i]
    #
    # def _init_center_features(self):
    #     self.center_features = torch.load('/home/liu/ytx/SS-OD/rcnn_voc_initial_centers.pt')
    #     print(self.center_features)
    #
    # def _accumulate_train_number_per_category(self, label_list):
    #     # pass
    #     # print(label_list)
    #     # []
    #     tmp_count = torch.zeros((1, self.n_cls)).to(self.cls_count.device)
    #     for labels in label_list:
    #         for c in range(self.n_cls):
    #             tmp_count[0][c] += torch.sum(labels == c)
    #
    #     out = gather_same_shape_tensors(tmp_count)
    #     self.cls_count[0] =self.cls_count[0] + torch.sum(out, dim=0)

        # print(self.cls_count)

    # def _extract_key(self, filename):
    #     name = filename.split('/')[-1]
    #     return name.split('.')[0]


    # def _ori_update_label_memory(self, det_bboxes, det_labels, det_scores, weak_unsup):
    #     pesudo_bboxes = self._transform_bbox(det_bboxes,
    #                                          [torch.from_numpy(meta["transform_matrix"]).inverse() for meta in
    #                                           weak_unsup["img_metas"]],
    #                                          [meta["ori_shape"] for meta in weak_unsup["img_metas"]])
    #
    #     for bboxes, labels, scores, meta_info in zip(pesudo_bboxes, det_labels, det_scores, weak_unsup['img_metas']):
    #         key = self._extract_key(meta_info['filename'])
    #         info = torch.cat([bboxes, torch.unsqueeze(scores, dim=1), torch.unsqueeze(labels, dim=1)], dim=1).cpu()
    #         gather_objects = [key, info]
    #         outputs = [None for _ in range(dist.get_world_size())]
    #         dist.all_gather_object(outputs, gather_objects)
    #         for item in outputs:
    #             # self.label_memory_bank[item[0]] = item[1]
    #             if item[0] in self.label_memory_bank.keys():
    #                 update_pesudo = self.label_memory_bank[item[0]]
    #                 update_pesudo = torch.cat([update_pesudo, item[1]], dim=0)
    #                 # print(update_pesudo)
    #                 nms_scores = torch.zeros((len(update_pesudo)), self.n_cls + 1)
    #                 scores = update_pesudo[:, 4]
    #                 labels = update_pesudo[:, 5]
    #                 for i in range(len(scores)):
    #                     # print(labels[i])
    #                     nms_scores[i][labels[i].type(torch.long)] = scores[i]
    #                 # print(nms_scores)
    #
    #                 nms_bboxes, nms_labels, inds = multiclass_nms(update_pesudo[:, :4],
    #                                             nms_scores,
    #                                             self.unreliable_thr,
    #                                             self.train_cfg.region_bg_nms_cfg,
    #                                             return_inds=True)
    #                 # print(nms_bboxes)
    #                 # print(nms_labels)
    #                 update_pesudo = torch.cat([nms_bboxes, torch.unsqueeze(nms_labels, dim=1)], dim=1).cpu()
    #                 # print(update_pesudo)
    #                 self.label_memory_bank[item[0]] = update_pesudo
    #             else:
    #                 self.label_memory_bank[item[0]] = item[1]
    #
    # def _update_label_memory(self, det_bboxes, det_labels, det_scores, weak_unsup):
    #     pesudo_bboxes = self._transform_bbox(det_bboxes,
    #                                          [torch.from_numpy(meta["transform_matrix"]).inverse() for meta in weak_unsup["img_metas"]],
    #                                          [meta["ori_shape"] for meta in weak_unsup["img_metas"]])
    #
    #     for bboxes, labels, scores, meta_info in zip(pesudo_bboxes, det_labels, det_scores, weak_unsup['img_metas']):
    #         key = self._extract_key(meta_info['filename'])
    #         info = torch.cat([bboxes, torch.unsqueeze(scores, dim=1), torch.unsqueeze(labels, dim=1)], dim=1).cpu()
    #         gather_objects = [key, info]
    #         outputs = [None for _ in range(dist.get_world_size())]
    #         dist.all_gather_object(outputs, gather_objects)
    #         for item in outputs:
    #             self.label_memory_bank[item[0]] = item[1]
    #             if item[0] in self.label_memory_bank.keys():
    #                 update_pesudo = self.label_memory_bank[item[0]]
    #                 update_pesudo = torch.cat([update_pesudo, item[1]], dim=0)
    #                 nms_scores = torch.zeros((len(update_pesudo)), self.n_cls + 1)
    #                 scores = update_pesudo[:, 4]
    #                 labels = update_pesudo[:, 5]
    #                 for i in range(len(scores)):
    #                     nms_scores[i][labels[i].type(torch.long)] = scores[i]
    #
    #                 nms_bboxes, nms_labels, inds = multiclass_nms(update_pesudo[:, :4],
    #                                             nms_scores,
    #                                             self.unreliable_thr,
    #                                             self.train_cfg.region_bg_nms_cfg,
    #                                             return_inds=True)
    #                 update_pesudo = torch.cat([nms_bboxes, torch.unsqueeze(nms_labels, dim=1)], dim=1).cpu()
    #                 self.label_memory_bank[item[0]] = update_pesudo
    #             else:
    #                 self.label_memory_bank[item[0]] = item[1]
    #
    # def _get_pesudo_from_memory(self, weak_unsup, device):
    #     memory_pesudo = [self.label_memory_bank[self._extract_key(meta_info['filename'])] if (self._extract_key(meta_info['filename']) in self.label_memory_bank.keys()) else torch.ones((0, 6))
    #                      for meta_info in weak_unsup['img_metas']]
    #     # memory_pesudo = [self.label_memory_bank[self._extract_key(meta_info['filename'])] for meta_info in weak_unsup['img_metas']]
    #     pesudo_bboxes = [pesudo[:, :4].to(device) for pesudo in memory_pesudo]
    #     pesudo_scores = [pesudo[:, 4].to(device) for pesudo in memory_pesudo]
    #     pesudo_labels = [pesudo[:, 5].type(torch.long).to(device) for pesudo in memory_pesudo]
    #     pesudo_bboxes = self._transform_bbox(pesudo_bboxes,
    #                                          [torch.from_numpy(meta["transform_matrix"]) for meta in
    #                                           weak_unsup["img_metas"]],
    #                                          [meta["img_shape"] for meta in weak_unsup["img_metas"]])
    #
    #     return pesudo_bboxes, pesudo_scores, pesudo_labels
    #
    # def _update_thr_label_memory(self, det_bboxes, det_labels, det_scores, weak_unsup):
    #     pesudo_bboxes = self._transform_bbox(det_bboxes,
    #                                          [torch.from_numpy(meta["transform_matrix"]).inverse() for meta in
    #                                           weak_unsup["img_metas"]],
    #                                          [meta["ori_shape"] for meta in weak_unsup["img_metas"]])
    #     for bboxes, labels, scores, meta_info in zip(pesudo_bboxes, det_labels, det_scores, weak_unsup['img_metas']):
    #         key = self._extract_key(meta_info['filename'])
    #         info = torch.cat([bboxes, torch.unsqueeze(scores, dim=1), torch.unsqueeze(labels, dim=1)], dim=1).cpu()
    #         gather_objects = [key, info]
    #         outputs = [None for _ in range(dist.get_world_size())]
    #         dist.all_gather_object(outputs, gather_objects)
    #         for item in outputs:
    #             self.thr_label_memory_bank[item[0]] = item[1]
    #
    # def _get_thr_pesudo_from_memory(self, weak_unsup, device):
    #     memory_pesudo = [self.thr_label_memory_bank[self._extract_key(meta_info['filename'])] if (self._extract_key(meta_info['filename']) in self.thr_label_memory_bank.keys()) else torch.ones((0, 6))
    #                      for meta_info in weak_unsup['img_metas']]
    #     pesudo_bboxes = [pesudo[:, :4].to(device) for pesudo in memory_pesudo]
    #     pesudo_scores = [pesudo[:, 4].to(device) for pesudo in memory_pesudo]
    #     pesudo_labels = [pesudo[:, 5].type(torch.long).to(device) for pesudo in memory_pesudo]
    #     pesudo_bboxes = self._transform_bbox(pesudo_bboxes,
    #                                          [torch.from_numpy(meta["transform_matrix"]) for meta in
    #                                           weak_unsup["img_metas"]],
    #                                          [meta["img_shape"] for meta in weak_unsup["img_metas"]])
    #
    #     return pesudo_bboxes, pesudo_scores, pesudo_labels

    def _analysis_recall_sim_thr(self, img_feat, unreliable_bboxes, unreliable_scores, unreliable_labels, true_bboxes, true_labels):
        # overlaps = [cal_bboxes_overlaps(u_bboxes, t_bboxes) for u_bboxes, t_bboxes in zip(unreliable_bboxes, true_bboxes)]
        # # print(overlaps)
        # ind_list = []
        # overlap_list = []
        # for i in range(len(overlaps)):
        #     overlap, ind = overlaps[i]
        #     ind = ind.long()
        #     overlap_list.append(overlap)
        #     ind_list.append(ind)
        #     # print(overlap, true_labels[i][ind], unreliable_labels[i])
        # overlap_list = torch.cat(overlap_list)
        # ind_list = torch.cat(ind_list)
        # unreliable_labels = torch.cat(unreliable_labels)
        # true_labels = torch.cat(true_labels)

        roi_list = []
        count_list = []
        for i in range(len(unreliable_bboxes)):
            count_list.append(len(unreliable_bboxes[i]))
            rois = torch.cat([unreliable_bboxes[i], unreliable_scores[i].reshape(-1, 1)], dim=1)
            img_inds = rois.new_full((rois.size(0), 1), i)
            rois = torch.cat([img_inds, rois[:, :4]], dim=-1)
            roi_list.append(rois)
        roi_list = torch.cat(roi_list, 0)
        # print(roi_list.shape)
        iou_pred = self.teacher.roi_head.extract_bboxes_features(img_feat, roi_list)

        # sim = torch.cosine_similarity(bboxes_feat.unsqueeze(1), self.center_features.unsqueeze(0),
        #                               dim=-1).detach().cpu()
        # for i in range(len(ind_list)):
        #     print(overlap_list[i], sim[i][unreliable_labels[i]], unreliable_labels[i], true_labels[ind_list[i]])
        self._add_analysis_recall_info(unreliable_bboxes,
                                       unreliable_scores,
                                       unreliable_labels,
                                       true_bboxes,
                                       true_labels,
                                       iou_pred)





    def _recall_rois(self, img_feat, unreliable_bboxes, unreliable_scores, unreliable_labels, true_labels=None):

        # roi_list = []
        # count_list = []
        # for i in range(len(unreliable_bboxes)):
        #     count_list.append(len(unreliable_bboxes[i]))
        #     rois = torch.cat([unreliable_bboxes[i], unreliable_scores[i].reshape(-1, 1)], dim=1)
        #     img_inds = rois.new_full((rois.size(0), 1), i)
        #     rois = torch.cat([img_inds, rois[:, :4]], dim=-1)
        #     roi_list.append(rois)
        # roi_list = torch.cat(roi_list, 0)
        # bboxes_feat, iou_pred = self.teacher.roi_head.extract_bboxes_features(img_feat, roi_list)
        # bboxes_feat = norm_tensor(nn.functional.normalize(bboxes_feat, dim=0))

        # sim = torch.cosine_similarity(bboxes_feat.unsqueeze(1), self.center_features.unsqueeze(0),
        #                                    dim=-1)
        # label_ind = torch.cat(unreliable_labels)

        # sim_list = []
        # for i in range(len(sim)):
        #     sim_list.append(sim[i][label_ind[i]])
        # sim_list = torch.tensor(sim_list).to(img_feat[0].device)

        # recall_masks1 = torch.split(sim_list > 0.7, count_list)
        # iou_masks = torch.split(torch.sigmoid(iou_pred.reshape(-1)) > 0.5, count_list)


        cls_pro = self._cal_recall_pro()
        recall_masks2 = []
        recall_pros = []
        #
        for labels, scores in zip(unreliable_labels, unreliable_scores):
            sample_pro = (1. - cls_pro[labels]) * (scores / self.pesudo_thr)
            # print(sample_pro)
            # sample_pro = (1. - cls_pro[labels]) * scores
            mask = torch.rand(len(sample_pro)).to(sample_pro.device) < sample_pro
            recall_masks2.append(mask.to(img_feat[0].device))
            recall_pros.append(sample_pro)
        #     # recall_bboxes.append()
        self._add_recall_pro(recall_pros)

        # recall_masks = [(mask1 & mask2 & mask3) for (mask1, mask2, mask3) in zip(recall_masks1, recall_masks2, iou_masks)]
        # recall_masks = [(mask1 & mask2) for (mask1, mask2) in zip(recall_masks2, iou_masks)]

        bboxes = [recall_bboxes[mask] for (recall_bboxes, mask) in zip(unreliable_bboxes, recall_masks2)]
        labels = [recall_labels[mask] for (recall_labels, mask) in zip(unreliable_labels, recall_masks2)]
        scores = [recall_scores[mask] for (recall_scores, mask) in zip(unreliable_scores, recall_masks2)]

        return bboxes, labels, scores




    def _cal_recall_pro(self):
        return torch.div(self.cls_count[0], torch.sum(self.cls_count[0]))

    def parse_informativeness(self, weak_unsup):
        from mmdet.core import (bbox2roi, roi2bbox)
        import torch.nn.functional as F

        self.teacher.eval()
        with torch.no_grad():
            x = self.teacher.extract_feat(weak_unsup['img'])
            proposal_list = self.teacher.rpn_head.simple_test_rpn(x, weak_unsup['img_metas'])
            rois = bbox2roi(proposal_list)

            if rois.shape[0] == 0:
                batch_size = len(proposal_list)
                det_bbox = rois.new_zeros(0, 5)
                det_label = rois.new_zeros((0,), dtype=torch.long)
                # There is no proposal in the whole batch
                return [det_bbox] * batch_size, [det_label] * batch_size

            bbox_results = self.teacher.roi_head._bbox_forward(x, rois)
            img_shapes = tuple(meta['img_shape'] for meta in weak_unsup['img_metas'])
            scale_factors = tuple(meta['scale_factor'] for meta in weak_unsup['img_metas'])

            cls_score = bbox_results['cls_score']
            # bbox_pred = bbox_results['bbox_pred']
            # num_proposals_per_img = tuple(len(p) for p in proposal_list)
            # rois = rois.split(num_proposals_per_img, 0)
            # cls_score = cls_score.split(num_proposals_per_img, 0)

    #         # # some detector with_reg is False, bbox_pred will be None
    #         if bbox_pred is not None:
    #             # TODO move this to a sabl_roi_head
    #             # the bbox prediction of some detectors like SABL is not Tensor
    #             if isinstance(bbox_pred, torch.Tensor):
    #                 bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
    #             else:
    #                 bbox_pred = self.bbox_head.bbox_pred_split(
    #                     bbox_pred, num_proposals_per_img)
    #         else:
    #             bbox_pred = (None,) * len(proposal_list)
    #         #
    #         # apply bbox post-processing to each image individually
    #         det_bboxes = []
    #         det_labels = []
    #         for i in range(len(proposal_list)):
    #             if rois[i].shape[0] == 0:
    #                 # There is no proposal in the single image
    #                 det_bbox = rois[i].new_zeros(0, 5)
    #                 det_label = rois[i].new_zeros((0,), dtype=torch.long)
    #                 if rcnn_test_cfg is None:
    #                     det_bbox = det_bbox[:, :4]
    #                     det_label = rois[i].new_zeros(
    #                         (0, self.bbox_head.fc_cls.out_features))
    #
    #             else:
    #                 det_bbox, det_label = self.bbox_head.get_bboxes(
    #                     rois[i],
    #                     cls_score[i],
    #                     bbox_pred[i],
    #                     img_shapes[i],
    #                     scale_factors[i],
    #                     rescale=rescale,
    #                     cfg=rcnn_test_cfg)
    #             det_bboxes.append(det_bbox)
    #             det_labels.append(det_label)
    #         return det_bboxes, det_labels

            # split batch bbox prediction back to each image
            cls_scores = F.softmax(cls_score, dim=-1)
            entropy = (- cls_scores * torch.log2(cls_scores)).sum(dim=1)
            print(cls_scores.shape, entropy.shape, rois)
            print(torch.max(cls_scores, dim=1).values, entropy)
            # bbox_pred = bbox_results['bbox_pred']
            # num_proposals_per_img = tuple(len(p) for p in proposals)
            # rois = rois.split(num_proposals_per_img, 0)
            # cls_score = cls_score.split(num_proposals_per_img, 0)




    def _gen_unreliable_pseudo_labels(self, weak_unsup, strong_unsup):
        self.teacher.eval()
        # self.parse_informativeness(weak_unsup)

        strong_unsup.update({"gt_bboxes_true": [bboxes for bboxes in strong_unsup["gt_bboxes"]]})
        strong_unsup.update({"gt_labels_true": [labels for labels in strong_unsup["gt_labels"]]})

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


            # self.student.eval()
            s_feat = self.student.extract_feat(weak_unsup['img'])
            # self.student.train()

        # filter bboxes using thr
        reliable_bboxes = [bboxes[scores >= self.pesudo_thr] for (bboxes, scores) in zip(result_bboxes, result_scores)]
        reliable_labels = [labels[scores >= self.pesudo_thr] for (labels, scores) in zip(det_labels, result_scores)]
        reliable_scores = [scores1[scores2 >= self.pesudo_thr] for (scores1, scores2) in zip(result_scores, result_scores)]
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
        
        if self.print_pesudo_summary:
            strong_unsup.update({"reliable_bboxes": [bboxes.cpu().clone() for bboxes in reliable_bboxes]})
            strong_unsup.update({"reliable_labels": [labels.cpu().clone() for labels in reliable_labels]})

            strong_unsup.update({"recall_bboxes": [bboxes.cpu().clone() for bboxes in unreliable_bboxes]})
            strong_unsup.update({"recall_labels": [labels.cpu().clone() for labels in unreliable_labels]})


        if self.print_pesudo_summary:
            self._analysis_recall_sim_thr(feat, unreliable_bboxes, unreliable_scores, unreliable_labels, [bboxes for bboxes in weak_unsup["gt_bboxes"]], [labels for labels in weak_unsup["gt_labels"]])


        # if self.is_label_memory is True:
        #     self._ori_update_label_memory(reliable_bboxes, reliable_labels, reliable_scores, weak_unsup)
        #     # self._update_label_memory_mean(gt_bboxes, gt_labels, gt_scores, weak_unsup)
        #     gt_bboxes, gt_scores, gt_labels = self._get_pesudo_from_memory(weak_unsup, reliable_bboxes[0].device)

        if self.is_recall is True:
            # recall_bboxes, recall_labels, recall_scores = self._recall_rois(feat, unreliable_bboxes, unreliable_scores, unreliable_labels)
            recall_bboxes = unreliable_bboxes
            recall_labels = unreliable_labels
            recall_scores = unreliable_scores


            # t_soft_labels, s_soft_labels = self._soft_label(feat, s_feat, recall_bboxes, recall_labels, weak_unsup)
            # strong_unsup.update({"gt_soft_labels": gt_soft_labels})
            # mask_list, mix_pro_list, t_score_list, s_score_list
            filter_mask, mix_pro_list, t_score_list, s_score_list = self._t_s_filter_bboxes(feat, s_feat, recall_bboxes)
            # _, mix_reliable_pro_list, t_reliable_score_list, s_reliable_score_list = self._t_s_filter_bboxes(feat, s_feat, reliable_bboxes)
            recall_bboxes = [bboxes[mask] for (bboxes, mask) in zip(recall_bboxes, filter_mask)]
            recall_labels = [labels[mask].type(torch.long) for (labels, mask) in zip(recall_labels, filter_mask)]
            recall_scores = [scores[mask] for (scores, mask) in zip(recall_scores, filter_mask)]
            # mix_soft_labels = [soft_labels[mask] for (soft_labels, mask) in zip(mix_soft_labels, filter_mask)]
            # strong_unsup.update({"recall_soft_labels": mix_soft_labels})

            unreliable_bboxes = [bboxes[~mask] for (bboxes, mask) in zip(unreliable_bboxes, filter_mask)]
            unreliable_labels = [labels[~mask].type(torch.long) for (labels, mask) in zip(unreliable_labels, filter_mask)]
            unreliable_scores = [scores[~mask] for (scores, mask) in zip(unreliable_scores, filter_mask)]
            unreliabel_mix_pro = [pro[~mask] for (pro, mask) in zip(mix_pro_list, filter_mask)]
            strong_unsup.update({"unreliable_mix_pro": unreliabel_mix_pro})



            gt_bboxes = [torch.cat([bboxes1, bboxes2], dim=0) for (bboxes1, bboxes2) in
                         zip(reliable_bboxes, recall_bboxes)]
            gt_labels = [torch.cat([labels1, labels2]) for (labels1, labels2) in
                         zip(reliable_labels, recall_labels)]
            gt_scores = [torch.cat([scores1, scores2]) for (scores1, scores2) in
                         zip(reliable_scores, recall_scores)]

        else:
            _, mix_pro_list, t_score_list, s_score_list = self._t_s_filter_bboxes(feat, s_feat, unreliable_bboxes)
            _, mix_reliable_pro_list, t_reliable_score_list, s_reliable_score_list = self._t_s_filter_bboxes(feat,
                                                                                                             s_feat,
                                                                                                             reliable_bboxes)

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
        # print(strong_unsup['gt_soft_labels'])
        if self.print_pesudo_summary:
            if self.is_recall:
                self._add_summary_bboxes([{"bboxes":torch.cat([bboxes, torch.unsqueeze(scores, dim=1)], dim=1).detach().cpu().numpy(),
                                           "labels":labels.detach().cpu().numpy(),
                                           "recall_bboxes": r_bboxes,
                                           "recall_labels": r_labels,
                                           "t_soft_labels":t_soft_label.detach().cpu().numpy(),
                                           "s_soft_labels":s_soft_label.detach().cpu().numpy(),
                                           "filter_mask": mask.detach().cpu().numpy(),
                                           "reliable_bboxes": re_bboxes,
                                           "reliable_labels": re_labels,
                                           "t_reliable_soft_labels": t_re_soft_label.detach().cpu().numpy(),
                                           "s_reliable_soft_labels": s_re_soft_label.detach().cpu().numpy(),
                                           } for
                                          (bboxes, labels, scores, r_bboxes, r_labels, t_soft_label, s_soft_label, mask,
                                           re_bboxes, re_labels, t_re_soft_label, s_re_soft_label) in
                                          zip(strong_unsup["gt_bboxes"], strong_unsup["gt_labels"], strong_unsup["gt_scores"],
                                              strong_unsup["recall_bboxes"], strong_unsup["recall_labels"], t_score_list, s_score_list, filter_mask,
                                              strong_unsup["reliable_bboxes"], strong_unsup["reliable_labels"], t_reliable_score_list, s_reliable_score_list
                                              )],
                                         [{"bboxes":bboxes.detach().cpu().numpy(),
                                           "labels":labels.detach().cpu().numpy()} for (bboxes, labels) in zip(strong_unsup["gt_bboxes_true"], strong_unsup["gt_labels_true"])])
                self._add_unreliable_bboxes([{"bboxes":torch.cat([bboxes, torch.unsqueeze(scores, dim=1)], dim=1).detach().cpu().numpy(),
                                              "labels":labels.detach().cpu().numpy()} for (bboxes, labels, scores) in zip(strong_unsup["unreliable_bboxes"], strong_unsup["unreliable_labels"], strong_unsup["unreliable_scores"])])
            else:
                self._add_summary_bboxes(
                    [{"bboxes": torch.cat([bboxes, torch.unsqueeze(scores, dim=1)], dim=1).detach().cpu().numpy(),
                      "labels": labels.detach().cpu().numpy(),
                      "recall_bboxes": r_bboxes,
                      "recall_labels": r_labels,
                      "t_soft_labels": t_soft_label.detach().cpu().numpy(),
                      "s_soft_labels": s_soft_label.detach().cpu().numpy(),
                      "reliable_bboxes": re_bboxes,
                      "reliable_labels": re_labels,
                      "t_reliable_soft_labels": t_re_soft_label.detach().cpu().numpy(),
                      "s_reliable_soft_labels": s_re_soft_label.detach().cpu().numpy(),
                      "gts_bboxes":gts_bboxes,
                      "gts_labels":gts_labels,
                      } for
                     (bboxes, labels, scores,
                      r_bboxes, r_labels, t_soft_label, s_soft_label,
                      re_bboxes, re_labels, t_re_soft_label, s_re_soft_label,
                      gts_bboxes, gts_labels) in
                     zip(strong_unsup["gt_bboxes"], strong_unsup["gt_labels"], strong_unsup["gt_scores"],
                         strong_unsup["recall_bboxes"], strong_unsup["recall_labels"], t_score_list, s_score_list,
                         strong_unsup["reliable_bboxes"], strong_unsup["reliable_labels"], t_reliable_score_list, s_reliable_score_list,
                         weak_unsup["gt_bboxes"], weak_unsup["gt_labels"])],
                    [{"bboxes": bboxes.detach().cpu().numpy(),
                      "labels": labels.detach().cpu().numpy()} for (bboxes, labels) in
                     zip(strong_unsup["gt_bboxes_true"], strong_unsup["gt_labels_true"])])
                self._add_unreliable_bboxes(
                    [{"bboxes": torch.cat([bboxes, torch.unsqueeze(scores, dim=1)], dim=1).detach().cpu().numpy(),
                      "labels": labels.detach().cpu().numpy()} for (bboxes, labels, scores) in
                     zip(strong_unsup["unreliable_bboxes"], strong_unsup["unreliable_labels"],
                         strong_unsup["unreliable_scores"])])

        return strong_unsup, weak_unsup


    def _soft_label(self, t_feat, s_feat, gt_bboxes, gt_labels, weak_unsup):
        from mmdet.core import bbox2roi
        import torch.nn.functional as F

        # feat = self.teacher.extract_feat(weak_unsup['img'])
        # extract proposal regions
        # proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, weak_unsup['img_metas'])
        # extract det results from the detector
        rois = bbox2roi(gt_bboxes)

        # if rois.shape[0] == 0:
        #     batch_size = len(gt_bboxes)
        #     det_bbox = rois.new_zeros(0, 5)
        #     det_label = rois.new_zeros((0,), dtype=torch.long)
        #     return [det_bbox] * batch_size, [det_label] * batch_size

        num_proposals_per_img = [len(p) for p in gt_bboxes]
        # rois = rois.split(num_proposals_per_img, 0)

        t_bbox_results = self.teacher.roi_head._bbox_forward(t_feat, rois)
        t_cls_score = F.softmax(t_bbox_results['cls_score'], dim=-1)


        # if self.is_soft_label_sharpen is True:
        #     t_cls_score = t_cls_score ** (1 / 0.5)
        #     t_cls_score = t_cls_score / t_cls_score.sum(dim=1, keepdim=True)
        ind = 0
        t_score_list = []
        for i in range(len(num_proposals_per_img)):
            t_score_list.append(t_cls_score[ind:ind+num_proposals_per_img[i]])
            ind += num_proposals_per_img[i]

        s_bbox_results = self.student.roi_head._bbox_forward(s_feat, rois)
        s_cls_score = F.softmax(s_bbox_results['cls_score'], dim=-1)
        s_score_list = []
        ind = 0
        for i in range(len(num_proposals_per_img)):
            s_score_list.append(s_cls_score[ind:ind+num_proposals_per_img[i]])
            ind += num_proposals_per_img[i]
        # print(gt_labels)
        # print(t_cls_score.shape)
        # t_cls_score = t_cls_score.split(num_proposals_per_img, 0)



        return t_score_list, s_score_list

    def _t_s_filter_bboxes(self,  t_feat, s_feat, gt_bboxes):
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
            top_thr = torch.topk(mix_pro, k=2, dim=1).values[:, :2]
            obj_mask = (top_thr[:, 0] - top_thr[:, 1]) > self.theta2
            # print(mix_pro)
            # print(mask)
            mask_list.append(bg_mask & obj_mask)
            mix_pro_list.append(mix_pro)
            # mix_pro = mix_pro ** (1 / 0.5)
            # mix_pro = mix_pro / mix_pro.sum(dim=1, keepdim=True)
            # mix_soft_labels.append(mix_pro)
        return mask_list, mix_pro_list, t_score_list, s_score_list

    def _nms_bboxes(self, gt_bboxes, gt_labels, gt_scores):
        return_bboxes = []
        return_labels = []
        return_scores = []
        for i in range(len(gt_bboxes)):

            bboxes = gt_bboxes[i]
            scores = gt_scores[i]
            labels = gt_labels[i]
            nms_scores = torch.zeros((len(gt_bboxes[i])), self.n_cls + 1).to(bboxes.device)
            for j in range(len(scores)):
                # print(labels[i])
                nms_scores[j][labels[j].type(torch.long)] = scores[j]
            # print(nms_scores)

            nms_bboxes, nms_labels, inds = multiclass_nms(bboxes,
                                                          nms_scores,
                                                          self.unreliable_thr,
                                                          self.train_cfg.region_bg_nms_cfg,
                                                          return_inds=True)
            return_bboxes.append(nms_bboxes[:, :4])
            return_labels.append(nms_labels)
            return_scores.append(nms_bboxes[:, 4])
        return return_bboxes, return_labels, return_scores
        #     # print(nms_bboxes)
        #     # print(nms_labels)
        #     update_pesudo = torch.cat([nms_bboxes, torch.unsqueeze(nms_labels, dim=1)], dim=1).cpu()

    def _compute_student_sup_negtive_loss(self, sup_data):
        self.student.train()
        # print(sup_data)
        feats = self.student.extract_feat(sup_data["img"])
        losses = dict()
        proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                  self.student.test_cfg.rpn)
        rpn_losses, proposal_list = self.student.rpn_head.forward_train(
            feats,
            sup_data["img_metas"],
            sup_data["gt_bboxes"],
            gt_labels=None,
            gt_bboxes_ignore=None,
            proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)

        # num_imgs = len(sup_data["img_metas"])
        # gt_bboxes_ignore = [None for _ in range(num_imgs)]
        # sampling_results = []
        # assign_results = []
        # for i in range(num_imgs):
        #     assign_result = self.student.roi_head.bbox_assigner.assign(
        #         proposal_list[i], sup_data["gt_bboxes"][i], gt_bboxes_ignore[i],
        #         sup_data["gt_labels"][i])
        #     assign_results.append(assign_result)
        #     sampling_result = self.student.roi_head.bbox_sampler.sample(
        #         assign_result,
        #         proposal_list[i],
        #         sup_data["gt_bboxes"][i],
        #         sup_data["gt_labels"][i],
        #         feats=[lvl_feat[i][None] for lvl_feat in feats])
        #     sampling_results.append(sampling_result)

        # bbox_results = self.student.roi_head._bbox_forward_train(feats,
        #                                                          sampling_results,
        #                                                          sup_data["gt_bboxes"],
        #                                                          sup_data["gt_labels"],
        #                                                          sup_data["img_metas"])
        # losses.update(bbox_results['loss_bbox'])
        # bbox_results = self.student.roi_head.kl_bbox_forward_train(feats,
        #                                                          sampling_results,
        #                                                          sup_data["gt_bboxes"],
        #                                                          sup_data["gt_labels"],
        #                                                          sup_data["img_metas"])
        if self.is_iou_loss is True:
            roi_losses = self.student.roi_head.iou_forward_train(feats,
                                                         sup_data["img_metas"],
                                                         proposal_list,
                                                         sup_data["gt_bboxes"],
                                                         sup_data["gt_labels"])
        else:
            roi_losses = self.student.roi_head.forward_train(feats, sup_data["img_metas"], proposal_list,
                                                        sup_data["gt_bboxes"], sup_data["gt_labels"])
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
            if self.print_pesudo_summary:
                self._add_neg_loss_info(roi_masks,
                                        [bboxes.detach().cpu().numpy() for bboxes in strong_unsup["unreliable_bboxes"]],
                                        bboxes_scores)

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
                                                       # gt_bboxes_ignore=strong_unsup["unreliable_bboxes"])
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

            if self.print_pesudo_summary is True:
                self._add_iou_info(rois.detach().cpu(),
                                   bbox_weights.detach().cpu())

        else:
            bbox_results = self.student.roi_head._bbox_forward_train(x, sampling_results,
                                                                 gt_bboxes, gt_labels,
                                                                 img_metas)
        losses.update(bbox_results['loss_bbox'])

        return losses


    # @torch.no_grad()
    # def _update_class_centers(self, sup_data, proposals=None, is_sup=True):
    #     self.teacher.eval()
    #     with torch.no_grad():
    #         feat = self.teacher.extract_feat(sup_data['img'])
    #         # print("feat:", feat[0].dtype)
    #         proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, sup_data['img_metas'])
    #
    #         det_bboxes, det_labels = self.teacher.roi_head.simple_test_bboxes(
    #             feat, sup_data['img_metas'], proposal_list, self.teacher.test_cfg.rcnn, rescale=False)
    #         if is_sup is True:
    #             gts_bboxes = sup_data["gt_bboxes"]
    #             gts_labels = sup_data['gt_labels']
    #         else:
    #             gts_bboxes = sup_data["reliable_bboxes"]
    #             gts_labels = sup_data['reliable_labels']
    #
    #         projector_bg_feats, projector_gt_feats, bg_feats, gt_feats = self.teacher.roi_head.extract_projector_features(
    #             feat, det_bboxes, gts_bboxes, self.train_cfg, self.train_cfg.region_bg_score_thr)
    #
    #     bg_feats = norm_tensor(nn.functional.normalize(bg_feats, dim=0))
    #     gt_feats = norm_tensor(nn.functional.normalize(gt_feats, dim=0))
    #
    #
    #     self._update_bg_centers(bg_feats, img_num=len(proposal_list))
    #     self._update_fg_centers(gt_feats, torch.cat(gts_labels, dim=0), img_num=len(proposal_list))
    #
    #     if torch.sum(torch.isnan(self.center_features)) != 0:
    #         print(self.center_features)
    #         print('centers has nan')
    #         exit()
    #
    # @torch.no_grad()
    # def _update_reliable_queues(self, sup_data, proposals=None):
    #     self.teacher.eval()
    #     with torch.no_grad():
    #         feat = self.teacher.extract_feat(sup_data['img'])
    #         proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, sup_data['img_metas'])
    #
    #         det_bboxes, det_labels = self.teacher.roi_head.simple_test_bboxes(
    #             feat, sup_data['img_metas'], proposal_list, self.teacher.test_cfg.rcnn, rescale=False)
    #         projector_bg_feats, projector_gt_feats, bg_feats, gt_feats = self.teacher.roi_head.extract_projector_features(
    #             feat, det_bboxes, sup_data["gt_bboxes"], self.train_cfg, self.train_cfg.region_bg_score_thr)
    #         # print('det bboxes:', det_bboxes)
    #         # print('proposals:', proposal_list)
    #
    #     #
    #     projector_bg_feats = norm_tensor(nn.functional.normalize(projector_bg_feats, dim=0))
    #     projector_gt_feats = norm_tensor(nn.functional.normalize(projector_gt_feats, dim=0))
    #     # bg_feats = norm_tensor(nn.functional.normalize(bg_feats, dim=0))
    #     # gt_feats = norm_tensor(nn.functional.normalize(gt_feats, dim=0))
    #
    #     # self._neg_dequeue_and_enqueue(bg_feats, img_num=len(proposal_list))
    #     # self._pos_dequeue_and_enqueue(gt_feats, torch.cat(sup_data["gt_labels"], dim=0), img_num=len(proposal_list))
    #     self._projector_neg_dequeue_and_enqueue(projector_bg_feats, img_num=len(proposal_list))
    #     self._projector_pos_dequeue_and_enqueue(projector_gt_feats, torch.cat(sup_data["gt_labels"], dim=0), img_num=len(proposal_list))
    #
    #
    # @torch.no_grad()
    # def _neg_dequeue_and_enqueue(self, keys, img_num=0):
    #     """Update neg region queue."""
    #
    #     broadcast_keys = torch.zeros(img_num * self.region_bg_max_num, self.bbox_feat_dim).to(keys.device)
    #     broadcast_mask = torch.zeros(img_num * self.region_bg_max_num).to(keys.device)
    #     broadcast_keys[:len(keys)] = keys
    #     broadcast_mask[:len(keys)] = 1.0
    #     bg_keys = gather_same_shape_tensors(broadcast_keys)
    #     bg_masks = gather_same_shape_tensors(broadcast_mask)
    #     keys = bg_keys[bg_masks == 1.0]
    #
    #     update_size = keys.shape[0]
    #     ptr = int(self.neg_queue_ptr)
    #
    #     if (ptr + update_size) > self.neg_queue_len:
    #         len_11 = self.neg_queue_len - ptr
    #         self.neg_queue[:, ptr:] = keys[:len_11].transpose(0, 1)
    #         ptr = (ptr + update_size) % self.neg_queue_len
    #         self.neg_queue[:, :ptr] = keys[len_11:].transpose(0, 1)
    #     else:
    #         self.neg_queue[:, ptr:ptr + update_size] = keys.transpose(0, 1)
    #         ptr = (ptr + update_size) % self.neg_queue_len  # move pointer
    #     #
    #     self.neg_queue_ptr[0] = ptr
    #
    # @torch.no_grad()
    # def _pos_dequeue_and_enqueue(self, keys, gt_labels, img_num=0):
    #     """Update pos region queue."""
    #     broadcast_keys = torch.zeros(img_num * self.region_fg_max_num, self.bbox_feat_dim).to(keys.device)
    #     broadcast_labels = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
    #     broadcast_mask = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
    #     # broadcast_mask[:]
    #     broadcast_keys[:len(keys)] = keys
    #     broadcast_labels[:len(keys)] = gt_labels
    #     broadcast_mask[:len(keys)] = 1.0
    #     # print(broadcast_mask)
    #     fg_keys = gather_same_shape_tensors(broadcast_keys)
    #     fg_masks = gather_same_shape_tensors(broadcast_mask)
    #     fg_labels = gather_same_shape_tensors(broadcast_labels)
    #     keys = fg_keys[fg_masks == 1.0]
    #     gts_labels = fg_labels[fg_masks == 1.0]
    #
    #     for i in range(self.n_cls):
    #         update_size = keys[gts_labels == i].shape[0]
    #         ptr = int(self.pos_queue_ptr[i])
    #
    #         if (ptr + update_size) > self.pos_queue_len:
    #             len_11 = self.pos_queue_len - ptr
    #             self.pos_queue[i, :, ptr:] = keys[gts_labels == i][:len_11].transpose(0, 1)
    #             ptr = (ptr + update_size) % self.pos_queue_len
    #             self.pos_queue[i, :, :ptr] = keys[gts_labels == i][len_11:].transpose(0, 1)
    #         else:
    #             self.pos_queue[i, :, ptr:ptr + update_size] = keys[gts_labels == i].transpose(0, 1)
    #             ptr = (ptr + update_size) % self.pos_queue_len  # move pointer
    #         #
    #         self.pos_queue_ptr[i][0] = ptr
    #
    # @torch.no_grad()
    # def _projector_neg_dequeue_and_enqueue(self, keys, img_num=0):
    #     """Update neg region queue."""
    #
    #     broadcast_keys = torch.zeros(img_num * self.region_bg_max_num, self.feat_dim).to(keys.device)
    #     broadcast_mask = torch.zeros(img_num * self.region_bg_max_num).to(keys.device)
    #     broadcast_keys[:len(keys)] = keys
    #     broadcast_mask[:len(keys)] = 1.0
    #     bg_keys = gather_same_shape_tensors(broadcast_keys)
    #     bg_masks = gather_same_shape_tensors(broadcast_mask)
    #     keys = bg_keys[bg_masks == 1.0]
    #
    #     update_size = keys.shape[0]
    #     ptr = int(self.projector_neg_queue_ptr)
    #
    #     if (ptr + update_size) > self.neg_queue_len:
    #         len_11 = self.neg_queue_len - ptr
    #         self.projector_neg_queue[:, ptr:] = keys[:len_11].transpose(0, 1)
    #         ptr = (ptr + update_size) % self.neg_queue_len
    #         self.projector_neg_queue[:, :ptr] = keys[len_11:].transpose(0, 1)
    #     else:
    #         self.projector_neg_queue[:, ptr:ptr + update_size] = keys.transpose(0, 1)
    #         ptr = (ptr + update_size) % self.neg_queue_len  # move pointer
    #     #
    #     self.projector_neg_queue_ptr[0] = ptr
    #
    # @torch.no_grad()
    # def _projector_pos_dequeue_and_enqueue(self, keys, gt_labels, img_num=0):
    #     """Update pos region queue."""
    #     broadcast_keys = torch.zeros(img_num * self.region_fg_max_num, self.feat_dim).to(keys.device)
    #     broadcast_labels = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
    #     broadcast_mask = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
    #     # broadcast_mask[:]
    #     broadcast_keys[:len(keys)] = keys
    #     broadcast_labels[:len(keys)] = gt_labels
    #     broadcast_mask[:len(keys)] = 1.0
    #     # print(broadcast_mask)
    #     fg_keys = gather_same_shape_tensors(broadcast_keys)
    #     fg_masks = gather_same_shape_tensors(broadcast_mask)
    #     fg_labels = gather_same_shape_tensors(broadcast_labels)
    #     keys = fg_keys[fg_masks == 1.0]
    #     gts_labels = fg_labels[fg_masks == 1.0]
    #
    #     for i in range(self.n_cls):
    #         update_size = keys[gts_labels == i].shape[0]
    #         ptr = int(self.projector_pos_queue_ptr[i])
    #
    #         if (ptr + update_size) > self.pos_queue_len:
    #             len_11 = self.pos_queue_len - ptr
    #             self.projector_pos_queue[i, :, ptr:] = keys[gts_labels == i][:len_11].transpose(0, 1)
    #             ptr = (ptr + update_size) % self.pos_queue_len
    #             self.projector_pos_queue[i, :, :ptr] = keys[gts_labels == i][len_11:].transpose(0, 1)
    #         else:
    #             self.projector_pos_queue[i, :, ptr:ptr + update_size] = keys[gts_labels == i].transpose(0, 1)
    #             ptr = (ptr + update_size) % self.pos_queue_len  # move pointer
    #         #
    #         self.projector_pos_queue_ptr[i][0] = ptr

    @torch.no_grad()
    def _cal_ignore_unreliable_bboxes(self, gt_bboxes, unreliable_bboxes):
        iou_results = [self.iou_calculator(gts, unrlia, mode='iof') for gts, unrlia in
                       zip(gt_bboxes, unreliable_bboxes)]

        masks = [overlaps.max(dim=0).values < 0.3 if overlaps.shape[0] != 0 else torch.ones(overlaps.shape[1],dtype=torch.bool).to(overlaps.device)
                 for overlaps in iou_results]

        result_bbox = [unralia[mask] for unralia, mask in zip(unreliable_bboxes, masks)]
        return result_bbox

    # @torch.no_grad()
    # def _update_fg_centers(self, keys, gt_labels, img_num=0):
    #     alpha = min(1 - 1 / (self.ema_iteration + 1), self.ema_decay)
    #
    #     broadcast_keys = torch.zeros(img_num * self.region_fg_max_num, self.feat_dim).to(keys.device)
    #     broadcast_labels = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
    #     broadcast_mask = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
    #     # broadcast_mask[:]
    #     broadcast_keys[:len(keys)] = keys
    #     broadcast_labels[:len(keys)] = gt_labels
    #     broadcast_mask[:len(keys)] = 1.0
    #     # print(broadcast_mask)
    #     fg_keys = gather_same_shape_tensors(broadcast_keys)
    #     fg_masks = gather_same_shape_tensors(broadcast_mask)
    #     fg_labels = gather_same_shape_tensors(broadcast_labels)
    #     keys = fg_keys[fg_masks == 1.0]
    #     gts_labels = fg_labels[fg_masks == 1.0]
    #     for i in range(self.n_cls):
    #         cls_keys = keys[gts_labels == i]
    #         if len(cls_keys) != 0:
    #             self.center_features[i] = self.center_features[i] * alpha + torch.mean(keys[gts_labels == i], dim=0) * (1 - alpha)
    #
    # @torch.no_grad()
    # def _update_bg_centers(self, keys, img_num=0):
    #     alpha = min(1 - 1 / (self.ema_iteration + 1), self.ema_decay)
    #
    #     broadcast_keys = torch.zeros(img_num * self.region_bg_max_num, self.feat_dim).to(keys.device)
    #     broadcast_mask = torch.zeros(img_num * self.region_bg_max_num).to(keys.device)
    #     broadcast_keys[:len(keys)] = keys
    #     broadcast_mask[:len(keys)] = 1.0
    #     bg_keys = gather_same_shape_tensors(broadcast_keys)
    #     bg_masks = gather_same_shape_tensors(broadcast_mask)
    #     keys = bg_keys[bg_masks == 1.0]
    #     if len(keys) != 0:
    #         self.center_features[self.n_cls] = self.center_features[self.n_cls] * alpha + torch.mean(keys, dim=0) * (1 - alpha)
