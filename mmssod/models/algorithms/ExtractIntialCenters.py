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

@DETECTORS.register_module()
class ExtractIntialCenters(BurnInTSModel):
    def __init__(self, teacher: dict, student: dict, train_cfg=None, test_cfg=None, n_cls=20):
        super().__init__(teacher, student, train_cfg, test_cfg, n_cls)
        print("train_cfg:", train_cfg)

        self.train_cfg = train_cfg

        # queues settings
        self.feat_dim = train_cfg.get("feat_dim", 1024)
        self.bbox_feat_dim = train_cfg.get("bbox_feat_dim", 1024)
        self.pos_queue_len = train_cfg.get("pos_queue_len", 100)
        self.neg_queue_len = train_cfg.get("neg_queue_len", 65536)
        self.region_bg_max_num = train_cfg.get("region_bg_max_num", 10)
        self.region_fg_max_num = train_cfg.get("region_fg_max_num", 80)

        # self.cls_count = torch.zeros(n_cls).to(self.device)
        self.register_buffer('cls_count', torch.zeros(1, n_cls))
        self._init_cls_count()
        # store the keys of pos/neg regions
        # self.register_buffer('pos_queue', torch.zeros(self.n_cls ,self.bbox_feat_dim, self.pos_queue_len))
        # self.register_buffer('pos_queue_ptr', torch.zeros((self.n_cls, 1), dtype=torch.long))
        # self.register_buffer('neg_queue', torch.zeros(self.bbox_feat_dim, self.neg_queue_len))
        # self.register_buffer('neg_queue_ptr', torch.zeros(1, dtype=torch.long))

        # self.register_buffer('projector_pos_queue', torch.zeros(self.n_cls, self.feat_dim, self.pos_queue_len))
        # self.register_buffer('projector_pos_queue_ptr', torch.zeros((self.n_cls, 1), dtype=torch.long))
        # self.register_buffer('projector_neg_queue', torch.zeros(self.feat_dim, self.neg_queue_len))
        # self.register_buffer('projector_neg_queue_ptr', torch.zeros(1, dtype=torch.long))

        self.register_buffer('center_features', torch.zeros(self.n_cls + 1, self.feat_dim))

        self.unreliable_thr = train_cfg.get("unreliable_thr", 0.3)
        self.neg_thr = train_cfg.get("neg_thr", 0.0005)
        self.is_region_est = train_cfg.get("is_region_est", False)
        self.is_neg_loss = train_cfg.get("is_neg_loss", False)
        self.is_sup_neg_loss = train_cfg.get("is_sup_neg_loss", False)
        self.contrast_loss_weight = train_cfg.get("contrast_loss_weight", 1.0)
        self.sup_contrast_loss_weight = train_cfg.get("sup_contrast_loss_weight", 0.5)

        self.is_ignore_ubreliable = train_cfg.get("ignore_ubreliable", False)
        self.is_recall = train_cfg.get("is_recall", False)
        self.is_weight_norm = train_cfg.get("is_weight_norm", True)

        self.is_label_memory = train_cfg.get("is_label_memory", False)
        self.is_unreliable_label_memory = train_cfg.get("is_unreliable_label_memory", False)
        self.label_memory_bank = {}
        # print(cfg)
        self.thr_label_memory_bank = {}
        # self.register_buffer('label_memory_bank', None)

        # self.label_memory_bank = torch.rand((10552600, 10, 4))

        # self.label_memory_bank = []
        # for i in range(self.label_memory_bank_len):
        #     self.label_memory_bank.append({"pesudo_bboxes":None,
        #                                    "pesudo_labels":None})
        # self.index_table = {}

        # print(self.label_memory_bank)
        self.ema_decay = 0.999
        self.ema_iteration = 0

    def forward_train(self, imgs, img_metas, **kwargs):
        kwargs.update({"img": imgs})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")

        losses = dict()

        # sup_loss = self._compute_student_sup_negtive_loss(data_groups["sup"])
        # self._accumulate_train_number_per_category(data_groups["sup"]["gt_labels"])
        # losses.update(sup_loss)

        # self._update_reliable_queues(data_groups["sup"])
        self._update_class_centers(data_groups["sup"])
        print('---------------------update')
        # weak_unsup, strong_unsup = self._split_unsup_data(data_groups["unsup"])
        #
        # strong_unsup = self._gen_unreliable_pseudo_labels(weak_unsup, strong_unsup)
        #
        # self._accumulate_train_number_per_category(strong_unsup['gt_labels'])
        #
        # unsup_loss = self._compute_student_unsup_negative_loss(weak_unsup, strong_unsup)
        # losses.update(unsup_loss)

        self.ema_iteration += 1


        del  kwargs
        # self._update_class_centers(data_groups["unsup"])
        return {'center_loss':[torch.tensor(0.).to('cuda:0')]}

    def _init_cls_count(self):
        labeled_count = [306, 353, 486, 290, 505, 229, 1250, 376, 798, 259, 215, 510, 362, 339, 4690, 514, 257, 248,
                         297, 324]
        for i in range(self.n_cls):
            self.cls_count[0][i] = labeled_count[i]

    def _accumulate_train_number_per_category(self, label_list):
        # pass
        # print(label_list)
        # []
        tmp_count = torch.zeros((1, self.n_cls)).to(self.cls_count.device)
        for labels in label_list:
            for c in range(self.n_cls):
                tmp_count[0][c] += torch.sum(labels == c)

        out = gather_same_shape_tensors(tmp_count)
        self.cls_count[0] = self.cls_count[0] + torch.sum(out, dim=0)

        # print(self.cls_count)
    def _save_center(self):
        torch.save(self.center_features, '/home/liu/ytx/SS-OD/rcnn_voc_initial_centers.pt')

    def _extract_key(self, filename):
        name = filename.split('/')[-1]
        return name.split('.')[0]

    def _ori_update_label_memory(self, det_bboxes, det_labels, det_scores, weak_unsup):
        pesudo_bboxes = self._transform_bbox(det_bboxes,
                                             [torch.from_numpy(meta["transform_matrix"]).inverse() for meta in
                                              weak_unsup["img_metas"]],
                                             [meta["ori_shape"] for meta in weak_unsup["img_metas"]])

        for bboxes, labels, scores, meta_info in zip(pesudo_bboxes, det_labels, det_scores, weak_unsup['img_metas']):
            key = self._extract_key(meta_info['filename'])
            info = torch.cat([bboxes, torch.unsqueeze(scores, dim=1), torch.unsqueeze(labels, dim=1)], dim=1).cpu()
            gather_objects = [key, info]
            outputs = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(outputs, gather_objects)
            for item in outputs:
                # self.label_memory_bank[item[0]] = item[1]
                if item[0] in self.label_memory_bank.keys():
                    update_pesudo = self.label_memory_bank[item[0]]
                    update_pesudo = torch.cat([update_pesudo, item[1]], dim=0)
                    # print(update_pesudo)
                    nms_scores = torch.zeros((len(update_pesudo)), self.n_cls + 1)
                    scores = update_pesudo[:, 4]
                    labels = update_pesudo[:, 5]
                    for i in range(len(scores)):
                        # print(labels[i])
                        nms_scores[i][labels[i].type(torch.long)] = scores[i]
                    # print(nms_scores)

                    nms_bboxes, nms_labels, inds = multiclass_nms(update_pesudo[:, :4],
                                                                  nms_scores,
                                                                  self.unreliable_thr,
                                                                  self.train_cfg.region_bg_nms_cfg,
                                                                  return_inds=True)
                    # print(nms_bboxes)
                    # print(nms_labels)
                    update_pesudo = torch.cat([nms_bboxes, torch.unsqueeze(nms_labels, dim=1)], dim=1).cpu()
                    # print(update_pesudo)
                    self.label_memory_bank[item[0]] = update_pesudo
                else:
                    self.label_memory_bank[item[0]] = item[1]

    def _update_label_memory(self, det_bboxes, det_labels, det_scores, weak_unsup):
        pesudo_bboxes = self._transform_bbox(det_bboxes,
                                             [torch.from_numpy(meta["transform_matrix"]).inverse() for meta in
                                              weak_unsup["img_metas"]],
                                             [meta["ori_shape"] for meta in weak_unsup["img_metas"]])

        for bboxes, labels, scores, meta_info in zip(pesudo_bboxes, det_labels, det_scores, weak_unsup['img_metas']):
            key = self._extract_key(meta_info['filename'])
            info = torch.cat([bboxes, torch.unsqueeze(scores, dim=1), torch.unsqueeze(labels, dim=1)], dim=1).cpu()
            gather_objects = [key, info]
            outputs = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(outputs, gather_objects)
            for item in outputs:
                self.label_memory_bank[item[0]] = item[1]

    def _get_pesudo_from_memory(self, weak_unsup, device):
        memory_pesudo = [self.label_memory_bank[self._extract_key(meta_info['filename'])] if (
                    self._extract_key(meta_info['filename']) in self.label_memory_bank.keys()) else torch.ones((0, 6))
                         for meta_info in weak_unsup['img_metas']]
        # memory_pesudo = [self.label_memory_bank[self._extract_key(meta_info['filename'])] for meta_info in weak_unsup['img_metas']]
        pesudo_bboxes = [pesudo[:, :4].to(device) for pesudo in memory_pesudo]
        pesudo_scores = [pesudo[:, 4].to(device) for pesudo in memory_pesudo]
        pesudo_labels = [pesudo[:, 5].type(torch.long).to(device) for pesudo in memory_pesudo]
        pesudo_bboxes = self._transform_bbox(pesudo_bboxes,
                                             [torch.from_numpy(meta["transform_matrix"]) for meta in
                                              weak_unsup["img_metas"]],
                                             [meta["img_shape"] for meta in weak_unsup["img_metas"]])

        return pesudo_bboxes, pesudo_scores, pesudo_labels

    def _update_thr_label_memory(self, det_bboxes, det_labels, det_scores, weak_unsup):
        pesudo_bboxes = self._transform_bbox(det_bboxes,
                                             [torch.from_numpy(meta["transform_matrix"]).inverse() for meta in
                                              weak_unsup["img_metas"]],
                                             [meta["ori_shape"] for meta in weak_unsup["img_metas"]])
        for bboxes, labels, scores, meta_info in zip(pesudo_bboxes, det_labels, det_scores, weak_unsup['img_metas']):
            key = self._extract_key(meta_info['filename'])
            info = torch.cat([bboxes, torch.unsqueeze(scores, dim=1), torch.unsqueeze(labels, dim=1)], dim=1).cpu()
            gather_objects = [key, info]
            outputs = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(outputs, gather_objects)
            for item in outputs:
                self.thr_label_memory_bank[item[0]] = item[1]

    def _get_thr_pesudo_from_memory(self, weak_unsup, device):
        memory_pesudo = [self.thr_label_memory_bank[self._extract_key(meta_info['filename'])] if (
                    self._extract_key(meta_info['filename']) in self.thr_label_memory_bank.keys()) else torch.ones(
            (0, 6))
                         for meta_info in weak_unsup['img_metas']]
        pesudo_bboxes = [pesudo[:, :4].to(device) for pesudo in memory_pesudo]
        pesudo_scores = [pesudo[:, 4].to(device) for pesudo in memory_pesudo]
        pesudo_labels = [pesudo[:, 5].type(torch.long).to(device) for pesudo in memory_pesudo]
        pesudo_bboxes = self._transform_bbox(pesudo_bboxes,
                                             [torch.from_numpy(meta["transform_matrix"]) for meta in
                                              weak_unsup["img_metas"]],
                                             [meta["img_shape"] for meta in weak_unsup["img_metas"]])

        return pesudo_bboxes, pesudo_scores, pesudo_labels

    def _analysis_recall_sim_thr(self, img_feat, unreliable_bboxes, unreliable_scores, unreliable_labels, true_bboxes,
                                 true_labels):
        overlaps = [cal_bboxes_overlaps(u_bboxes, t_bboxes) for u_bboxes, t_bboxes in
                    zip(unreliable_bboxes, true_bboxes)]
        # print(overlaps)
        for i in range(len(overlaps)):
            overlap, ind = overlaps[i]
            ind = ind.long()
            # print(overlap, true_labels[i][ind], unreliable_labels[i])

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
        bboxes_feat = self.teacher.roi_head.extract_bboxes_features(img_feat, roi_list)
        bboxes_feat = norm_tensor(nn.functional.normalize(bboxes_feat, dim=0))

        sim = torch.cosine_similarity(bboxes_feat.unsqueeze(1), self.center_features.unsqueeze(0),
                                      dim=-1)

    def _recall_rois(self, img_feat, unreliable_bboxes, unreliable_scores, unreliable_labels, true_labels=None):

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
        bboxes_feat = self.teacher.roi_head.extract_bboxes_features(img_feat, roi_list)
        bboxes_feat = norm_tensor(nn.functional.normalize(bboxes_feat, dim=0))

        sim = torch.cosine_similarity(bboxes_feat.unsqueeze(1), self.center_features.unsqueeze(0),
                                      dim=-1)
        label_ind = torch.cat(unreliable_labels)
        # print(label_ind)
        # print(sim)
        # sim > 0.7
        # torch.split()
        sim_list = []
        for i in range(len(sim)):
            sim_list.append(sim[i][label_ind[i]])
        sim_list = torch.tensor(sim_list)
        # print(sim_list)
        recall_masks = torch.split(sim_list > 0.7, count_list)
        # print(recall_masks)

        # cls_pro = self._cal_recall_pro()
        # recall_masks = []
        # recall_pros = []
        #
        # for labels, scores in zip(unreliable_labels, unreliable_scores):
        #     sample_pro = (1. - cls_pro[labels]) * (scores / self.pesudo_thr)
        #     # print(sample_pro)
        #     # sample_pro = (1. - cls_pro[labels]) * scores
        #     mask = torch.rand(len(sample_pro)).to(sample_pro.device) < sample_pro
        #     recall_masks.append(mask)
        #     recall_pros.append(sample_pro)
        #     # recall_bboxes.append()
        # self._add_recall_pro(recall_pros)

        bboxes = [recall_bboxes[mask] for (recall_bboxes, mask) in zip(unreliable_bboxes, recall_masks)]
        labels = [recall_labels[mask] for (recall_labels, mask) in zip(unreliable_labels, recall_masks)]
        scores = [recall_scores[mask] for (recall_scores, mask) in zip(unreliable_scores, recall_masks)]

        return bboxes, labels, scores

    def _filter_pesudo_bboxes(self, history_bboxes, history_labels, history_scores, new_bboxes, new_labels):
        masks = []
        for i in range(len(history_bboxes)):
            iou_results = cal_bboxes_all_overlaps(history_bboxes[i], new_bboxes[i])
            tmp_labels1 = new_labels[i].reshape(1, -1).repeat((len(history_bboxes[i]), 1))
            tmp_labels2 = history_labels[i].reshape(1, -1).repeat((len(new_bboxes[i]), 1)).T
            tmp_labels2 = torch.masked_fill(tmp_labels2, iou_results < 0.5, -1)
            aa = tmp_labels1 == tmp_labels2
            masks.append((torch.sum(aa, dim=1) >= 1))
        return masks

    def _filter_incorrect_bboxes(self, det_bboxes, det_labels):
        masks = []
        filter_tensor = torch.tensor([-1]).to(det_bboxes[0].device)
        for i in range(len(det_bboxes)):
            iou_results = cal_bboxes_all_overlaps(det_bboxes[i], det_bboxes[i])
            tmp_labels1 = det_labels[i].reshape(1, -1).repeat((len(det_bboxes[i]), 1)).T
            tmp_labels1 = torch.masked_fill(tmp_labels1, iou_results < 0.95, -1)
            mask = []
            if tmp_labels1.shape[1] == 0:
                masks.append(torch.tensor([], dtype=torch.long).to(det_bboxes[0].device))
                continue

            for i in range(tmp_labels1.shape[1]):
                label_list = torch.unique(tmp_labels1[:, i], sorted=False)
                # print(label_list)
                label_list = label_list[~torch.isin(label_list, filter_tensor)]
                if len(label_list) != 1:
                    mask.append(False)
                else:
                    mask.append(True)
                # print(label_list)
            masks.append(torch.tensor(mask).to(det_bboxes[0].device))
        return masks
        # for j in range(len(det_bboxes)):
        #     print(iou_results[:, j][iou_results[:, j] > 0.95])
        # print(tmp_labels1)
        # print(tmp_labels1[iou_results < 0.95])
        # tmp_labels1 = torch.masked_fill(tmp_labels1, iou_results < 0.95, -1)
        # print(torch.masked_select(tmp_labels1, iou_results < 0.95))
        # if len(tmp_labels1) == 0:

        # else:
        # print(tmp_labels1)
        # print(torch.unique(tmp_labels1, sorted=False, dim=0, return_counts=True))
        # torch.unique_dim()

        # tmp_labels2 = det_labels[i].reshape(1, -1).repeat((len(det_bboxes[i]), 1))

        # print(tmp_labels2)
        # tmp_labels1 = torch.masked_fill(tmp_labels1, iou_results < 0.5, -1)

    def _cal_recall_pro(self):
        return torch.div(self.cls_count[0], torch.sum(self.cls_count[0]))

    def _gen_unreliable_pseudo_labels(self, weak_unsup, strong_unsup):
        self.teacher.eval()

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

        # filter bboxes using thr
        reliable_bboxes = [bboxes[scores >= self.pesudo_thr] for (bboxes, scores) in zip(result_bboxes, result_scores)]
        reliable_labels = [labels[scores >= self.pesudo_thr] for (labels, scores) in zip(det_labels, result_scores)]
        reliable_scores = [scores1[scores2 >= self.pesudo_thr] for (scores1, scores2) in
                           zip(result_scores, result_scores)]

        unreliable_bboxes = [bboxes[(scores >= self.unreliable_thr) & (scores < self.pesudo_thr)] for (bboxes, scores)
                             in zip(result_bboxes, result_scores)]
        unreliable_labels = [labels[(scores >= self.unreliable_thr) & (scores < self.pesudo_thr)] for (labels, scores)
                             in zip(det_labels, result_scores)]
        unreliable_scores = [scores1[(scores2 >= self.unreliable_thr) & (scores2 < self.pesudo_thr)] for
                             (scores1, scores2)
                             in zip(result_scores, result_scores)]

        self._analysis_recall_sim_thr(feat, unreliable_bboxes, unreliable_scores, unreliable_labels,
                                      [bboxes for bboxes in strong_unsup["gt_bboxes"]],
                                      [labels for labels in strong_unsup["gt_labels"]])

        if self.is_label_memory is True:
            self._ori_update_label_memory(reliable_bboxes, reliable_labels, reliable_scores, weak_unsup)
            # self._update_label_memory_mean(gt_bboxes, gt_labels, gt_scores, weak_unsup)
            gt_bboxes, gt_scores, gt_labels = self._get_pesudo_from_memory(weak_unsup, reliable_bboxes[0].device)

        if self.is_recall is True:
            recall_bboxes, recall_labels, recall_scores = self._recall_rois(feat, unreliable_bboxes, unreliable_scores,
                                                                            unreliable_labels)
            gt_bboxes = [torch.cat([bboxes1, bboxes2], dim=0) for (bboxes1, bboxes2) in
                         zip(reliable_bboxes, recall_bboxes)]
            gt_labels = [torch.cat([labels1, labels2]) for (labels1, labels2) in
                         zip(reliable_labels, recall_labels)]
            gt_scores = [torch.cat([scores1, scores2]) for (scores1, scores2) in
                         zip(reliable_scores, recall_scores)]

        if self.is_unreliable_label_memory is True:
            recall_bboxes, recall_labels, recall_scores = self._recall_rois(feat, unreliable_bboxes, unreliable_scores,
                                                                            unreliable_labels)
            # self._update_label_memory(gt_bboxes, gt_labels, gt_scores, weak_unsup)
            history_bboxes, history_scores, history_labels = self._get_pesudo_from_memory(weak_unsup,
                                                                                          reliable_bboxes[0].device)
            new_bboxes = [torch.cat([bboxes1, bboxes2], dim=0) for (bboxes1, bboxes2) in
                          zip(reliable_bboxes, recall_bboxes)]
            new_labels = [torch.cat([labels1, labels2]) for (labels1, labels2) in zip(reliable_labels, recall_labels)]
            new_scores = [torch.cat([scores1, scores2]) for (scores1, scores2) in zip(reliable_scores, recall_scores)]
            masks = self._filter_pesudo_bboxes(history_bboxes, history_labels, history_scores, new_bboxes, new_labels)

            self._update_label_memory(reliable_bboxes, reliable_labels, reliable_scores, weak_unsup)

            # gt_bboxes = [torch.cat([bboxes[mask], bboxes2]) for (bboxes, mask, bboxes2) in zip(history_bboxes, masks, gt_bboxes)]
            # gt_labels = [torch.cat([labels[mask], labels2]).type(torch.long) for (labels, mask, labels2) in zip(history_labels, masks, gt_labels)]
            # gt_scores = [torch.cat([scores[mask], scores2]) for (scores, mask, scores2) in zip(history_scores, masks, gt_scores)]

            # gt_bboxes, gt_labels, gt_scores = self._nms_bboxes(gt_bboxes, gt_labels, gt_scores)

            tmp_bboxes, tmp_scores, tmp_labels = self._get_thr_pesudo_from_memory(weak_unsup, reliable_bboxes[0].device)
            gt_bboxes = [torch.cat([bboxes1, bboxes2, bboxes3], dim=0) for (bboxes1, bboxes2, bboxes3) in
                         zip(tmp_bboxes, reliable_bboxes, recall_bboxes)]
            gt_labels = [torch.cat([labels1, labels2, labels3]) for (labels1, labels2, labels3) in
                         zip(tmp_labels, reliable_labels, recall_labels)]
            gt_scores = [torch.cat([scores1, scores2, scores3]) for (scores1, scores2, scores3) in
                         zip(tmp_scores, reliable_scores, recall_scores)]

            # gt_bboxes, gt_labels, gt_scores = self._nms_bboxes(gt_bboxes, gt_labels, gt_scores)

            # print(gt_bboxes, gt_labels, gt_scores)
            # thr_bboxes = [torch.cat([bboxes[~mask], bboxes2]) for (bboxes, mask, bboxes2) in zip(history_bboxes, masks, recall_bboxes)]
            # thr_labels = [torch.cat([labels[~mask], labels2]).type(torch.long) for (labels, mask, labels2) in zip(history_labels, masks, recall_labels)]
            # thr_scoers = [torch.cat([scores[~mask], scores2]) for (scores, mask, scores2) in zip(history_scores, masks, recall_scores)]
            mask2 = [scores > self.pesudo_thr for scores in history_scores]
            mask3 = [(~tmp_mask1) & tmp_mask2 for (tmp_mask1, tmp_mask2) in zip(masks, mask2)]
            thr_bboxes = [bboxes[mask] for (bboxes, mask) in zip(history_bboxes, mask3)]
            thr_labels = [labels[mask].type(torch.long) for (labels, mask) in zip(history_labels, mask3)]
            thr_scoers = [scores[mask] for (scores, mask) in zip(history_scores, mask3)]
            self._update_thr_label_memory(thr_bboxes, thr_labels, thr_scoers, weak_unsup)

        # print(gt_bboxes)
        # print(gt_scores)
        M = self._extract_transform_matrix(weak_unsup, strong_unsup)
        reliable_bboxes = self._transform_bbox(
            reliable_bboxes,
            M,
            [meta["img_shape"] for meta in strong_unsup["img_metas"]],
        )

        strong_unsup.update({"reliable_bboxes": reliable_bboxes})
        strong_unsup.update({"reliable_labels": reliable_labels})
        strong_unsup.update({"reliable_scores": reliable_scores})

        if self.is_label_memory is False and self.is_recall is False and self.is_unreliable_label_memory is False:
            gt_bboxes = reliable_bboxes
            gt_labels = reliable_labels
            gt_scores = reliable_scores
        else:
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

        if self.print_pesudo_summary:
            self._add_summary_bboxes(
                [{"bboxes": torch.cat([bboxes, torch.unsqueeze(scores, dim=1)], dim=1).detach().cpu().numpy(),
                  "labels": labels.detach().cpu().numpy()} for (bboxes, labels, scores) in
                 zip(strong_unsup["gt_bboxes"], strong_unsup["gt_labels"], strong_unsup["gt_scores"])],
                [{"bboxes": bboxes.detach().cpu().numpy(),
                  "labels": labels.detach().cpu().numpy()} for (bboxes, labels) in
                 zip(strong_unsup["gt_bboxes_true"], strong_unsup["gt_labels_true"])])
            self._add_unreliable_bboxes(
                [{"bboxes": torch.cat([bboxes, torch.unsqueeze(scores, dim=1)], dim=1).detach().cpu().numpy(),
                  "labels": labels.detach().cpu().numpy()} for (bboxes, labels, scores) in
                 zip(strong_unsup["unreliable_bboxes"], strong_unsup["unreliable_labels"],
                     strong_unsup["unreliable_scores"])])
        return strong_unsup

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

        num_imgs = len(sup_data["img_metas"])
        gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], sup_data["gt_bboxes"][i], gt_bboxes_ignore[i],
                sup_data["gt_labels"][i])
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                sup_data["gt_bboxes"][i],
                sup_data["gt_labels"][i],
                feats=[lvl_feat[i][None] for lvl_feat in feats])
            sampling_results.append(sampling_result)

        bbox_results = self.student.roi_head._bbox_forward_train(feats,
                                                                 sampling_results,
                                                                 sup_data["gt_bboxes"],
                                                                 sup_data["gt_labels"],
                                                                 sup_data["img_metas"])
        losses.update(bbox_results['loss_bbox'])
        losses = {"sup_" + k: v for k, v in losses.items()}

        if self.is_sup_neg_loss is True:
            sup_neg_loss = self.student.roi_head._sup_negtive_loss(feats,
                                                                   sampling_results,
                                                                   sup_data["gt_bboxes"],
                                                                   sup_data["gt_labels"],
                                                                   self.center_features,
                                                                   n_cls=self.n_cls)
            sup_neg_loss = weighted_all_loss(sup_neg_loss, self.sup_contrast_loss_weight)
            losses.update(sup_neg_loss)

        return losses

    def _compute_student_unsup_negative_loss(self, weak_unsup, strong_unsup):
        with torch.no_grad():
            weak_feat = self.teacher.extract_feat(weak_unsup["img"])
        strong_feat = self.student.extract_feat(strong_unsup["img"])

        self.student.train()
        losses = dict()

        if self.is_region_est is False:
            roi_losses = self._compute_student_unsup_losses(strong_unsup)
            roi_losses = {"unsup_" + k: v for k, v in roi_losses.items()}
            roi_losses = weighted_all_loss(roi_losses, self.unsup_loss_weight)
            losses.update(roi_losses)
        else:
            # student RPN forward and loss
            proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                      self.student.test_cfg.rpn)

            rpn_losses, proposal_list = self.student.rpn_head.forward_train(
                strong_feat,
                strong_unsup["img_metas"],
                strong_unsup["gt_bboxes"],
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            weighted_roi_losses = self._compute_student_weighted_rcnn_losses(strong_feat,
                                                                             strong_unsup["img_metas"],
                                                                             proposal_list,
                                                                             strong_unsup["gt_bboxes"],
                                                                             strong_unsup["gt_labels"],
                                                                             gt_bboxes_true=strong_unsup[
                                                                                 "gt_bboxes_true"],
                                                                             gt_labels_true=strong_unsup[
                                                                                 "gt_labels_true"])
            weighted_roi_losses = {"unsup_" + k: v for k, v in weighted_roi_losses.items()}
            # weighted_roi_losses = weighted_all_loss(weighted_roi_losses, self.unsup_loss_weight)

            losses.update(weighted_roi_losses)

        if self.is_neg_loss is True:
            # contrast_losses, mask_info = self.student.roi_head._negtive_loss(x=strong_feat,
            #                                                                  weak_feats=weak_feat,
            #                                                                  trans_m=strong_unsup["trans_m"],
            #                                                                  gt_bboxes=strong_unsup["unreliable_bboxes"],
            #                                                                  gt_labels=strong_unsup["unreliable_labels"],
            #                                                                  img_metas=strong_unsup["img_metas"],
            #                                                                  pos_queue=self.projector_pos_queue,
            #                                                                  neg_queue=self.projector_neg_queue,
            #                                                                  weak_imgs=weak_unsup["img"],
            #                                                                  strong_imgs=strong_unsup["img"],
            #                                                                  n_cls=self.n_cls,
            #                                                                  thr=self.neg_thr)
            contrast_losses, mask_info = self.student.roi_head._negtive_center_loss(x=strong_feat,
                                                                                    weak_feats=weak_feat,
                                                                                    trans_m=strong_unsup["trans_m"],
                                                                                    gt_bboxes=strong_unsup[
                                                                                        "unreliable_bboxes"],
                                                                                    gt_labels=strong_unsup[
                                                                                        "unreliable_labels"],
                                                                                    img_metas=strong_unsup["img_metas"],
                                                                                    center_feats=self.center_features,
                                                                                    weak_imgs=weak_unsup["img"],
                                                                                    strong_imgs=strong_unsup["img"],
                                                                                    n_cls=self.n_cls,
                                                                                    thr=self.neg_thr)
            # print('before:', contrast_losses)
            if self.print_pesudo_summary:
                self._add_neg_loss_info(mask_info.detach().cpu(),
                                        [bboxes.detach().cpu().numpy() for bboxes in strong_unsup["unreliable_bboxes"]])

                # strong_unsup["unreliable_bboxes"].detach().cpu())

            contrast_losses = weighted_all_loss(contrast_losses, self.contrast_loss_weight)
            # print('after:', contrast_losses)
            losses.update(contrast_losses)

        del weak_feat

        return self._check_losses_item(losses, strong_feat[0].device)

    def _compute_student_unsup_losses(self, strong_unsup, proposals=None):
        feat = self.student.extract_feat(strong_unsup["img"])
        losses = dict()

        if self.is_ignore_ubreliable is True:
            ignore_bboxes = self._cal_ignore_unreliable_bboxes(strong_unsup["gt_bboxes"],
                                                               strong_unsup["unreliable_bboxes"])
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

        roi_losses = self._compute_student_rcnn_losses(feat,
                                                       strong_unsup["img_metas"],
                                                       proposal_list,
                                                       strong_unsup["gt_bboxes"],
                                                       strong_unsup["gt_labels"],
                                                       gt_bboxes_ignore=ignore_bboxes,
                                                       gt_bboxes_true=strong_unsup["gt_bboxes_true"],
                                                       gt_labels_true=strong_unsup["gt_labels_true"])
        # gt_bboxes_ignore=strong_unsup["unreliable_bboxes"])
        losses.update(roi_losses)

        return losses

    def _compute_student_weighted_rcnn_losses(self,
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
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox_results, rois, rois_mask, label_weights, roi_labels = self.student.roi_head._weighted_bbox_forward_train(x,
        #                                                                                                               sampling_results,
        #                                                                                                               gt_bboxes,
        #                                                                                                               gt_labels,
        #                                                                                                               self.projector_pos_queue,
        #                                                                                                               self.projector_neg_queue,
        #                                                                                                               self.n_cls,
        #                                                                                                               True)
        bbox_results, rois, rois_mask, label_weights, roi_labels = self.student.roi_head._center_weighted_bbox_forward_train(
            x,
            sampling_results,
            gt_bboxes,
            gt_labels,
            self.center_features,
            self.n_cls,
            True,
            is_weight_norm=self.is_weight_norm)

        if self.print_pesudo_summary:
            for idx_img in range(num_imgs):
                mask = (rois[:, 0] == idx_img)
                pos_bboxes = rois[mask][rois_mask[mask] == 1.0]
                neg_bboxes = rois[mask][rois_mask[mask] == 0.0]

                pos_roi_labels = roi_labels[mask][rois_mask[mask] == 1.0]
                neg_roi_labels = roi_labels[mask][rois_mask[mask] == 0.0]

                pos_weight = label_weights[mask][rois_mask[mask] == 1.0]
                neg_weight = label_weights[mask][rois_mask[mask] == 0.0]

                self._add_unsup_sampling_bboxes(pos_bboxes.detach().cpu(),
                                                neg_bboxes.detach().cpu(),
                                                gt_bboxes_true[idx_img].detach().cpu(),
                                                img_metas[idx_img],
                                                pos_weight.detach().cpu(),
                                                neg_weight.detach().cpu(),
                                                pos_roi_labels.detach().cpu(),
                                                neg_roi_labels.detach().cpu(),
                                                )
        losses.update(bbox_results['loss_bbox'])

        return losses

    @torch.no_grad()
    def _update_class_centers(self, sup_data, proposals=None, is_sup=True):
        self.teacher.eval()
        with torch.no_grad():
            feat = self.teacher.extract_feat(sup_data['img'])
            # print("feat:", feat[0].dtype)
            proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, sup_data['img_metas'])

            det_bboxes, det_labels = self.teacher.roi_head.simple_test_bboxes(
                feat, sup_data['img_metas'], proposal_list, self.teacher.test_cfg.rcnn, rescale=False)
            if is_sup is True:
                gts = sup_data["gt_bboxes"]
            else:
                gts = sup_data["reliable_bboxes"]
            projector_bg_feats, projector_gt_feats, bg_feats, gt_feats = self.teacher.roi_head.extract_projector_features(
                feat, det_bboxes, gts, self.train_cfg, self.train_cfg.region_bg_score_thr)

        bg_feats = norm_tensor(nn.functional.normalize(bg_feats, dim=0))
        gt_feats = norm_tensor(nn.functional.normalize(gt_feats, dim=0))

        self._update_bg_centers(bg_feats, img_num=len(proposal_list))
        self._update_fg_centers(gt_feats, torch.cat(sup_data["gt_labels"], dim=0), img_num=len(proposal_list))

        if torch.sum(torch.isnan(self.center_features)) != 0:
            print(self.center_features)
            print('centers has nan')
            exit()

    @torch.no_grad()
    def _update_reliable_queues(self, sup_data, proposals=None):
        self.teacher.eval()
        with torch.no_grad():
            feat = self.teacher.extract_feat(sup_data['img'])
            proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, sup_data['img_metas'])

            det_bboxes, det_labels = self.teacher.roi_head.simple_test_bboxes(
                feat, sup_data['img_metas'], proposal_list, self.teacher.test_cfg.rcnn, rescale=False)
            projector_bg_feats, projector_gt_feats, bg_feats, gt_feats = self.teacher.roi_head.extract_projector_features(
                feat, det_bboxes, sup_data["gt_bboxes"], self.train_cfg, self.train_cfg.region_bg_score_thr)
            # print('det bboxes:', det_bboxes)
            # print('proposals:', proposal_list)

        #
        projector_bg_feats = norm_tensor(nn.functional.normalize(projector_bg_feats, dim=0))
        projector_gt_feats = norm_tensor(nn.functional.normalize(projector_gt_feats, dim=0))
        # bg_feats = norm_tensor(nn.functional.normalize(bg_feats, dim=0))
        # gt_feats = norm_tensor(nn.functional.normalize(gt_feats, dim=0))

        # self._neg_dequeue_and_enqueue(bg_feats, img_num=len(proposal_list))
        # self._pos_dequeue_and_enqueue(gt_feats, torch.cat(sup_data["gt_labels"], dim=0), img_num=len(proposal_list))
        self._projector_neg_dequeue_and_enqueue(projector_bg_feats, img_num=len(proposal_list))
        self._projector_pos_dequeue_and_enqueue(projector_gt_feats, torch.cat(sup_data["gt_labels"], dim=0),
                                                img_num=len(proposal_list))

    @torch.no_grad()
    def _neg_dequeue_and_enqueue(self, keys, img_num=0):
        """Update neg region queue."""

        broadcast_keys = torch.zeros(img_num * self.region_bg_max_num, self.bbox_feat_dim).to(keys.device)
        broadcast_mask = torch.zeros(img_num * self.region_bg_max_num).to(keys.device)
        broadcast_keys[:len(keys)] = keys
        broadcast_mask[:len(keys)] = 1.0
        bg_keys = gather_same_shape_tensors(broadcast_keys)
        bg_masks = gather_same_shape_tensors(broadcast_mask)
        keys = bg_keys[bg_masks == 1.0]

        update_size = keys.shape[0]
        ptr = int(self.neg_queue_ptr)

        if (ptr + update_size) > self.neg_queue_len:
            len_11 = self.neg_queue_len - ptr
            self.neg_queue[:, ptr:] = keys[:len_11].transpose(0, 1)
            ptr = (ptr + update_size) % self.neg_queue_len
            self.neg_queue[:, :ptr] = keys[len_11:].transpose(0, 1)
        else:
            self.neg_queue[:, ptr:ptr + update_size] = keys.transpose(0, 1)
            ptr = (ptr + update_size) % self.neg_queue_len  # move pointer
        #
        self.neg_queue_ptr[0] = ptr

    @torch.no_grad()
    def _pos_dequeue_and_enqueue(self, keys, gt_labels, img_num=0):
        """Update pos region queue."""
        broadcast_keys = torch.zeros(img_num * self.region_fg_max_num, self.bbox_feat_dim).to(keys.device)
        broadcast_labels = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
        broadcast_mask = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
        # broadcast_mask[:]
        broadcast_keys[:len(keys)] = keys
        broadcast_labels[:len(keys)] = gt_labels
        broadcast_mask[:len(keys)] = 1.0
        # print(broadcast_mask)
        fg_keys = gather_same_shape_tensors(broadcast_keys)
        fg_masks = gather_same_shape_tensors(broadcast_mask)
        fg_labels = gather_same_shape_tensors(broadcast_labels)
        keys = fg_keys[fg_masks == 1.0]
        gts_labels = fg_labels[fg_masks == 1.0]

        for i in range(self.n_cls):
            update_size = keys[gts_labels == i].shape[0]
            ptr = int(self.pos_queue_ptr[i])

            if (ptr + update_size) > self.pos_queue_len:
                len_11 = self.pos_queue_len - ptr
                self.pos_queue[i, :, ptr:] = keys[gts_labels == i][:len_11].transpose(0, 1)
                ptr = (ptr + update_size) % self.pos_queue_len
                self.pos_queue[i, :, :ptr] = keys[gts_labels == i][len_11:].transpose(0, 1)
            else:
                self.pos_queue[i, :, ptr:ptr + update_size] = keys[gts_labels == i].transpose(0, 1)
                ptr = (ptr + update_size) % self.pos_queue_len  # move pointer
            #
            self.pos_queue_ptr[i][0] = ptr

    @torch.no_grad()
    def _projector_neg_dequeue_and_enqueue(self, keys, img_num=0):
        """Update neg region queue."""

        broadcast_keys = torch.zeros(img_num * self.region_bg_max_num, self.feat_dim).to(keys.device)
        broadcast_mask = torch.zeros(img_num * self.region_bg_max_num).to(keys.device)
        broadcast_keys[:len(keys)] = keys
        broadcast_mask[:len(keys)] = 1.0
        bg_keys = gather_same_shape_tensors(broadcast_keys)
        bg_masks = gather_same_shape_tensors(broadcast_mask)
        keys = bg_keys[bg_masks == 1.0]

        update_size = keys.shape[0]
        ptr = int(self.projector_neg_queue_ptr)

        if (ptr + update_size) > self.neg_queue_len:
            len_11 = self.neg_queue_len - ptr
            self.projector_neg_queue[:, ptr:] = keys[:len_11].transpose(0, 1)
            ptr = (ptr + update_size) % self.neg_queue_len
            self.projector_neg_queue[:, :ptr] = keys[len_11:].transpose(0, 1)
        else:
            self.projector_neg_queue[:, ptr:ptr + update_size] = keys.transpose(0, 1)
            ptr = (ptr + update_size) % self.neg_queue_len  # move pointer
        #
        self.projector_neg_queue_ptr[0] = ptr

    @torch.no_grad()
    def _projector_pos_dequeue_and_enqueue(self, keys, gt_labels, img_num=0):
        """Update pos region queue."""
        broadcast_keys = torch.zeros(img_num * self.region_fg_max_num, self.feat_dim).to(keys.device)
        broadcast_labels = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
        broadcast_mask = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
        # broadcast_mask[:]
        broadcast_keys[:len(keys)] = keys
        broadcast_labels[:len(keys)] = gt_labels
        broadcast_mask[:len(keys)] = 1.0
        # print(broadcast_mask)
        fg_keys = gather_same_shape_tensors(broadcast_keys)
        fg_masks = gather_same_shape_tensors(broadcast_mask)
        fg_labels = gather_same_shape_tensors(broadcast_labels)
        keys = fg_keys[fg_masks == 1.0]
        gts_labels = fg_labels[fg_masks == 1.0]

        for i in range(self.n_cls):
            update_size = keys[gts_labels == i].shape[0]
            ptr = int(self.projector_pos_queue_ptr[i])

            if (ptr + update_size) > self.pos_queue_len:
                len_11 = self.pos_queue_len - ptr
                self.projector_pos_queue[i, :, ptr:] = keys[gts_labels == i][:len_11].transpose(0, 1)
                ptr = (ptr + update_size) % self.pos_queue_len
                self.projector_pos_queue[i, :, :ptr] = keys[gts_labels == i][len_11:].transpose(0, 1)
            else:
                self.projector_pos_queue[i, :, ptr:ptr + update_size] = keys[gts_labels == i].transpose(0, 1)
                ptr = (ptr + update_size) % self.pos_queue_len  # move pointer
            #
            self.projector_pos_queue_ptr[i][0] = ptr

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

    @torch.no_grad()
    def _update_fg_centers(self, keys, gt_labels, img_num=0):
        alpha = min(1 - 1 / (self.ema_iteration + 1), self.ema_decay)

        broadcast_keys = torch.zeros(img_num * self.region_fg_max_num, self.feat_dim).to(keys.device)
        broadcast_labels = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
        broadcast_mask = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
        # broadcast_mask[:]
        broadcast_keys[:len(keys)] = keys
        broadcast_labels[:len(keys)] = gt_labels
        broadcast_mask[:len(keys)] = 1.0
        # print(broadcast_mask)
        fg_keys = gather_same_shape_tensors(broadcast_keys)
        fg_masks = gather_same_shape_tensors(broadcast_mask)
        fg_labels = gather_same_shape_tensors(broadcast_labels)
        keys = fg_keys[fg_masks == 1.0]
        gts_labels = fg_labels[fg_masks == 1.0]
        for i in range(self.n_cls):
            cls_keys = keys[gts_labels == i]
            if len(cls_keys) != 0:
                self.center_features[i] = self.center_features[i] * alpha + torch.mean(keys[gts_labels == i], dim=0) * (
                            1 - alpha)

    @torch.no_grad()
    def _update_bg_centers(self, keys, img_num=0):
        alpha = min(1 - 1 / (self.ema_iteration + 1), self.ema_decay)

        broadcast_keys = torch.zeros(img_num * self.region_bg_max_num, self.feat_dim).to(keys.device)
        broadcast_mask = torch.zeros(img_num * self.region_bg_max_num).to(keys.device)
        broadcast_keys[:len(keys)] = keys
        broadcast_mask[:len(keys)] = 1.0
        bg_keys = gather_same_shape_tensors(broadcast_keys)
        bg_masks = gather_same_shape_tensors(broadcast_mask)
        keys = bg_keys[bg_masks == 1.0]
        if len(keys) != 0:
            self.center_features[self.n_cls] = self.center_features[self.n_cls] * alpha + torch.mean(keys, dim=0) * (
                        1 - alpha)
