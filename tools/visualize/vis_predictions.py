import argparse
import os
import os.path as osp
import time
import warnings
import sys
sys.path.append('/home/liu/ytx/SS-OD')
from mmcv import Config, DictAction
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmssod.utils.patch import patch_config
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.image import tensor2imgs
from mmdet.core.visualization import imshow_gt_det_bboxes
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import numpy as np
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmssod.utils.cluster import K_means

# CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
#                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
#                'tvmonitor')
CLASSES = np.array(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor', 'BG'])

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet model prediction visualization')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    print('1')

    cfg = Config.fromfile(args.config)
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1)
    cfg = patch_config(cfg)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')


    # build the dataloader
    dataset = build_dataset(cfg.unsup_test)
    # dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if cfg.get('train_cfg') != None:
        cfg.model = cfg.model_wrapper
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        prog_bar = mmcv.ProgressBar(len(dataset))
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        model.eval()
        results = []

        for i, data in enumerate(data_loader):
            gt_bboxes = data.pop("gt_bboxes")
            gt_labels = data.pop("gt_labels")
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            # result = model_inference1(model.module, data)
            # result = model_inference_cascade(model.module, data)
            # print(result)
            det_bboxes = np.vstack(result[0])
            # print('result bbox:', det_bboxes)
            # print('true bbox:', gt_bboxes)
            # bboxes = np.vstack(bbox_result)

            results.extend(result)

            batch_size = len(result)

            # save_det_results(gt_bboxes,
            #                  gt_labels,
            #                  data,
            #                  result,
            #                  args.show_dir,
            #                  score_thr=args.show_score_thr)

            for _ in range(batch_size):
                prog_bar.update()
            continue
        metric = dataset.evaluate(results)
        print(metric)

        metric = dataset.evaluate(results, iou_thrs=[0.5])
        metric = dataset.evaluate(results, iou_thrs=[0.55])
        metric = dataset.evaluate(results, iou_thrs=[0.6])
        metric = dataset.evaluate(results, iou_thrs=[0.65])
        metric = dataset.evaluate(results, iou_thrs=[0.70])
        metric = dataset.evaluate(results, iou_thrs=[0.75])
        metric = dataset.evaluate(results, iou_thrs=[0.80])
        metric = dataset.evaluate(results, iou_thrs=[0.85])
        metric = dataset.evaluate(results, iou_thrs=[0.90])
        metric = dataset.evaluate(results, iou_thrs=[0.95])

            # if i == 100:
            #     exit()
            #
            # # test_cfg = {'score_thr': 0.7,
            # #             'nms': {'type': 'nms', 'iou_threshold': 0.5},
            # #             # 'nms': None,
            # #             'max_per_img': 128}
            # # test_cfg = Config(test_cfg)
            #
            # model.module.teacher.test_cfg.rcnn.score_thr=0.3
            #
            # with torch.no_grad():
            #     result = model(return_loss=False, rescale=False, **data)
            #
            # batch_size = len(result)
            #
            # det_bboxes = np.vstack(result[0])
            # det_labels = [
            #     np.full(bbox.shape[0], i, dtype=np.int32)
            #     for i, bbox in enumerate(result[0])
            # ]
            # det_labels = np.concatenate(det_labels)
            #
            # mask = det_bboxes[:, 4] > 0.3
            # det_bboxes = det_bboxes[mask]
            # det_labels = det_labels[mask]
            #
            #
            #
            # if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
            #     img_tensor = data['img'][0]
            # else:
            #     img_tensor = data['img'][0].data[0]
            # img_metas = data['img_metas'][0].data[0]
            # imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            #
            # assert len(imgs) == len(img_metas)
            #
            # for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            #     h, w, _ = img_meta['img_shape']
            #     img_show = img[:h, :w, :]
            #
            #     ori_h, ori_w = img_meta['ori_shape'][:-1]
            #     # img_show = mmcv.imresize(img_show, (ori_w, ori_h))
            #
            #     if args.show_dir:
            #         out_file = osp.join(args.show_dir, img_meta['ori_filename'])
            #     else:
            #         out_file = None
            #     print(out_file)
            #     imshow_gt_det_bboxes(img_show,
            #                          {"gt_bboxes": gt_bboxes[0][i].numpy(),
            #                           "gt_labels": gt_labels[0][i].numpy()},
            #                          result[i],
            #                          class_names=dataset.CLASSES,
            #                          score_thr=args.show_score_thr,
            #                          show=False,
            #                          out_file=out_file)
            #
            # for _ in range(batch_size):
            #     prog_bar.update()

def save_det_results(gt_bboxes,
                     gt_labels,
                     data,
                     result,
                     show_dir,
                     score_thr=0.7):
    det_bboxes = np.vstack(result[0])
    det_labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result[0])
    ]
    det_labels = np.concatenate(det_labels)

    mask = det_bboxes[:, 4] > 0.3
    det_bboxes = det_bboxes[mask]
    det_labels = det_labels[mask]
    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])

    assert len(imgs) == len(img_metas)

    for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        # ori_h, ori_w = img_meta['ori_shape'][:-1]
        # img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        out_file = osp.join(show_dir, img_meta['ori_filename'])
        print(out_file)
        imshow_gt_det_bboxes(img_show,
                             {"gt_bboxes": gt_bboxes[0][i].numpy(),
                              "gt_labels": gt_labels[0][i].numpy()},
                             result[i],
                             class_names=CLASSES.tolist(),
                             score_thr=score_thr,
                             show=False,
                             out_file=out_file)



@torch.no_grad()
def model_inference1(model, data):
    # print(data['img'])
    model = model.half()
    imgs = [img.cuda().half() for img in data['img']]
    # img_metas = data['img_metas']
    img_metas = data['img_metas'][0].data[0]

    feat = model.extract_feat(imgs[0])
    # print(data['img_metas'][0].data)
    proposal_list = model.rpn_head.simple_test_rpn(feat, img_metas)

    test_cfg ={'score_thr': 0.5,
               # 'nms': {'type': 'nms', 'iou_threshold': 1.0},
               'nms': None,
               'max_per_img': 128}
    test_cfg = Config(test_cfg)
    det_bboxes1, det_labels1 = model.roi_head.simple_test_bboxes(
        feat, img_metas, proposal_list, test_cfg, rescale=False)
    # print("det 1:", det_bboxes1)

    test_cfg ={'score_thr': 0.7,
               'nms': {'type': 'nms', 'iou_threshold': 0.5},
               # 'nms': None,
               'max_per_img': 128}
    test_cfg = Config(test_cfg)
    det_bboxes2, det_labels2 = model.roi_head.simple_test_bboxes(
        feat, img_metas, proposal_list, test_cfg, rescale=False)
    # print("det 2:", det_bboxes2)
    # print("det 2:", det_labels2)

    from mmssod.models.utils.eval_utils import cal_bboxes_overlaps

    overlaps, inds = cal_bboxes_overlaps(det_bboxes1[0][:, :4], det_bboxes2[0][:, :4])
    # print('det bboxes:', len(det_bboxes2[0][:, :4]))
    # print('inds:', inds)

    # print(det_labels1)
    # print(det_labels2)
    # print('mask:', mask)
    # print(det_labels1[0][mask])

    if len(det_bboxes2[0]) == 0:
        res_bboxes = det_bboxes2
        res_labels = det_labels2

    else:
        # # mean bboxes
        # from mmssod.models.utils.bbox_utils import mean_bboxes
        # mean_bboxes_list = []
        # overlaps_mask = overlaps > 0.5
        #
        # for i in range(len(det_bboxes2[0])):
        #     ind_mask = inds == i
        #     mask = ind_mask & overlaps_mask
        #
        #     bboxes = mean_bboxes(det_bboxes1[0][mask])
        #     mean_bboxes_list.append(bboxes)
        #
        # res_bboxes = torch.cat(mean_bboxes_list, dim=0)
        # res_bboxes = [torch.cat((res_bboxes, det_bboxes2[0][:, 4].reshape((-1, 1))), dim=1)]
        # res_labels = det_labels2


        # k-means
        bboxes_list = []
        labels_list = []
        overlaps_mask = overlaps > 0.5
        for i in range(len(det_bboxes2[0])):
            ind_mask = inds == i
            mask = ind_mask & overlaps_mask
            _, centers = K_means(data=det_bboxes1[0][mask], k=2).forward()
            print(centers)
            if len(centers) > 1:
                centers = torch.vstack(centers)
            elif len(centers) == 1:
                centers = centers.reshape((1, -1))
            else:
                continue
            # print(centers)
            if len(centers) != 0:
                aa = torch.argmax(centers[:, 4])
            # print(aa)
            # centers = torch.mean(centers).reshape((1, -1))
            bboxes_list.append(centers[aa].reshape((1, -1)))
            labels_list.append(det_labels1[0][mask][:1])

        res_bboxes = [torch.cat(bboxes_list, dim=0)]
        res_labels = [torch.cat(labels_list)]
        # print("res_bboxes:", res_bboxes)
        # print("res_labels:", res_labels)


        # # no nms
        # mask = overlaps > 0.5
        #
        # selected =  det_labels1[0][mask] == det_labels2[0][inds][mask]
        # res_bboxes = [det_bboxes1[0][mask][selected]]
        # res_labels = [det_labels1[0][mask][selected]]


    print("len:", len(res_bboxes))
    bbox_results = [
        bbox2result(res_bboxes[i], res_labels[i],
                    20)
        for i in range(len(res_bboxes))
    ]
    return bbox_results


@torch.no_grad()
def model_inference_cascade(model, data):
    # print(data['img'])
    print("cascade")
    model = model.half()
    imgs = [img.cuda().half() for img in data['img']]
    # img_metas = data['img_metas']
    img_metas = data['img_metas'][0].data[0]

    feat = model.extract_feat(imgs[0])
    # print(data['img_metas'][0].data)
    proposal_list = model.rpn_head.simple_test_rpn(feat, img_metas)
    test_cfg ={'score_thr': 0.3,
               # 'nms': {'type': 'nms', 'iou_threshold': 1.0},
               'nms': None,
               'max_per_img': 128}
    test_cfg = Config(test_cfg)
    # print("proposal:", proposal_list)

    det_bboxes1, det_labels1 = model.roi_head.simple_test_bboxes(
        feat, img_metas, proposal_list, test_cfg, rescale=False)
    # print("det 1:", det_bboxes1)

    test_cfg ={'score_thr': 0.7,
               'nms': {'type': 'nms', 'iou_threshold': 0.5},
               # 'nms': None,
               'max_per_img': 128}
    test_cfg = Config(test_cfg)
    det_bboxes2, det_labels2 = model.roi_head.simple_test_bboxes(
        feat, img_metas, proposal_list, test_cfg, rescale=False)

    from mmssod.models.utils.eval_utils import cal_bboxes_overlaps

    overlaps, inds = cal_bboxes_overlaps(det_bboxes1[0][:, :4], det_bboxes2[0][:, :4])

    if len(det_bboxes2[0]) == 0:
        res_bboxes = det_bboxes2
        res_labels = det_labels2

    else:
        mask = overlaps > 0.5

        selected =  det_labels1[0][mask] == det_labels2[0][inds][mask]
        res_bboxes = [det_bboxes1[0][mask][selected]]
        res_labels = [det_labels1[0][mask][selected]]


    # print("res_bboxes", res_bboxes)
    test_cfg = {'score_thr': 0.7,
                'nms': {'type': 'nms', 'iou_threshold': 0.5},
                # 'nms': None,
                'max_per_img': 128}
    test_cfg = Config(test_cfg)
    det_bboxes_stage_2, det_labels_stage_2 = model.roi_head.simple_test_bboxes(
        feat, img_metas, res_bboxes, test_cfg, rescale=True)


    bbox_results = [
        bbox2result(det_bboxes_stage_2[i], det_labels_stage_2[i],
                    20)
        for i in range(len(det_bboxes_stage_2))
    ]
    return bbox_results

@torch.no_grad()
def inference_neg(model, data, det_bboxes):


    model = model.half()
    # print(data['img'])
    imgs = [img.cuda().half() for img in data['img']]
    # img_metas = data['img_metas']
    img_metas = data['img_metas'][0].data[0]

    feat = model.extract_feat(imgs[0])
    # print(data['img_metas'][0].data)
    # proposal_list = det_bboxes[:, :4]

    # test_cfg ={'score_thr': 0.0001,
               # 'nms': {'type': 'nms', 'iou_threshold': 0.0},
               # 'nms': {'type':'soft_nms', 'iou_threshold': 0.3, 'min_score':0.05},
               # 'max_per_img': 100}

    # {
    #             score_thr=0.05,
    #             # nms=dict(type='nms', iou_threshold=0.5),
    #             nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
    #             max_per_img=100}

    # det_bboxes, det_labels = model.roi_head.simple_test_bboxes(
    #     feat, img_metas, proposal_list, model.test_cfg.rcnn, rescale=False)
    # print(proposal_list)
    # print(det_bboxes)
    # rois = bbox2roi(proposal_list)
    det_rois = bbox2roi([torch.tensor(det_bboxes)]).to(feat[0].device)
    # print('rois:', rois.shape, rois)
    # print('det_rois:', det_rois.shape, det_rois)


    bbox_results = model.roi_head._bbox_forward(feat, det_rois)
    img_shapes = tuple(meta['img_shape'] for meta in img_metas)
    scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

    # split batch bbox prediction back to each image
    cls_score = bbox_results['cls_score']
    bbox_pred = bbox_results['bbox_pred']
    # print('rois:', det_rois.shape, cls_score.shape, bbox_pred.shape)
    num_proposals_per_img = tuple(len(p) for p in [det_bboxes])
    # rois = det_rois.split(num_proposals_per_img, 0)
    cls_score = cls_score.split(num_proposals_per_img, 0)
    for i in range(len([det_bboxes])):
        scores = F.softmax(
            cls_score[i], dim=-1) if cls_score[i] is not None else None
        # print(scores)
        for j in range(len(scores)):
            score_mask = scores[j] < 0.05
            # print(scores[j][score_mask])
            print(CLASSES[score_mask.cpu().numpy()])
    # some detector with_reg is False, bbox_pred will be None
    # if bbox_pred is not None:
    #     # TODO move this to a sabl_roi_head
    #     # the bbox prediction of some detectors like SABL is not Tensor
    #     if isinstance(bbox_pred, torch.Tensor):
    #         bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
    #     else:
    #         bbox_pred = model.roi_head.bbox_head.bbox_pred_split(
    #             bbox_pred, num_proposals_per_img)
    # else:
    #     bbox_pred = (None,) * len(proposal_list)
    #
    # # apply bbox post-processing to each image individually
    # det_bboxes = []
    # det_labels = []
    # for i in range(len(proposal_list)):
    #     if rois[i].shape[0] == 0:
    #         # There is no proposal in the single image
    #         det_bbox = rois[i].new_zeros(0, 5)
    #         det_label = rois[i].new_zeros((0,), dtype=torch.long)
    #         if test_cfg is None:
    #             det_bbox = det_bbox[:, :4]
    #             det_label = rois[i].new_zeros(
    #                 (0, model.roi_head.bbox_head.fc_cls.out_features))
    #
    #     else:
    #         # det_bbox, det_label = model.roi_head.bbox_head.get_bboxes(
    #         #     rois[i],
    #         #     cls_score[i],
    #         #     bbox_pred[i],
    #         #     img_shapes[i],
    #         #     scale_factors[i],
    #         #     rescale=False,
    #         #     cfg=test_cfg)
    #         if model.roi_head.bbox_head.custom_cls_channels:
    #             scores = model.roi_head.bbox_head.loss_cls.get_activation(cls_score[i])
    #         else:
    #             scores = F.softmax(
    #                 cls_score[i], dim=-1) if cls_score[i] is not None else None
    #         # bbox_pred would be None in some detector when with_reg is False,
    #         # e.g. Grid R-CNN.
    #
    #         if bbox_pred[i] is not None:
    #             bboxes = model.roi_head.bbox_head.bbox_coder.decode(
    #                 rois[i][..., 1:], bbox_pred[i], max_shape=img_shapes[i])
    #         else:
    #             bboxes = rois[i][:, 1:].clone()
    #             if img_shapes[i] is not None:
    #                 bboxes[:, [0, 2]].clamp_(min=0, max=img_shapes[i][1])
    #                 bboxes[:, [1, 3]].clamp_(min=0, max=img_shapes[i][0])
    #
    #         if False and bboxes.size(0) > 0:
    #             scale_factor = bboxes.new_tensor(scale_factors[i])
    #             bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
    #                 bboxes.size()[0], -1)
    #
    #         print('bboxes:', scores.shape, bboxes.shape)
    #         num_classes = scores.size(1) - 1
    #         # exclude background category
    #         if bboxes.shape[1] > 4:
    #             bboxes = bboxes.view(scores.size(0), -1, 4)
    #         else:
    #             bboxes = bboxes[:, None].expand(
    #                 scores.size(0), num_classes, 4)
    #
    #         # scores = scores[:, :-1]
    #         scores = scores[:, :-1]
    #
    #
    #         labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    #         labels = labels.view(1, -1).expand_as(scores)
    #
    #         bboxes = bboxes.reshape(-1, 4)
    #         print('socres;', scores.shape)
    #         scores = scores.reshape(-1)
    #         print('socres;', scores.shape)
    #
    #         labels = labels.reshape(-1)
    #
    #         bboxes = torch.cat([bboxes, scores[:, None]], -1)
    #
    #         det_bbox, det_label = bboxes, labels
    #         # det_bboxes1, det_labels1 = multiclass_nms(bboxes, scores,
    #         #                                           test_cfg['score_thr'],
    #         #                                           test_cfg['nms'],
    #         #                                           test_cfg['max_per_img'])
    #         #
    #         # det_bbox, det_label = det_bboxes1, det_labels1
    #         # print(det_bbox.shape, det_bbox)
    #
    #     det_bboxes.append(det_bbox)
    #     det_labels.append(det_label)
    #
    #
    #
    #
    #
    #
    #
    # bbox_results = [
    #     bbox2result(det_bboxes[i], det_labels[i],
    #                 20)
    #     for i in range(len(det_bboxes))
    # ]
    # return bbox_results

if __name__ == '__main__':
    main()
