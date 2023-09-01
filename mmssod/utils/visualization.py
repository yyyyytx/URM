from mmcv.image import tensor2imgs
from mmcv.visualization import imshow_bboxes, imshow_det_bboxes
import torch
from mmcv import imdenormalize
import numpy as np

CLASSES = np.array(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor', 'BG'])

def visual_norm_imgs(img_tensor, bboxes, labels, img_meta, out_file=None):
    mean = np.array(img_meta['img_norm_cfg']['mean'], dtype=np.float32)
    std = np.array(img_meta['img_norm_cfg']['std'], dtype=np.float32)

    img = img_tensor.cpu().float().numpy().transpose(1, 2, 0)

    img = imdenormalize(
        img, mean, std, to_bgr=True).astype(np.uint8)

    # imshow_bboxes(img, bboxes.cpu().detach().numpy(), out_file=out_file)
    imshow_det_bboxes(img,
                      bboxes.cpu().detach().numpy(),
                      labels.cpu().detach().numpy(),
                      CLASSES,
                      show=False,
                      out_file=out_file)
    # if isinstance(img_tensor, torch.Tensor):
    # img_tensor = img_tensor.data[0]
    # assert len(img_tensors) == len(bboxes)
    # assert len(bboxes) == len(img_metas)
    # print(img_metas[0]['img_norm_cfg'])
    # imgs = tensor2imgs(img_tensors,
    #                    mean=img_metas[0]['img_norm_cfg']['mean'],
    #                    std=img_metas[0]['img_norm_cfg']['std'],
    #                    to_rgb=True)
    #
    #
    # for img_idx in range(len(img_tensors)):
    #     # h, w, _ = img_metas[['img_shape']
    #     # img_show = imgs[img_idx][:h, :w, :]
    #     imshow_det_bboxes(imgs[img_idx], bboxes[img_idx].numpy())

