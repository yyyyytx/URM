#!/usr/bin/env bash

CONFIG=/home/liu/ytx/SS-OD/configs/baseline/faster_rcnn_coco_10.py
WORK_DIR=/home/liu/ytx/SS-OD/outputs/baseline/faster_rcnn_coco_10



python -m torch.distributed.launch --nproc_per_node=2 --master_port=9500 \
 /home/liu/ytx/SS-OD/tools/full_train.py ${CONFIG} --seed 0 --work-dir=${WORK_DIR} --launcher pytorch