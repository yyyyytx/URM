#!/usr/bin/env bash

CONFIG=../../configs/soft_teacher/faster_rcnn_1x_voc_3090x2.py
WORK_DIR=../../outputs/baseline_faster_rcnn_1x_voc_3090x2/

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
 ../baseline_train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher pytorch
#python ../baseline_train.py ${CONFIG} --work-dir=${WORK_DIR}