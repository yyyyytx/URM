#!/usr/bin/env bash

CONFIG=../../configs/fulltraining/faster_rcnn_1x_voc.py
WORK_DIR=../../outputs/full/faster_rcnn_voc/



python -m torch.distributed.launch --nproc_per_node=2 --master_port=9500 \
 ../full_train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher pytorch