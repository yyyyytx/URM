#!/usr/bin/env bash

CONFIG=/home/liu/ytx/SS-OD/configs/_base_/datasets/voc_baseline_cocofmt.py
WORK_DIR=/home/liu/ytx/SS-OD/outputs/baseline/faster_rcnn_voc_cocofmt/


export PYTHONPATH=$PYTHONPATH:/home/liu/ytx/SS-OD
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=9500 \
# /home/liu/ytx/SS-OD/tools/full_train.py ${CONFIG} --seed 0 --work-dir=${WORK_DIR} --launcher pytorch
python  /home/liu/ytx/SS-OD/tools/full_train.py ${CONFIG} --seed 0 --work-dir=${WORK_DIR}