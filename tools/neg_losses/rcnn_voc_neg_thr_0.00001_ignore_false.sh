#!/usr/bin/env bash

CONFIG="/home/liu/ytx/SS-OD/configs/NegativeLoss/rcnn_voc_neg_thr_0.00001_ignore_false.py"
WORK_DIR="/home/liu/ytx/SS-OD/outputs/NegativeLoss/9k_rcnn_voc_neg_unreliable_0.3_thr_0.00001_ignore_false"
GPUS=2

PROJECT_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:/home/liu/ytx/SS-OD
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=9501  \
    /home/liu/ytx/SS-OD/tools/BurnInTS_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0  --launcher pytorch