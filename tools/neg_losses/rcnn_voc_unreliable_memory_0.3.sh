#!/usr/bin/env bash

CONFIG="/home/liu/ytx/SS-OD/configs/NegativeLoss/rcnn_voc_unreliable_label_memory_0.3.py"
WORK_DIR="/home/liu/ytx/SS-OD/outputs/NegativeLoss/rcnn_voc_unreliable_label_memory_0.3"
GPUS=2

PROJECT_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:/home/liu/ytx/SS-OD
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=9501  \
    /home/liu/ytx/SS-OD/tools/BurnInTS_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0  --launcher pytorch