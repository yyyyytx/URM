#!/usr/bin/env bash

CONFIG="/home/liu/ytx/SS-OD/configs/BurnInTS/faster_rcnn_coco_10.py"
WORK_DIR="/home/liu/ytx/SS-OD/outputs/BurnInTS/faster_rcnn_coco_10"
GPUS=2

PROJECT_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:/home/liu/ytx/SS-OD
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=9500  \
    /home/liu/ytx/SS-OD/tools/BurnInTS_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0 --launcher pytorch
#python -m torch.utils.bottleneck /home/liu/ytx/SS-OD/tools/BurnInTS_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0