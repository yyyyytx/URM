#!/usr/bin/env bash

CONFIG="/home/liu/ytx/SS-OD/configs/region_estimation/faster_rcnn_voc.py"
WORK_DIR="/home/liu/ytx/SS-OD/outputs/region_estimation/faster_rcnn_voc_thr_0.9"
GPUS=1

PROJECT_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:/home/liu/ytx/SS-OD
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=9501  \
    /home/liu/ytx/SS-OD/tools/BurnInTS_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0 --launcher pytorch
#python -m torch.utils.bottleneck /home/liu/ytx/SS-OD/tools/BurnInTS_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0