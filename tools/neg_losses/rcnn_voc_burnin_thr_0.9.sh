#!/usr/bin/env bash

CONFIG="/home/liu/ytx/SS-OD/configs/NegativeLoss/rcnn_voc_burnin_0.9.py"
WORK_DIR="/home/liu/ytx/SS-OD/outputs/NegativeLoss/60k_rcnn_voc_burnin_0.9_4batch"
GPUS=2

PROJECT_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:/home/liu/ytx/SS-OD
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=9501  \
    /home/liu/ytx/SS-OD/tools/BurnInTS_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0  --launcher pytorch