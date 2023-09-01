#!/usr/bin/env bash

CONFIG="/home/liu/ytx/SS-OD/configs/pretrain/rcnn_dota.py"
WORK_DIR="/home/liu/ytx/SS-OD/outputs/dota_pretrain"
GPUS=2

PROJECT_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:/home/liu/ytx/SS-OD
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=9501  \
    /home/liu/ytx/SS-OD/tools/baseline_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0  --launcher pytorch