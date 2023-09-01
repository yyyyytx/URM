#!/usr/bin/env bash

CONFIG="/home/liu/ytx/SS-OD/configs/URM/voc_urm_neg_weight_5.0.py"
WORK_DIR="/home/liu/ytx/SS-OD/outputs/TSSemi/voc_urm_neg_weight_5.0"
GPUS=2

PROJECT_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:/home/liu/ytx/SS-OD
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=9501  \
    /home/liu/ytx/SS-OD/tools/train_urm.py $CONFIG --work-dir=${WORK_DIR} --seed 0  --launcher pytorch