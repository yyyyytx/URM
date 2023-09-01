#!/usr/bin/env bash

CONFIG="/home/liu/ytx/SS-OD/configs/PosOnly/faster_rcnn_voc.py"
WORK_DIR="/home/liu/ytx/SS-OD/outputs/PosOnly/faster_rcnn_voc"
GPUS=2

PROJECT_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:/home/liu/ytx/SS-OD
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=9500  \
    /home/liu/ytx/SS-OD/tools/BurnInTS_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0 --resume-from /home/liu/ytx/SS-OD/outputs/BurnInTS/faster_rcnn_voc_burnin/epoch_6.pth --launcher pytorch
