#!/usr/bin/env bash

CONFIG="/home/liu/ytx/SS-OD/configs/NegativeLoss/rcnn_voc.py"
WORK_DIR="/home/liu/ytx/SS-OD/outputs/NegativeLoss/rcnn_voc_neg_test"
GPUS=1

PROJECT_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:/home/liu/ytx/SS-OD
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=9501  \
    /home/liu/ytx/SS-OD/tools/BurnInTS_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0  --launcher pytorch
#    --resume-from /home/liu/ytx/SS-OD/outputs/NegativeLoss/rcnn_voc_neg_unreliable_0.3_thr_0.0005/epoch_2.pth

#python -m torch.utils.bottleneck /home/liu/ytx/SS-OD/tools/BurnInTS_train.py $CONFIG --work-dir=${WORK_DIR} --seed 0
