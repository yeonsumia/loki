#!/bin/bash

BATCH_SIZE=4096
H_DIM=32
D_DEPTH=32
WDS=1e-5
LR=1e-4
NUM_LAYER=4
NUM_HEAD=4
FACTOR=8
NUM_EPOCHS=200

LOG_PATH="./log/train_vae"
mkdir -p $LOG_PATH

LOG_FILE="$LOG_PATH/VAE_500k_hdim${H_DIM}_depth${D_DEPTH}_LR_${LR}_WD_${WDS}_L${NUM_LAYER}_H${NUM_HEAD}_F${FACTOR}_beta0.01_bsize${BATCH_SIZE}_epochs${NUM_EPOCHS}.log"
PYTHONPATH=. python vae/train.py \
                        --gpu 0 \
                        --lr 1e-4 \
                        --batch_size $BATCH_SIZE \
                        --min_beta 1e-5 \
                        --max_beta 1e-2 \
                        --epochs $NUM_EPOCHS \
                        --factor $FACTOR \
                        --n_head $NUM_HEAD \
                        --n_layer $NUM_LAYER \
                        --d_depth $D_DEPTH \
                        --h_dim $H_DIM \
                        --wds >> $LOG_FILE 2>&1 &