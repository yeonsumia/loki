#!/bin/bash

NUM_WALKER=$1
NUM_CLUSTERS=$2
CLUSTER_LABEL=$3
RNG_SEED=$4
RESUME_ITER=$5

DROP_FREQ=2
NUM_DROP=2
NUM_SAMPLE=128

LOG_PATH="./log/train_loki_resume"
mkdir -p $LOG_PATH

LOG_FILE="$LOG_PATH/cluster${NUM_CLUSTERS}_idx${CLUSTER_LABEL}_walker${NUM_WALKER}_freq${DROP_FREQ}_drop${NUM_DROP}_seed$RNG_SEED.log"

CKPT_PATH="./output/loki/ft/kmeans_cluster/$NUM_CLUSTERS/$CLUSTER_LABEL/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED"

VAE_PATH="VAE_500k_hdim32_depth32_LR_0.0001_WD_1e-05_L4_H4_F8_beta0.01_bsize4096_epochs200"

cd metamorph
PYTHONPATH=./ python tools/train_loki.py \
                        --cfg ./configs/ft.yaml \
                        --vae_path $VAE_PATH \
                        OUT_DIR $CKPT_PATH \
                        LOKI.TRAIN True \
                        LOKI.NUM_WALKER $NUM_WALKER \
                        LOKI.NUM_DROP_WALKER $NUM_DROP \
                        LOKI.DROP_FREQ $DROP_FREQ \
                        PPO.MAX_STATE_ACTION_PAIRS 1e8 \
                        LOG_PERIOD 10 \
                        LOKI.NUM_CLUSTERS $NUM_CLUSTERS \
                        LOKI.CLUSTER_LABEL $CLUSTER_LABEL \
                        LOKI.DROP_WARMUP $DROP_FREQ \
                        LOKI.SAMPLE_SIZE $NUM_SAMPLE \
                        LOKI.MUTATE_SAMPLE False \
                        ENV.TYPE "ft" \
                        LOKI.INIT_DIR $CKPT_PATH \
                        LOKI.RESUME_ITER $RESUME_ITER \
                        PPO.CHECKPOINT_PATH "$CKPT_PATH/Unimal-v0.pt" \
                        RNG_SEED $RNG_SEED >> ../$LOG_FILE 2>&1 &
