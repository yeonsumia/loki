#!/bin/bash

NUM_WALKER=$1
NUM_CLUSTERS=$2
CLUSTER_LABEL=$3
RNG_SEED=$4
ENV_NAME=$5

DROP_FREQ=2
NUM_DROP=2

LOG_PATH="./log/finetune_loki/$ENV_NAME"
mkdir -p $LOG_PATH

LOG_FILE="$LOG_PATH/cluster${NUM_CLUSTERS}_idx${CLUSTER_LABEL}_walker${NUM_WALKER}_freq${DROP_FREQ}_drop${NUM_DROP}_seed$RNG_SEED.log"


CKPT_PATH="./output/loki/ft/kmeans_cluster/$NUM_CLUSTERS/$CLUSTER_LABEL/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED"
SAVE_PATH="./output/loki/$ENV_NAME/kmeans_cluster/$NUM_CLUSTERS/$CLUSTER_LABEL/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED"
XML_PATH="$CKPT_PATH/xml_step/14" # TODO: 1218

WALKER_PATH="$CKPT_PATH/xml_step"
mkdir -p metamorph/$WALKER_PATH/xml
cp "metamorph/$XML_PATH"/[0-9]*.xml metamorph/$WALKER_PATH/xml

# xml to pkl
PYTHONPATH=. python tools/xml_2_pkl.py --input_dir "metamorph/$WALKER_PATH"

cd metamorph
if [ "$ENV_NAME" = "many_obstacle" ]; then
    PYTHONPATH=./ python tools/train_ppo.py --cfg ./configs/obstacle.yaml \
                                        LOKI.FINETUNE True \
                                        OUT_DIR $SAVE_PATH \
                                        ENV.WALKER_DIR $WALKER_PATH \
                                        PPO.CHECKPOINT_PATH "$CKPT_PATH/Unimal-v0.pt" \
                                        MODEL.FINETUNE.FULL_MODEL True \
                                        ENV_TYPE $ENV_NAME \
                                        OBJECT.NUM_OBSTACLES 150 \
                                        > ../$LOG_FILE 2>&1 &
else
    PYTHONPATH=./ python tools/train_ppo.py --cfg ./configs/$ENV_NAME.yaml \
                                        LOKI.FINETUNE True \
                                        OUT_DIR $SAVE_PATH \
                                        ENV.WALKER_DIR $WALKER_PATH \
                                        PPO.CHECKPOINT_PATH "$CKPT_PATH/Unimal-v0.pt" \
                                        MODEL.FINETUNE.FULL_MODEL True \
                                        ENV_TYPE $ENV_NAME \
                                        > ../$LOG_FILE 2>&1 &
fi

                    


