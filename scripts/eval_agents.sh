#!/bin/bash

OUT_DIR=$1
EVAL_XML_PATH=$2
ENV_TYPE=$3
NUM_NODES=$4
NODE_ID=$5

cd derl
PYTHONPATH=. python tools/evolution.py \
            --cfg configs/eval/$ENV_TYPE.yml \
            --OUT_DIR $OUT_DIR \
            EVO.IS_EVO_TASK True \
            INIT_EVAL False \
            EVO.NUM_PROCESSES 5 \
            EVO.USE_EXIST_AGENT True \
            NUM_NODES $NUM_NODES \
            ENV_TYPE "$ENV_TYPE" \
            NODE_ID $NODE_ID \
            EVAL_XML_PATH "$EVAL_XML_PATH"
