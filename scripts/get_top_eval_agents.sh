#!/bin/bash

OUT_DIR=$1
ENV_TYPE=$2
TOP_K=$3

cd derl
PYTHONPATH=./ python tools/evo_single_proc.py \
                --cfg configs/evo/ft.yml \
                --proc-id 0 \
                --select-top-k-agent-env \
                --top-k $TOP_K \
                OUT_DIR $OUT_DIR \
                ENV_TYPE $ENV_TYPE