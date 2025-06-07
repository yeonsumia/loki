#!/bin/bash

OUT_DIR=$1 # webdataset

RNG_SEED=$2

MIN_SEQ_LEN=$3 # 4

MAX_SEQ_LEN=$4 # 10

INIT_POPULATION=$5 # 500k

cd derl
PYTHONPATH=. python tools/evolution.py --cfg configs/evo/ft.yml --OUT_DIR ./$OUT_DIR/ft --RNG_SEED $RNG_SEED --min_seq_len $MIN_SEQ_LEN --max_seq_len $MAX_SEQ_LEN --init_population_size $INIT_POPULATION

cd ..
PYTHONPATH=. python tools/pkl_2_vec_new.py $OUT_DIR/ft/unimal_init