#!/bin/bash

NUM_CLUSTERS=$1
RNG_SEED=$2

BASE_DIR_BASE="./output/loki/ft/kmeans_cluster/$NUM_CLUSTERS"
DST_DIR="$BASE_DIR_BASE/top_100_agents/xml"
mkdir -p "$DST_DIR"

NUM_WALKER=20
DROP_FREQ=2
NUM_DROP=2

# Enumerate the top 2-3 agents indices for each cluster 
# (based on their training rewards in the multi-agent policy model)

# Example:
# declare -A IDX_MAP=(
#   [0]="14 9 1"
#   [1]="11 5"
#   [2]="13 17"
#   [3]="1 3 5"
#   [4]="0 5 3"
#   [5]="11 19 10"
#   [6]="18 14 0"
#   [7]="13 9 6"
#   [8]="6 7 11"
#   [9]="0 10 13"
#   [10]="7 18 10"
#   [11]=""
#   [12]="11 6 12"
#   [13]="3 7 8"
#   [14]="0 7 9"
#   [15]="11 17 14"
#   [16]="4 9 14"
#   [17]="17 6"
#   [18]="1 6"
#   [19]="11 13 16"
#   [20]="1 2 15"
#   [21]="17 6"
#   [22]=""
#   [23]="3 8 18"
#   [24]="6 7 15"
#   [25]="3 14 1"
#   [26]="12 19 17"
#   [27]="8 13"
#   [28]="2 9 8"
#   [29]="0 17"
#   [30]="13 1"
#   [31]="0 16"
#   [32]="11 9 2"
#   [33]="1 14 10"
#   [34]="0 10"
#   [35]="1 12"
#   [36]="5 8 16"
#   [37]="10 2 18"
#   [38]="1 8 15"
#   [39]="8 7"
# )

for CLUSTER_IDX in "${!IDX_MAP[@]}"; do
  IDX_LIST=(${IDX_MAP[$CLUSTER_IDX]})
  BASE_DIR="$BASE_DIR_BASE/$CLUSTER_IDX/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED/xml_step/1218/"
  for i in "${IDX_LIST[@]}"; do
    SRC_FILE="$BASE_DIR/$i.xml"
    DST_FILE="$DST_DIR/cluster${NUM_CLUSTERS}_idx${CLUSTER_IDX}_seed${SEED}_$i.xml"
    if [[ -f "$SRC_FILE" ]]; then
      cp "$SRC_FILE" "$DST_FILE"
      echo "Copied $SRC_FILE to $DST_FILE"
    else
      echo "Warning: $SRC_FILE not found"
    fi
  done
done