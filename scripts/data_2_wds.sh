#!/bin/bash

OUT_DIR=$1

PYTHONPATH=. python tools/data_2_wds.py ./derl/$OUT_DIR $OUT_DIR 