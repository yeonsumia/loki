#!/bin/bash

NUM_CLUSTERS=$1

PYTHONPATH=./ python vae/latent_cluster.py \
                --gpu 0 \
                --ckpt_dir VAE_500k_hdim32_depth32_LR_0.0001_WD_1e-05_L4_H4_F8_beta0.01_bsize4096_epochs200 \
                --n_clusters $NUM_CLUSTERS \
                --make_cluster
