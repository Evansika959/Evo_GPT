#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python run_exp.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/host_set2.yaml \
    --max_layers 36 \
    --min_layers 2 \
    --pop_size 16 \
    --offspring 8 \
    --generations 50 \
    --exp_name infi_val_optimized \
    --conda_env reallmforge \
    --max_iters 10000 \
    2>&1 | tee -a "$log"