#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python run_exp.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/hosts_east4a.yaml \
    --max_layers 32 \
    --min_layers 2 \
    --pop_size 64 \
    --offspring 32 \
    --generations 20 \
    --exp_name infi_large_pop \
    --conda_env reallmforge \
    --max_iters 10000 \
    --objectives params val_loss \
    --constraint params=800000000 \
    --constraint val_loss=3.2 \
    2>&1 | tee -a "$log"