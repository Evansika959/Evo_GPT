#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_reallmasic_${ts}.log"

python run_reallmasic_exp.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/host_for_reallmasic.yaml \
    --search_space_config search_space_def/hw_constrained_space.yaml \
    --pop_size 16 \
    --max_layers 24 \
    --min_layers 2 \
    --offspring 8 \
    --generations 50 \
    --exp_name infi_reallmasic_all_causal_fixed \
    --conda_env reallmforge \
    --max_iters 10000 \
    2>&1 | tee -a "$log"
