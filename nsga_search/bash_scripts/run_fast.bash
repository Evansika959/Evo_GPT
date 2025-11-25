#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python run_exp.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/host_east4.yaml \
    --search_space_config ./search_space_def/small_search_space.yaml \
    --max_layers 4 \
    --min_layers 2 \
    --pop_size 4 \
    --offspring 4 \
    --generations 2 \
    --exp_name infi_hw_try \
    --conda_env reallmforge \
    --max_iters 100 \
    2>&1 | tee -a "$log"