#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_kv_size_${ts}.log"

python run_exp_sw_only.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/host_set2.yaml \
    --search_space_config search_space_def/flexible_search_space.yaml \
    --resume_ckpt /home/xinting/Evo_GPT/optimization_and_search/nsga_search/ckpts/infi_flex_kv_size/pkl/1103_0143_pop_gen16.pkl \
    --pop_size 24 \
    --max_layers 24 \
    --min_layers 2 \
    --offspring 12 \
    --generations 34 \
    --exp_name infi_flex_kv_size \
    --conda_env reallmforge \
    --max_iters 10000 \
    2>&1 | tee -a "$log"
