#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_medium_hw_${ts}.log"

python run_exp.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/host_for_reallmasic.yaml \
    --search_space_config ./search_space_def/default_search_space.yaml \
    --resume_ckpt /home/xinting/Evo_GPT/optimization_and_search/nsga_search/ckpts/infi_hw_medium_corrected/pkl/1101_0311_pop_gen60.pkl \
    --max_layers 24 \
    --min_layers 2 \
    --pop_size 18 \
    --offspring 9 \
    --generations 40 \
    --exp_name infi_hw_med_continue \
    --conda_env reallmforge \
    --max_iters 10000 \
    2>&1 | tee -a "$log"