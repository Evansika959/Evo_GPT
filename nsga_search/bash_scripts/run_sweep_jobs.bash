#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_sweep_jobs_${ts}.log"

python sweep_jobs.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts-file ../host_configs/hosts_3instances.yaml \
    --config_yaml tests/grid_qk_trend2.yaml \
    --run_dir_name run_sweep_qk_2 \
    --max_iters 20000 \
"$@" 2>&1 | tee -a "$log"
