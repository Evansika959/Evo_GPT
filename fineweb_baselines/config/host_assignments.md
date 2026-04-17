# Host Assignment Plan

## Hosts in use (from /home/xinting/hosts.yaml)

| Host | IP | Status |
|------|----|----|
| local | (this machine, instance-h100-17) | Training NSGA models sequentially |
| host1 | 10.150.0.29 (instance-h100-20) | Training standard baselines |
| host2 | 10.150.0.49 (instance-h100-19) | Training SmoLLM2-360M |

Note: 10.150.0.48 was previously used for NSGA-Best3+Best4 but released per user request (reserved for other tasks). 10.150.0.38 was planned but SSH auth not set up.

## Current training assignments

### Local machine
Config: `nsga_models.yaml` with skip-marks in `results_nsga/results_nsga.yaml`
Sequence:
- **NSGA-Best1** (166M, 21L) ✅ DONE (val_loss=2.9596, ARC-Easy=40.53%, BoolQ=52.84%)
- **NSGA-Best2** (170M, 19L) 🔄 TRAINING
- NSGA-Best3 (105M, 7L) ⏭ SKIPPED (marked done in log)
- NSGA-Best4 (120M, 7L) ⏭ SKIPPED (marked done in log)
- **NSGA-Best5** (199M, 22L) ⏳ queued (runs after Best2)

### 10.150.0.29 (host_pythia_gpt2.yaml)
- **Pythia-160M** (162M) ✅ DONE
- **GPT2-Small** (124M) 🔄 TRAINING

### 10.150.0.49 (host_smollm2_360m.yaml)
- **SmoLLM2-360M** (362M) 🔄 TRAINING (heavy, grad checkpointing, ~40h total)

## Not yet trained

- **NSGA-Best3** (105M, 7L) — no host assigned yet
- **NSGA-Best4** (120M, 7L) — no host assigned yet
- **Qwen-2.5-0.5B** (494M) — no host assigned yet

## Training configuration (all models)

| Setting | Value |
|---------|-------|
| Dataset | fineweb-edu-sample-10BT |
| Max iterations | 100,000 |
| Batch size | 64 × 2 grad accum = 128 effective |
| LR | 3e-4 → 3e-5 cosine |
| Warmup | 2,000 iters |
| Eval interval | 2,500 |
| Precision | bfloat16 |

## NSGA model specifics (rotary + peri-ln + SwiGLU + concat heads)

All NSGA models in `nsga_models.yaml` use:
- `use_rotary_embeddings: true`
- `use_peri_ln: true` + `use_pre_ln: true`
- `mlp_variant: swiglu`
- `use_concat_heads: true`
- `bias: false`, `wte_weight_tying: true`
