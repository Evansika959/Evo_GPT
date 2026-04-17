# FineWeb-Edu Baseline Training Experiments

Train standard transformer baselines and NSGA-evolved architectures on FineWeb-Edu-10BT (~9B tokens) for comparison.

## Configuration files

Two YAML configs live in `config/`:

- **`nsga_models.yaml`** — NSGA-evolved heterogeneous architectures (5 models). All use: rotary embeddings, peri-LN, SwiGLU, `use_concat_heads=true`, per-layer attention variant (infinite + identity mix).
- **`standard_baselines.yaml`** — Canonical baselines (5 models): GPT2-Small (124M), Pythia-160M, SmoLLM2-135M, SmoLLM2-360M, Qwen-2.5-0.5B. Each keeps its own original recipe.

## Training configuration (common)

| Setting | Value |
|---------|-------|
| Dataset | fineweb-edu-sample-10BT (~9B train tokens) |
| Context length | 1024 |
| Batch size | 64 micro × 2 grad accum = 128 effective |
| Max iterations | 100,000 (~13.1B tokens seen) |
| LR | 3e-4 → 3e-5 (cosine) |
| Warmup | 2,000 iters |
| Precision | bfloat16 |
| Eval | every 2,500 iters, 200 batches |

## Local training (one host)

```bash
# Run both YAMLs sequentially
bash fineweb_baselines/scripts/run.sh

# Or individually
bash fineweb_baselines/scripts/run_nsga.sh
bash fineweb_baselines/scripts/run_baselines.sh

# Smoke test (1K iters)
bash fineweb_baselines/scripts/run_test.sh
```

## Distributed training (multi-host, manual dispatch)

```bash
# 1. Stage code + data to each remote host (one-time, ~20GB)
bash fineweb_baselines/scripts/remote_setup.sh user@host1 ~/.ssh/key
bash fineweb_baselines/scripts/remote_setup.sh user@host2 ~/.ssh/key

# 2. Launch one config on each host (detached via nohup)
bash fineweb_baselines/scripts/remote_launch.sh user@host1 nsga ~/.ssh/key
bash fineweb_baselines/scripts/remote_launch.sh user@host2 baselines ~/.ssh/key

# 3. Pull results back after training completes
bash fineweb_baselines/scripts/remote_collect.sh user@host1 nsga ~/.ssh/key
bash fineweb_baselines/scripts/remote_collect.sh user@host2 baselines ~/.ssh/key
```

## Plotting

```bash
python fineweb_baselines/plot_scripts/plot_loss_curves.py
```

## Directory structure

```
fineweb_baselines/
├── README.md
├── .gitignore                    # Ignores results/, plots/
├── config/
│   ├── nsga_models.yaml
│   └── standard_baselines.yaml
├── scripts/
│   ├── run.sh                    # Local: run both configs
│   ├── run_nsga.sh               # Local: NSGA models only
│   ├── run_baselines.sh          # Local: standard baselines only
│   ├── run_test.sh               # Local: 1K-iter smoke test
│   ├── remote_setup.sh           # Stage code + data to a remote host
│   ├── remote_launch.sh          # Launch a config on a remote host (nohup)
│   └── remote_collect.sh         # Rsync results back
├── plot_scripts/
│   └── plot_loss_curves.py
├── plots/                        # Generated figures (gitignored)
└── results_*/                    # Training logs + checkpoints (gitignored)
```

## Previous results summary (for reference)

| Model | Params | Best Val Loss | ARC-Easy | BoolQ |
|-------|--------|---------------|----------|-------|
| GPT2-Small | 124M | 3.0757 | 38.07% | 61.07% |
| Pythia-160M | 162M | 3.0438 | 37.02% | 56.06% |
| SmoLLM2-135M | 135M | 3.0105 | 40.00% | 58.81% |
| NSGA-Best1 | 167M | 2.9902 | 39.30% | 58.13% |
| NSGA-Best2 | 170M | 3.0101 | 38.60% | 58.44% |
| NSGA-Best3 | 105M | 3.1588 | 37.89% | 55.90% |
| NSGA-Best4 | 120M | 3.0986 | 36.14% | 59.42% |

Note: these earlier runs used abs_pos + pre_ln for NSGA models. The current `nsga_models.yaml` uses rotary + peri_ln — retraining is needed for fresh comparison.
