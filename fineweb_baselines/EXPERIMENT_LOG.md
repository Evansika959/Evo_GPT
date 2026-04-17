# FineWeb-Edu Baseline Training — Live Experiment Log

_Last updated: 2026-04-17 18:45_

## Machines

| Host | IP / Hostname | Current job | Status |
|------|---------------|-------------|--------|
| local | instance-h100-17 | **NSGA-Best2** (nsga_models.yaml row1, rotary+peri-ln) | 🔄 15K / 100K (loss 3.24) |
| host1 | 10.150.0.29 (instance-h100-20) | **NSGA-Best3** (host29_nsga_best3_best4.yaml) → NSGA-Best4 | 🔄 5K / 100K (loss 3.59) |
| host2 | 10.150.0.49 (instance-h100-19) | **SmoLLM2-360M** (host_smollm2_360m.yaml) | 🔄 25K / 100K (loss 3.02) |
| host3 | 10.150.0.48 (instance-h100-18) | reserved for other tasks | ⛔ released |
| host4 | 10.150.0.38 | SSH access blocked | ⛔ not set up |

## Completed models (saved to `/home/xinting/Evo_GPT_checkpoints_backup/`)

| Model | Params | Run | Val Loss | ARC-Easy | BoolQ | Ckpt path |
|-------|--------|-----|----------|----------|-------|-----------|
| GPT2-Small | 124M | Round 1 (abs_pos) | 3.0757 | 38.07% | 61.07% | `gpt2_small_124M/` |
| Pythia-160M | 162M | Round 1 | 3.0438 | 37.02% | 56.06% | `pythia_160M/` |
| Pythia-160M (retrain) | 162M | This round on host29 | 3.0405 | 36.84% | 56.09% | (on host 29) |
| SmoLLM2-135M | 135M | Round 1 | 3.0105 | 40.00% | 58.81% | `smollm2_135M/` |
| NSGA-Best1 (orig abs_pos) | 167M | Round 1 | 2.9902 | 39.30% | 58.13% | `nsga_best1_167M/` |
| **NSGA-Best1 (rotary+peri-ln)** | **166M** | This round local | **2.9596** | **40.53%** | 52.84% | `nsga_best1_rotary_periln_166M/` |
| NSGA-Best2 (orig) | 170M | Round 1 | 3.0101 | 38.60% | 58.44% | `nsga_best2_170M/` |
| NSGA-Best3 (orig) | 105M | Round 2 | 3.1588 | 37.89% | 55.90% | `nsga_best3_105M/` |
| NSGA-Best4 (orig) | 120M | Round 2 | 3.0986 | 36.14% | 59.42% | `nsga_best4_120M/` |

## Models still to train (rotary + peri-ln variants)

- **NSGA-Best2** (rotary+peri-ln) — **local, running** (was at 12.5K)
- **NSGA-Best3** (rotary+peri-ln) — **host29, running** (was at 2.5K)
- **NSGA-Best4** (rotary+peri-ln) — queued after Best3 on host29
- **NSGA-Best5** (rotary+peri-ln, 199M, 22L) — queued after Best2 on local (Best3/4 marked as skip)
- **SmoLLM2-360M** (362M) — **host49, running** (was at 22.5K)
- **Qwen-2.5-0.5B** (494M) — not started, no host assigned
- **GPT2-Small retrain on host 29** — KILLED (was at 25K), no need to redo (round 1 backup exists)

## Config files

- `config/nsga_models.yaml` — all 5 NSGA models (Best1..Best5), rotary + peri-ln (for local sequential run)
- `config/standard_baselines.yaml` — all 5 canonical baselines
- `config/host29_nsga_best3_best4.yaml` — Best3 + Best4 (host 29)
- `config/host_pythia_gpt2.yaml` — Pythia + GPT2-Small (was on host 29, replaced)
- `config/host_smollm2_360m.yaml` — SmoLLM2-360M (host 49)

## Timeline / Decisions

| Time | Event |
|------|-------|
| 2026-04-16 earlier | Launched NSGA-Best1 on local (rotary+peri-ln variant of round 1 models) |
| 2026-04-16 05:42 | Launched Pythia+GPT2 on host 29, SmoLLM2-360M on host 49 |
| 2026-04-16 13:55 | Launched NSGA-Best3+Best4 on host 48 |
| 2026-04-17 ~15:00 | NSGA-Best1 done (val_loss=2.96); local moved to NSGA-Best2 |
| 2026-04-17 ~15:00 | Pythia done on host29 (val_loss=3.04); GPT2-Small started |
| 2026-04-17 ~16:40 | Skipped Best3 & Best4 on local (marked in results_nsga.yaml) so local goes Best2 → Best5 |
| 2026-04-17 ~16:42 | Killed NSGA-Best3+Best4 on host 48 (host reserved for other tasks) |
| 2026-04-17 ~17:35 | Killed GPT2-Small retrain on host 29 (was at 25K); relaunched NSGA-Best3+Best4 on host 29 |
| 2026-04-17 ~17:43 | Evaluated NSGA-Best1 (rotary+peri-ln): ARC-Easy 40.53%, BoolQ 52.84% |
| 2026-04-17 ~17:44 | Host 29 now running NSGA-Best3+Best4 |

## Monitors running

- Status poll every 15 min: hosts 29 + 49 (task `bzzzydrov`)
- TB log auto-sync every 10 min: hosts 29 + 49 (task `b6103wbju`)
- Local train log tail: (task `bt2jqh6ll`)
