# Qwen3 IHA Morphing

This folder implements checkpoint morphing plus uptraining for a Qwen3 ~1B model into an IHA-style architecture with per-layer KV heads and MLP width.

## Install

```bash
pip install "transformers>=4.49.0" datasets torch
```

## Example schedule

```bash
cat > examples/schedule.json << 'EOF'
{
  "num_key_value_heads_per_layer": [8,8,8,8,8,8,8,4,4,4,2,2],
  "intermediate_size_per_layer": [4096,4096,4096,4096,4096,4096,6144,6144,6144,6144,6144,6144]
}
EOF
```

## Morph checkpoint

```bash
python3 morph_qwen3.py \
  --model_id Qwen/Qwen3-1.7B \
  --out_dir ./morphed_output \
  --schedule examples/schedule.json
```

## Verify shapes and cache

```bash
python3 verify_shapes.py --model_dir ./morphed_output --device auto
```

## Uptrain (two phases)

```bash
python3 uptrain.py \
  --model_dir ./morphed_output \
  --output_dir ./qwen3_iha_uptrain \
  --dataset_name allenai/c4 \
  --dataset_config_name en \
  --text_column text
```

## Compare PPL (standalone)

```bash
python3 compare_ppl.py \
  --model_dir ./morphed_output \
  --output_dir ./qwen3_iha_uptrain \
  --original_model_id Qwen/Qwen3-1.7B \
  --ppl_dtype bfloat16
```

## Notes

- KV head reduction uses mean-pooling of K/V heads within each group.
- MLP shrink uses top-k neuron selection by L2 norm of gate projection rows.
- Cache size is reduced proportional to per-layer KV heads.
