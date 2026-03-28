You are an expert ML engineer working with HuggingFace Transformers and PyTorch.

Goal
Implement “checkpoint morphing + uptraining” on a pretrained Qwen3 ~1B-class decoder-only LLM (I will use Qwen/Qwen3-1.7B or Qwen/Qwen3-1.7B-Base). The purpose is to modify the pretrained model into my proposed IHA (Infinite-Head Attention) style architecture to reduce KV-cache cost and potentially reduce parameters / hardware cost for edge deployment.

Key Qwen3 implementation details you must respect
- Qwen3 uses RoPE, RMSNorm, and a gated MLP (gate_proj, up_proj, down_proj).
- Qwen3 attention in HF uses separate q_proj, k_proj, v_proj, o_proj and supports GQA via num_key_value_heads.
- Past_key_values / cache must work after morphing (KV-cache stores K/V with num_kv_heads).

What I want to implement (two morph operations)

(1) Attention morph (per-layer KV-head schedule; “IHA via GQA-like KV sharing”)
- Starting from the pretrained Qwen3 model, I want to change the number of KV heads per layer:
  - Keep num_attention_heads (query heads) unchanged.
  - Reduce num_key_value_heads per layer according to a schedule I provide:
    * Example schedule: early layers keep 8 KV heads, later layers drop to 4, then 2 (must always divide num_attention_heads).
- Weight mapping requirement: DO NOT randomly reinitialize K/V when reducing KV heads.
  - Construct each new KV head by mean-pooling the original KV heads within a group.
  - Implement this by reshaping k_proj.weight and v_proj.weight to [num_kv, head_dim, hidden_size] (or equivalent), then grouping and averaging along the num_kv dimension.
  - Handle bias if present (Qwen3 attention_bias is usually false, but code must be robust).
- After morphing:
  - Each layer’s attention module must have k_proj/v_proj output dim = (num_kv_heads[layer] * head_dim).
  - The attention forward must expand/repeat KV to match Q heads (GQA repeat_kv pattern) and produce identical final attn_output shape.
  - Cache must store K/V with num_kv_heads[layer] (smaller KV-cache).
- IMPORTANT: HuggingFace’s stock Qwen3Config has a single global num_key_value_heads.
  - You must implement a custom config/model that supports per-layer KV heads.
  - Approach: store “num_key_value_heads_per_layer” list in config, and modify the layer init to use that per-layer value for k_proj/v_proj dims and num_key_value_groups.

(2) MLP morph (per-layer FFN width schedule; staged allocation)
- Qwen3 MLP is gated: gate_proj: [hidden_size -> intermediate_size], up_proj: [hidden_size -> intermediate_size], down_proj: [intermediate_size -> hidden_size].
- I want per-layer intermediate_size (FFN width) following a schedule I provide (e.g., early smaller, mid larger, late baseline).
- Shrink case (intermediate_size_new < old):
  - Do structured neuron selection from pretrained weights:
    * Compute L2 norm of columns of gate_proj.weight (and/or up_proj.weight).
    * Pick top-k indices.
  - Slice consistently:
    * gate_proj.weight: select rows? (careful with HF Linear layout) Ensure final shapes match [intermediate_size_new, hidden_size] or [hidden_size, intermediate_size_new] depending on weight layout.
    * up_proj.weight: same indices
    * down_proj.weight: slice corresponding input dimension
  - Bias handling: Qwen3 MLP is typically bias=False, but code should be robust.
- Expand case (intermediate_size_new > old):
  - Copy old weights into the leading block and initialize new neurons safely (small init / near-zero).
- Like KV heads, HuggingFace stock Qwen3 uses a single global intermediate_size.
  - Implement custom config/model to support “intermediate_size_per_layer”.

Repo / file deliverables (produce complete runnable code)
Create a small folder with:

1) iha_config.py
- Defines and validates per-layer schedules:
  - num_key_value_heads_per_layer: list[int] length = num_hidden_layers
  - intermediate_size_per_layer: list[int] length = num_hidden_layers
- Validations:
  - num_attention_heads % num_kv_heads[layer] == 0 for all layers
  - head_dim consistent
  - intermediate sizes are positive and reasonable

2) modeling_qwen3_iha.py and configuration_qwen3_iha.py
- Implement Qwen3IHAConfig extending Qwen3Config (or wrapping it):
  - Add fields: num_key_value_heads_per_layer, intermediate_size_per_layer
- Implement Qwen3IHAForCausalLM / Qwen3IHAModel:
  - DecoderLayer init uses per-layer KV heads and per-layer MLP dims.
  - Attention module is a modified Qwen3Attention supporting per-layer KV heads.
  - MLP module is a modified Qwen3MLP supporting per-layer intermediate_size.
- Ensure:
  - AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True) can load the morphed checkpoint.
  - Generation works.

3) morph_qwen3.py
- Loads base pretrained Qwen3 model from HF (I will provide model_id).
- Builds target IHA schedules (from iha_config.py).
- Applies:
  - KV head morph per layer (mean pooling K/V groups)
  - MLP resize per layer (slice/expand gate_proj/up_proj/down_proj)
- Saves:
  - morphed model weights (safetensors ok)
  - updated config (with per-layer lists)
  - tokenizer files copied (so it’s a loadable HF directory)
- Prints:
  - params before/after
  - estimated KV-cache bytes/token at context length 4096 and 8192

4) verify_shapes.py
- Loads original model and morphed model.
- Runs a forward pass on a small dummy batch (random token IDs or a short prompt).
- Verifies:
  - logits shape [B, T, vocab]
  - attention internal shapes are consistent (at least sanity checks)
  - past_key_values works:
    * run 2-step decode (prefill with prompt, then one token with cache)
    * ensure cache stores num_kv_heads[layer] and grows in sequence length correctly
  - no NaNs

5) uptrain.py
- Implements short continued pretraining (“uptraining”) on a dataset I provide (minipile or small text).
- Two-phase schedule:
  Phase 1 (stabilize):
    - freeze all weights except modified parameters:
      * attention k_proj and v_proj (and maybe o_proj if needed)
      * MLP gate_proj, up_proj, down_proj in layers that changed
    - train for N steps with low LR
  Phase 2 (adapt):
    - unfreeze all weights
    - train for M steps with conservative LR schedule
- Use HF Trainer or a clean PyTorch loop (either is fine).
- Support bf16/fp16, gradient clipping, and saving checkpoints.

6) README.md
- Exact commands:
  - pip install requirements (transformers version must support Qwen3)
  - morph_qwen3.py --model_id Qwen/Qwen3-1.7B --out_dir ./morphed_output --schedule examples/...
  - verify_shapes.py --model_dir ./morphed_output
  - uptrain.py --model_dir ./morphed_output --data ... --output_dir ...

Constraints / scope
- Focus on correctness and HF-loadability first.
- Do NOT suggest unrelated compression (LoRA etc.). This project is specifically about architecture morphing to IHA + uptraining.
- Default morph example:
  - Start from Qwen3-1.7B (num_attention_heads=16, num_key_value_heads=8, head_dim=128).
  - Modify later layers to num_kv_heads=4 or 2.
  - Shrink early-layer MLP intermediate_size (e.g., 6144 -> 4096) while keeping later layers at 6144.
- Provide good comments and docstrings.

Now produce the full code and README.