# How to Run

## Prerequisites

```bash
pip install -r requirements.txt
```

---

## Original LatentMAS (unchanged behaviour)

### Single agent baseline

```bash
python run.py \
  --method baseline \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --max_samples 100 \
  --max_new_tokens 2048
```

### Text-based multi-agent (TextMAS)

```bash
python run.py \
  --method text_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --max_samples 100 \
  --max_new_tokens 2048
```

### Latent multi-agent (LatentMAS) — HuggingFace path

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 20 \
  --max_samples 100 \
  --max_new_tokens 2048
```

### Latent multi-agent (LatentMAS) — vLLM path (faster, needs 2 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 20 \
  --use_vllm \
  --use_second_HF_model \
  --enable_prefix_caching \
  --device cuda:0 \
  --device2 cuda:1 \
  --max_samples 100 \
  --max_new_tokens 2048
```

### With latent-space realignment

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 20 \
  --latent_space_realign \
  --max_samples 100 \
  --max_new_tokens 2048
```

---

## LAKV — LatentMAS with KV-Cache Compression (this branch)

### Step 1 — Run calibration once per model (offline)

Produces a `layer_profile_*.json` that tells the compressor which layers to keep at INT8, INT4, or drop entirely. Only needs to run once; reuse the file across experiments.

```bash
python -m compression.calibrate \
  --model_name Qwen/Qwen3-8B \
  --n_samples 50 \
  --latent_steps 20 \
  --device cuda \
  --output calibration_artifacts/layer_profile_Qwen3-8B.json
```

For other model sizes:

```bash
# 4B
python -m compression.calibrate --model_name Qwen/Qwen3-4B --output calibration_artifacts/layer_profile_Qwen3-4B.json

# 14B
python -m compression.calibrate --model_name Qwen/Qwen3-14B --output calibration_artifacts/layer_profile_Qwen3-14B.json
```

---

### Step 2 — Run with compression

#### Adaptive compression (full LAKV: INT8 / INT4 / layer drop)

Uses the calibration profile for per-layer tier assignment. Best compression ratio (~8x), minimal accuracy loss.

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 20 \
  --max_samples 100 \
  --max_new_tokens 2048 \
  --compression_mode adaptive \
  --calibration_file calibration_artifacts/layer_profile_Qwen3-8B.json
```

#### Uniform INT8 (all surviving layers at INT8, no layer dropping)

Lighter compression (~2x), useful as an ablation or when no calibration file is available.

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 20 \
  --max_samples 100 \
  --max_new_tokens 2048 \
  --compression_mode uniform_int8
```

> `--calibration_file` is optional for `uniform_int8`. If omitted, a heuristic tier assignment is used automatically.

#### Adaptive compression + vLLM path

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 20 \
  --use_vllm \
  --use_second_HF_model \
  --enable_prefix_caching \
  --device cuda:0 \
  --device2 cuda:1 \
  --max_samples 100 \
  --max_new_tokens 2048 \
  --compression_mode adaptive \
  --calibration_file calibration_artifacts/layer_profile_Qwen3-8B.json
```

---

## Benchmarking both modes side by side

Run these back to back and compare the `accuracy` and `time_per_sample_sec` fields in the JSON output.

```bash
# Baseline LatentMAS
python run.py --method latent_mas --model_name Qwen/Qwen3-8B --task gsm8k \
  --latent_steps 20 --max_samples 500 --max_new_tokens 2048 \
  --compression_mode none \
  > results_no_compression.json

# LAKV adaptive
python run.py --method latent_mas --model_name Qwen/Qwen3-8B --task gsm8k \
  --latent_steps 20 --max_samples 500 --max_new_tokens 2048 \
  --compression_mode adaptive \
  --calibration_file calibration_artifacts/layer_profile_Qwen3-8B.json \
  > results_lakv_adaptive.json
```

---

## Available tasks

| Flag            | Dataset                        |
| --------------- | ------------------------------ |
| `gsm8k`         | GSM8K math word problems       |
| `aime2024`      | AIME 2024 competition problems |
| `aime2025`      | AIME 2025 competition problems |
| `gpqa`          | GPQA-Diamond science questions |
| `arc_easy`      | ARC Easy                       |
| `arc_challenge` | ARC Challenge                  |
| `mbppplus`      | MBPP+ code generation          |
| `humanevalplus` | HumanEval+ code generation     |
| `medqa`         | MedQA medical reasoning        |

## Available models

| Flag value       | Size |
| ---------------- | ---- |
| `Qwen/Qwen3-4B`  | 4B   |
| `Qwen/Qwen3-8B`  | 8B   |
| `Qwen/Qwen3-14B` | 14B  |
