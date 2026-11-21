#!/usr/bin/env bash
# Memo: evaluation commands (run as needed)

# =========================
# Local (mac/linux)
# =========================

# Generate dataset v0.1 (HF Dataset.save_to_disk format) (optional)
# python scripts/generate_dataset_v0_1.py \
#   --output-dir datasets/generated_datasets/v0.1 \
#   --negative-count 2000

# Shuffle v0.1 -> v0.1.1 (recommended for any --limit-based sweeps)
python scripts/shuffle_dataset_v0_1_to_v0_1_1.py \
  --input-dir datasets/generated_datasets/v0.1 \
  --output-dir datasets/generated_datasets/v0.1.1 \
  --seed 42

# Keyword baseline (base rules only)
python src/keyword_baseline_abnormal.py \
  --dataset-path datasets/generated_datasets/v0.1.1/dataset_hf

# Keyword baseline (+ external profanity list)
python src/keyword_baseline_abnormal.py \
  --dataset-path datasets/generated_datasets/v0.1.1/dataset_hf \
  --extra-wordlist configs/dsojevic_profanity_en.txt \
  --extra-weight 1

# Fineweb EDU baseline (may be slow / require GPU; requires deps + model availability)
# pip install -U transformers torch
python src/fineweb_edu_baseline_abnormal.py \
  --dataset-path datasets/generated_datasets/v0.1.1/dataset_hf \
  --model-id HuggingFaceFW/fineweb-edu-classifier \
  --device auto \
  --batch-size 32

# Neuron-level baseline (activation anomaly score; requires a causal LM + (ideally) GPU)
# Example model: Qwen/Qwen3.5-2B (large download)
python src/neuron_baseline_abnormal.py \
  --dataset-path datasets/generated_datasets/v0.1.1/dataset_hf \
  --model-id Qwen/Qwen3.5-2B \
  --device auto \
  --dtype auto \
  --layer -1 \
  --pooling mean \
  --method diag_mahalanobis \
  --batch-size 8 \
  --max-length 256

# Hybrid: keyword shortcut + neuron (aim: keep ACC/F1 while improving throughput)
python src/hybrid_keyword_neuron_abnormal.py \
  --dataset-path datasets/generated_datasets/v0.1.1/dataset_hf \
  --model-id Qwen/Qwen3.5-2B \
  --device auto \
  --dtype auto \
  --layer -11 \
  --pooling mean \
  --method persona_vector \
  --feature-normalize l2 \
  --batch-size 8 \
  --max-length 256 \
  --timing

# Hybrid: keyword abnormal shortcut + fineweb normal gate + neuron (best-effort throughput)
python src/hybrid_keyword_fineweb_neuron_abnormal.py \
  --dataset-path datasets/generated_datasets/v0.1.1/dataset_hf \
  --fineweb-model-id HuggingFaceFW/fineweb-edu-classifier \
  --fineweb-device auto \
  --fineweb-batch-size 32 \
  --fineweb-max-length 512 \
  --mode normal_only \
  --model-id Qwen/Qwen3.5-2B \
  --device auto \
  --dtype auto \
  --layer -11 \
  --pooling mean \
  --method persona_vector \
  --feature-normalize l2 \
  --batch-size 8 \
  --max-length 256 \
  --timing

# Throughput benchmark (samples/s) for all methods (keyword/fineweb/qwen neuron)
# Requires local weights for fineweb + qwen, so it's usually best to run on AutoDL.
# pip install -e .
# python scripts/benchmark_throughput.py \
#   --dataset-path datasets/generated_datasets/v0.1.1/dataset_hf \
#   --keyword-extra-wordlist configs/dsojevic_profanity_en.txt \
#   --fineweb-model-id HuggingFaceFW/fineweb-edu-classifier \
#   --fineweb-device auto \
#   --qwen-model-id Qwen/Qwen3.5-2B \
#   --qwen-device auto \
#   --qwen-layer -11

# Build seed+FineWeb mixed JSONL for complete token-anomaly binary evaluation.
# If httpx complains about missing socksio, clear ALL_PROXY/all_proxy and keep HTTP(S) proxy.
ALL_PROXY= all_proxy= PYTHONPATH=src python scripts/build_seed_fineweb_mixed_dataset.py \
  --seed-path datasets/third_datasets/seeds_filtered_fullL1.jsonl \
  --output-path /root/autodl-fs/datasets/FilterGuard/seed_fineweb_mixed/seed_fineweb_128_s42.jsonl \
  --seed-count 64 \
  --fineweb-count 64 \
  --seed 42 \
  --dev-ratio 0.2 \
  --fineweb-dataset HuggingFaceFW/fineweb-edu \
  --fineweb-config sample-10BT \
  --fineweb-split train \
  --fineweb-revision main

# Token NLL spike on seed+FineWeb mixed JSONL (writes reproducible artifacts).
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python src/token_anomaly_baseline_abnormal.py \
  --dataset-path /root/autodl-fs/datasets/FilterGuard/seed_fineweb_mixed/seed_fineweb_128_s42.jsonl \
  --method nll_spike \
  --model-id /root/autodl-fs/models/Qwen3-0.6B-Base \
  --trust-remote-code \
  --device cuda \
  --dtype auto \
  --batch-size 16 \
  --max-length 128 \
  --nll-score-mode max_z \
  --output-dir /root/autodl-fs/results/FilterGuard/token_anomaly \
  --run-id nll_spike_seed_fineweb_128_s42_qwen3_0p6b

# Length-normalized NLL spike variant for mixed short prompt + long web text data.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python src/token_anomaly_baseline_abnormal.py \
  --dataset-path /root/autodl-fs/datasets/FilterGuard/seed_fineweb_mixed/seed_fineweb_128_s42.jsonl \
  --method nll_spike \
  --model-id /root/autodl-fs/models/Qwen3-0.6B-Base \
  --trust-remote-code \
  --device cuda \
  --dtype auto \
  --batch-size 16 \
  --max-length 128 \
  --nll-score-mode max_z_len_norm \
  --output-dir /root/autodl-fs/results/FilterGuard/token_anomaly \
  --run-id nll_spike_len_norm_seed_fineweb_128_s42_qwen3_0p6b

# Token-count diagnostic mode. Strong on seed+FineWeb mixed data because seed
# prompts are much shorter than FineWeb normal documents; do not treat as a
# semantic anomaly detector without cross-source validation.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python src/token_anomaly_baseline_abnormal.py \
  --dataset-path /root/autodl-fs/datasets/FilterGuard/seed_fineweb_mix_4k.jsonl \
  --method nll_spike \
  --model-id /root/autodl-fs/models/Qwen3.5-0.8B-Base \
  --trust-remote-code \
  --device cuda \
  --dtype bf16 \
  --batch-size 32 \
  --max-length 128 \
  --nll-score-mode token_count \
  --output-dir /root/autodl-fs/results/FilterGuard/token_anomaly \
  --run-id nll_spike_token_count_seed_fineweb_mix_4k_qwen35_08b_maxlen128

# Rebuild a length-controlled v2 seed+FineWeb mix. FineWeb rows are clipped to
# the selected seed prompt token-length distribution under the Qwen3.5 tokenizer.
PYTHONPATH=src /root/miniconda3/bin/conda run -n kv --no-capture-output \
  python scripts/build_seed_fineweb_mixed_dataset.py \
    --seed-path datasets/third_datasets/seeds_filtered_fullL1.jsonl \
    --local-normal-jsonl /root/autodl-fs/datasets/fineweb-edu/sample/10BT/000_00000.parquet \
    --output-path /root/autodl-fs/datasets/FilterGuard/seed_fineweb_mix_v2_lenctl/seed_fineweb_mix_v2_lenctl_4k_s42_qwen35tok_m128.jsonl \
    --seed-count 2000 \
    --fineweb-count 2000 \
    --seed 42 \
    --dev-ratio 0.2 \
    --min-normal-chars 80 \
    --max-normal-chars 4000 \
    --length-control seed_token \
    --length-tokenizer-id /root/autodl-fs/models/Qwen3.5-0.8B-Base \
    --length-tokenizer-trust-remote-code \
    --length-max-tokens 128 \
    --length-min-tokens 2 \
    --length-text-scope user

# ONION-style token deletion on seed+FineWeb mixed JSONL (bounded candidates + pending deletions).
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python src/token_anomaly_baseline_abnormal.py \
  --dataset-path /root/autodl-fs/datasets/FilterGuard/seed_fineweb_mixed/seed_fineweb_128_s42.jsonl \
  --method onion_delta \
  --model-id /root/autodl-fs/models/Qwen3-0.6B-Base \
  --trust-remote-code \
  --device cuda \
  --dtype auto \
  --batch-size 16 \
  --deletion-batch-size 32 \
  --max-length 128 \
  --onion-score-mode topk_mean_delta \
  --top-k 3 \
  --onion-candidates 8 \
  --onion-max-pending-deletions 256 \
  --output-dir /root/autodl-fs/results/FilterGuard/token_anomaly \
  --run-id onion_delta_seed_fineweb_128_s42_qwen3_0p6b_c8

# =========================
# AutoDL / GPUHub (remote)
# =========================

AUTODL_HOST=connect.westc.gpuhub.com
AUTODL_PORT=13174
PROJECT=FilterGuard

# Prepare remote dirs (avoid writing system disk)
ssh -o StrictHostKeyChecking=no -p "$AUTODL_PORT" "root@${AUTODL_HOST}" \
  "mkdir -p /root/autodl-tmp/${PROJECT} \
    /root/autodl-fs/models \
    /root/autodl-fs/datasets/${PROJECT}/v0.1.1 \
    /root/autodl-fs/.cache/huggingface \
    /root/autodl-fs/.cache/torch"

# Copy code to /root/autodl-tmp (avoid writing system disk)
scp -o StrictHostKeyChecking=no -P "$AUTODL_PORT" -r src configs scripts docs main.py pyproject.toml .gitignore \
  "root@${AUTODL_HOST}:/root/autodl-tmp/${PROJECT}/"

# Copy generated dataset to /root/autodl-fs
scp -o StrictHostKeyChecking=no -P "$AUTODL_PORT" -r datasets/generated_datasets/v0.1.1/dataset_hf \
  "root@${AUTODL_HOST}:/root/autodl-fs/datasets/${PROJECT}/v0.1.1/"

# Download model to /root/autodl-fs/models and run fineweb baseline (CUDA)
ssh -o StrictHostKeyChecking=no -p "$AUTODL_PORT" "root@${AUTODL_HOST}" "PROJECT=${PROJECT} bash -s" <<'EOF'
set -e

source /etc/network_turbo
export https_proxy=http://localhost:7890

export XDG_CACHE_HOME=/root/autodl-fs/.cache
export HF_HOME=/root/autodl-fs/.cache/huggingface
export TORCH_HOME=/root/autodl-fs/.cache/torch

mkdir -p /root/autodl-fs/models/fineweb-edu-classifier

/root/miniconda3/bin/conda run -n kv --no-capture-output python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceFW/fineweb-edu-classifier",
    local_dir="/root/autodl-fs/models/fineweb-edu-classifier",
)
print("model_ready")
PY

cd "/root/autodl-tmp/${PROJECT}"

# Make src/ modules importable for scripts/benchmark_throughput.py (avoid PYTHONPATH hacks)
/root/miniconda3/bin/conda run -n kv --no-capture-output python -m pip install -e . --no-deps

# Avoid any accidental re-download during evaluation.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python src/fineweb_edu_baseline_abnormal.py \
    --dataset-path "/root/autodl-fs/datasets/${PROJECT}/v0.1.1/dataset_hf" \
    --model-id /root/autodl-fs/models/fineweb-edu-classifier \
    --device cuda \
    --batch-size 32

# Throughput benchmark (all methods)
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python scripts/benchmark_throughput.py \
    --dataset-path "/root/autodl-fs/datasets/${PROJECT}/v0.1.1/dataset_hf" \
    --keyword-extra-wordlist configs/dsojevic_profanity_en.txt \
    --fineweb-model-id /root/autodl-fs/models/fineweb-edu-classifier \
    --fineweb-device cuda \
    --qwen-model-id /root/autodl-fs/models/Qwen3.5-2B-Base \
    --qwen-device cuda \
    --qwen-layer -11 \
    --offline

# Hybrid: keyword shortcut + neuron (run on AutoDL)
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python src/hybrid_keyword_neuron_abnormal.py \
    --dataset-path "/root/autodl-fs/datasets/${PROJECT}/v0.1.1/dataset_hf" \
    --model-id /root/autodl-fs/models/Qwen3.5-2B-Base \
    --trust-remote-code \
    --device cuda \
    --dtype auto \
    --layer -11 \
    --pooling mean \
    --method persona_vector \
    --feature-normalize l2 \
    --batch-size 8 \
    --max-length 256 \
    --offline \
    --timing

# Hybrid: keyword abnormal shortcut + fineweb normal gate + neuron (run on AutoDL)
/root/miniconda3/bin/conda run -n kv --no-capture-output \
  python src/hybrid_keyword_fineweb_neuron_abnormal.py \
    --dataset-path "/root/autodl-fs/datasets/${PROJECT}/v0.1.1/dataset_hf" \
    --rules-config configs/keyword_rules_base.json \
    --fineweb-model-id /root/autodl-fs/models/fineweb-edu-classifier \
    --fineweb-device cuda \
    --fineweb-dtype auto \
    --fineweb-score-mode int0_5 \
    --fineweb-batch-size 32 \
    --fineweb-max-length 512 \
    --mode normal_only \
    --model-id /root/autodl-fs/models/Qwen3.5-2B-Base \
    --trust-remote-code \
    --device cuda \
    --dtype auto \
    --layer -11 \
    --pooling mean \
    --method persona_vector \
    --feature-normalize l2 \
    --batch-size 8 \
    --max-length 256 \
    --offline \
    --timing

# (Optional) Neuron-level baseline on AutoDL (download a CausalLM to /root/autodl-fs/models first)
# /root/miniconda3/bin/conda run -n kv --no-capture-output python - <<'PY'
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="Qwen/Qwen3.5-2B", local_dir="/root/autodl-fs/models/Qwen3.5-2B")
# print("model_ready")
# PY
# /root/miniconda3/bin/conda run -n kv --no-capture-output \
#   python src/neuron_baseline_abnormal.py \
#     --dataset-path "/root/autodl-fs/datasets/${PROJECT}/v0.1.1/dataset_hf" \
#     --model-id /root/autodl-fs/models/Qwen3.5-2B \
#     --device cuda \
#     --dtype auto \
#     --layer -1 \
#     --pooling mean \
#     --method diag_mahalanobis \
#     --batch-size 8 \
#     --max-length 256
EOF
