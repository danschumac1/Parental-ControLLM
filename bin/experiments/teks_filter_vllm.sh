#!/bin/bash
# 63199
# chmod +x ./bin/experiments/teks_filter_vllm.sh
# Usage: ./bin/experiments/teks_filter_vllm.sh
# Background: nohup ./bin/experiments/teks_filter_vllm.sh > ./logs/teks_filter_vllm.log 2>&1 &
# Monitor: tail -f ./logs/teks_filter_vllm.log
'''
4139
nohup env \
    CUDA_VISIBLE_DEVICES=0,1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve google/gemma-4-31B-it \
        --host 127.0.0.1 \
        --port 8002 \
        --tensor-parallel-size 2 \
        --max-model-len 40960 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --gpu-memory-utilization 0.90 \
    > ./logs/vllm_server.log 2>&1 &

tail -f ./logs/vllm_server.log

'''
# -------------------------
# BACKEND & MODELS
# -------------------------

BACKEND="vllm"

MODELS=(
    # "google/gemma-4-12B-it"
    "google/gemma-4-31B-it"
    # "microsoft/phi-4"
    # "Qwen/Qwen3.5-9B"
    # "meta-llama/Llama-3.3-70B-Instruct"
)

# BACKEND="openai"
# MODELS=(
#     "gpt-5.4-nano"
# )

# -------------------------
# EXPERIMENT CONFIG
# -------------------------

INPUT_DATA_PATH="./data/generated/sample.tsv"

N_ROWS=-1
SAVE_EVERY=5

TEMPERATURE=0.0
MAX_TOKENS=1024
TOP_P=1.0

# -------------------------
# LOGGING
# -------------------------

mkdir -p ./logs

echo "Starting TEKS Filter Experiments"
echo "Backend: ${BACKEND}"
echo "Input: ${INPUT_DATA_PATH}"
echo "-----------------------------------"

# -------------------------
# RUN EXPERIMENTS
# -------------------------

for model in "${MODELS[@]}"; do

    echo "----------------------------------------------------------------"
    echo "STARTING: ${model}"
    echo "Backend: ${BACKEND}"
    echo "----------------------------------------------------------------"

    python ./src/experiments/teks_filter.py \
        --input_data_path "${INPUT_DATA_PATH}" \
        --backend "${BACKEND}" \
        --model "${model}" \
        --n_rows "${N_ROWS}" \
        --save_every "${SAVE_EVERY}" \
        --temperature "${TEMPERATURE}" \
        --max_tokens "${MAX_TOKENS}" \
        --top_p "${TOP_P}"

    if [ $? -ne 0 ]; then
        echo "ERROR: ${model}"
    else
        echo "FINISHED: ${model}"
    fi

    echo "----------------------------------------------------------------"

    sleep 2

done

echo "TEKS Filter Experiments Complete"