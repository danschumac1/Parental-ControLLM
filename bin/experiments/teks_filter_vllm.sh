#!/usr/bin/env bash

# =============================================================================
# TEKS FILTER EXPERIMENTS
# =============================================================================
# Run:
#   chmod +x ./bin/experiments/teks_filter_vllm.sh
#   nohup ./bin/experiments/teks_filter_vllm.sh > ./logs/teks_filter_vllm.log 2>&1 &
#
# Monitor:
#   tail -f ./logs/teks_filter_vllm.log
#
# Requires:
#   - vLLM server running on localhost:8002
#   - ./resources/.env containing SLACK_WEBHOOK_URL
# =============================================================================

LOG_FILE="./logs/teks_filter_vllm.log"
source ./bin/_slack_traps.sh

# =============================================================================
# VLLM SERVER COMMANDS (REFERENCE ONLY)
# =============================================================================

: <<'SERVER_NOTES'

# GEMMA 4 12B                           3522583
nohup env \
    CUDA_VISIBLE_DEVICES=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve google/gemma-4-12B-it \
        --host 127.0.0.1 \
        --port 8001 \
        --gpu-memory-utilization 0.90 \
    > ./logs/gemma4_12b_server.log 2>&1 &

tail -f ./logs/gemma4_12b_server.log
nohup ./bin/experiments/teks_filter_vllm.sh > ./logs/teks_filter_gemma4_12b.log 2>&1 &
tail -f ./logs/teks_filter_gemma4_12b.log


# GEMMA 4 31B
nohup env \
    CUDA_VISIBLE_DEVICES=0,1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve google/gemma-4-31B-it \
        --host 127.0.0.1 \
        --port 8002 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 65536 \
    > ./logs/gemma4_31b_server.log 2>&1 &

tail -f ./logs/gemma4_31b_server.log


# LLAMA 3.3 70B
nohup env \
    CUDA_VISIBLE_DEVICES=0,1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve meta-llama/Llama-3.3-70B-Instruct \
        --host 127.0.0.1 \
        --port 8002 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 65536 \
    > ./logs/llama3_70b_server.log 2>&1 &

tail -f ./logs/llama3_70b_server.log


# QWEN 3.5 4B
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve Qwen/Qwen3.5-4B \
        --host 127.0.0.1 \
        --port 8002 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
    > ./logs/qwen35_4b_server.log 2>&1 &

tail -f ./logs/qwen35_4b_server.log


# QWEN 3.5 9B
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve Qwen/Qwen3.5-9B \
        --host 127.0.0.1 \
        --port 8002 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
    > ./logs/qwen35_9b_server.log 2>&1 &

tail -f ./logs/qwen35_9b_server.log


# MINISTRAL 3 8B
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve mistralai/Ministral-3-8B-Instruct-2512 \
        --host 127.0.0.1 \
        --port 8002 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --tokenizer-mode mistral \
        --config-format mistral \
        --load-format mistral \
    > ./logs/ministral3_8b_server.log 2>&1 &

tail -f ./logs/ministral3_8b_server.log


# MINISTRAL 3 14B
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve mistralai/Ministral-3-14B-Instruct-2512 \
        --host 127.0.0.1 \
        --port 8002 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --tokenizer-mode mistral \
        --config-format mistral \
        --load-format mistral \
    > ./logs/ministral3_14b_server.log 2>&1 &

tail -f ./logs/ministral3_14b_server.log

SERVER_NOTES

# =============================================================================
# BACKEND & MODELS
# =============================================================================

BACKEND="vllm"

MODELS=(
    # "google/gemma-4-12B-it"
    "google/gemma-4-31B-it"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "Qwen/Qwen3.5-4B"
    # "Qwen/Qwen3.5-9B"
    # "mistralai/Ministral-3-8B-Instruct-2512"
    # "mistralai/Ministral-3-14B-Instruct-2512"
    # "microsoft/phi-4"
)

# =============================================================================
# DATASETS
# =============================================================================

DATASETS=(
    # "situations"
    "eduQ"
)

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================

N_ROWS=-1
SAVE_EVERY=5

TEMPERATURE=0.0
MAX_TOKENS=1024
TOP_P=1.0

# =============================================================================
# SETUP
# =============================================================================

mkdir -p ./logs

echo "=============================================="
echo "STARTING TEKS FILTER EXPERIMENTS"
echo "=============================================="
echo "Host:     $(hostname)"
echo "Backend:  ${BACKEND}"
echo "Models:   ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Log:      ${LOG_FILE}"
echo "=============================================="

if [ ${#MODELS[@]} -eq 0 ] || [ ${#DATASETS[@]} -eq 0 ]; then
    echo "ERROR: No models or datasets selected."
    exit 1
fi

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

for dataset in "${DATASETS[@]}"; do
    INPUT_DATA_PATH="./data/generated/${dataset}.tsv"

    if [ ! -f "${INPUT_DATA_PATH}" ]; then
        echo "ERROR: Dataset not found: ${INPUT_DATA_PATH}"
        exit 1
    fi

    for model in "${MODELS[@]}"; do
        echo "----------------------------------------------"
        echo "STARTING: ${model}"
        echo "DATASET:  ${dataset}"
        echo "BACKEND:  ${BACKEND}"
        echo "----------------------------------------------"

        run "TEKS Filter | Model: ${model} | Dataset: ${dataset}" \
            python ./src/experiments/teks_filter.py \
                --input_data_path "${INPUT_DATA_PATH}" \
                --backend "${BACKEND}" \
                --model "${model}" \
                --n_rows "${N_ROWS}" \
                --save_every "${SAVE_EVERY}" \
                --temperature "${TEMPERATURE}" \
                --max_tokens "${MAX_TOKENS}" \
                --top_p "${TOP_P}"

        echo "FINISHED: ${model} | ${dataset}"
        echo "----------------------------------------------"

        sleep 2
    done
done

# =============================================================================
# SUCCESS NOTIFICATION
# =============================================================================

echo "=============================================="
echo "TEKS FILTER EXPERIMENTS COMPLETE"
echo "=============================================="

./bin/_notify_slack.sh "SUCCESS" \
"✅ TEKS Filter experiments completed successfully
Script: $(basename "$0")
Host: $(hostname)
Backend: ${BACKEND}
Models: ${MODELS[*]}
Datasets: ${DATASETS[*]}
Log: ${LOG_FILE}"