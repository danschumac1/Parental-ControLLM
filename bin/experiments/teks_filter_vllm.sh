#!/usr/bin/env bash

# =============================================================================
# TEKS FILTER EXPERIMENTS
# =============================================================================
# Run:
#   chmod +x ./bin/experiments/teks_filter_vllm.sh
#   nohup ./bin/experiments/teks_filter_vllm.sh > ./logs/teks_filter_launcher.log 2>&1 &
#
# Monitor:
#   tail -f ./logs/teks_filter_launcher.log
#
# Individual model logs are generated automatically.
# =============================================================================

LOG_FILE="./logs/teks_filter_launcher.log"
source ./bin/_slack_traps.sh

# =============================================================================
# VLLM SERVER COMMANDS (REFERENCE ONLY)
# =============================================================================
#region
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
    CUDA_VISIBLE_DEVICES=2 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve google/gemma-4-31B-it \
        --host 127.0.0.1 \
        --port 8002 \
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
#endregion

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
# GPU ASSIGNMENTS
# =============================================================================
# Port convention: GPU 0 -> 8001, GPU 1 -> 8002, etc.
# These assignments describe where the servers are already running.
# This script does NOT launch or move vLLM servers.
# =============================================================================

declare -A MODEL_GPUS=(
    ["google/gemma-4-12B-it"]=1
    ["google/gemma-4-31B-it"]=2
    ["meta-llama/Llama-3.3-70B-Instruct"]=0
    ["Qwen/Qwen3.5-4B"]=0
    ["Qwen/Qwen3.5-9B"]=0
    ["mistralai/Ministral-3-8B-Instruct-2512"]=0
    ["mistralai/Ministral-3-14B-Instruct-2512"]=0
    ["microsoft/phi-4"]=0
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
SAVE_EVERY=1

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
echo "=============================================="

if (( ${#MODELS[@]} == 0 || ${#DATASETS[@]} == 0 )); then
    run "Validate experiment selection" false
fi

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

for model in "${MODELS[@]}"; do

    if [[ ! -v MODEL_GPUS["$model"] ]]; then
        echo "ERROR: No GPU assigned to ${model}"
        run "Missing GPU assignment: ${model}" false
    fi

    GPU="${MODEL_GPUS[$model]}"
    PORT="800$((GPU))"
    SAFE_MODEL="${model//\//--}"

    for dataset in "${DATASETS[@]}"; do

        INPUT_DATA_PATH="./data/generated/${dataset}.tsv"
        LOG_FILE="./logs/teks_filter_${SAFE_MODEL}_${dataset}_gpu${GPU}_port${PORT}.log"

        echo "----------------------------------------------"
        echo "Model:   ${model}"
        echo "Dataset: ${dataset}"
        echo "GPU:     ${GPU}"
        echo "Port:    ${PORT}"
        echo "Log:     ${LOG_FILE}"
        echo "----------------------------------------------"

        if [[ ! -f "${INPUT_DATA_PATH}" ]]; then
            echo "ERROR: Dataset not found: ${INPUT_DATA_PATH}"
            run "Missing dataset: ${INPUT_DATA_PATH}" false
        fi

        {
            echo ""
            echo "=============================================="
            echo "START: $(date)"
            echo "Model: ${model}"
            echo "Dataset: ${dataset}"
            echo "GPU: ${GPU}"
            echo "Port: ${PORT}"
            echo "=============================================="
        } >> "${LOG_FILE}"

        # Verify that the expected model is served on this port.
        run "Verify server | ${model} | Port ${PORT}" \
            python -c '
import json, sys, urllib.request
port, expected = sys.argv[1:]
url = f"http://127.0.0.1:{port}/v1/models"
with urllib.request.urlopen(url, timeout=10) as response:
    models = json.load(response)["data"]
available = [m["id"] for m in models]
if expected not in available:
    raise RuntimeError(f"Expected {expected}, found {available} on port {port}")
print(f"Verified: {expected} on port {port}")
' "${PORT}" "${model}" >> "${LOG_FILE}" 2>&1

        # Run TEKS filter and write to model-specific log.
        run "TEKS Filter | ${model} | ${dataset} | GPU ${GPU} | Port ${PORT}" \
            python -u ./src/experiments/teks_filter.py \
                --input_data_path "${INPUT_DATA_PATH}" \
                --backend "${BACKEND}" \
                --model "${model}" \
                --port "${PORT}" \
                --n_rows "${N_ROWS}" \
                --save_every "${SAVE_EVERY}" \
                --temperature "${TEMPERATURE}" \
                --max_tokens "${MAX_TOKENS}" \
                --top_p "${TOP_P}" \
            >> "${LOG_FILE}" 2>&1

        echo "FINISHED: ${model} | ${dataset}"

        {
            echo "=============================================="
            echo "FINISHED: $(date)"
            echo "=============================================="
        } >> "${LOG_FILE}"

        sleep 2
    done
done

# =============================================================================
# SUCCESS NOTIFICATION
# =============================================================================

echo "=============================================="
echo "TEKS FILTER EXPERIMENTS COMPLETE"
echo "=============================================="

LOG_FILE="./logs/teks_filter_launcher.log"

./bin/_notify_slack.sh "SUCCESS" \
"✅ TEKS Filter experiments completed successfully
Script: $(basename "$0")
Host: $(hostname)
Backend: ${BACKEND}
Models: ${MODELS[*]}
Datasets: ${DATASETS[*]}
Logs: ./logs/teks_filter_*"