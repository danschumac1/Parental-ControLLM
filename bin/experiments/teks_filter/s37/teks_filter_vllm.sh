
#!/usr/bin/env bash

# =============================================================================
# TEKS FILTER EXPERIMENTS
# =============================================================================
# Run:
#   chmod +x ./bin/experiments/teks_filter_vllm.sh
#   nohup bash ./bin/experiments/teks_filter/s37/teks_filter_vllm.sh > ./logs/teks_filter_launcher.log 2>&1 &
#
# Monitor:
#   tail -f ./logs/teks_filter_launcher.log
#
# Input:
#   data/splits/test.tsv
#
# Output:
#   data/generated/teks_filter_results/<model>/<query_type>/outputs.jsonl
#
# Individual model logs are generated automatically.
# Existing outputs are reused to resume incomplete experiments.
# =============================================================================

LOG_FILE="./logs/teks_filter_launcher.log"
source ./bin/_slack_traps.sh

# =============================================================================
# VLLM SERVER COMMANDS (REFERENCE ONLY)
# =============================================================================
#region
: <<'SERVER_NOTES'

# GEMMA 4 12B
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve google/gemma-4-12B-it \
        --host 127.0.0.1 \
        --port 8002 \
        --gpu-memory-utilization 0.90 \
    > ./logs/gemma4_12b_server.log 2>&1 &

tail -f ./logs/gemma4_12b_server.log


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
        --port 8005 \
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
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
    > ./logs/qwen35_4b_server.log 2>&1 &

tail -f ./logs/qwen35_4b_server.log


# QWEN 3.5 9B
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve Qwen/Qwen3.5-9B \
        --host 127.0.0.1 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
    > ./logs/qwen35_9b_server.log 2>&1 &

tail -f ./logs/qwen35_9b_server.log


# MINISTRAL 3 8B
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve mistralai/Ministral-3-8B-Instruct-2512 \
        --host 127.0.0.1 \
        --port 8000 \
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
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 65536 \
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
    "google/gemma-4-12B-it"
    # "google/gemma-4-31B-it"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "Qwen/Qwen3.5-4B"
    # "Qwen/Qwen3.5-9B"
    # "mistralai/Ministral-3-8B-Instruct-2512"
    # "mistralai/Ministral-3-14B-Instruct-2512"
    # "microsoft/phi-4"
)

# =============================================================================
# SERVER PORT ASSIGNMENTS
# =============================================================================
# These are server ports, not necessarily physical GPU numbers.
# This script does NOT launch or move vLLM servers.
# =============================================================================

declare -A MODEL_PORTS=(
    ["google/gemma-4-12B-it"]=8002
    ["google/gemma-4-31B-it"]=8002
    ["meta-llama/Llama-3.3-70B-Instruct"]=8005
    ["Qwen/Qwen3.5-4B"]=8000
    ["Qwen/Qwen3.5-9B"]=8000
    ["mistralai/Ministral-3-8B-Instruct-2512"]=8000
    ["mistralai/Ministral-3-14B-Instruct-2512"]=8000
    ["microsoft/phi-4"]=8000
)

# =============================================================================
# QUERY TYPES
# =============================================================================
# Both columns are loaded from the same test.tsv file.
# =============================================================================

QUERY_TYPES=(
    "situation"
    # "eduQ"
)

INPUT_DATA_PATH="./data/splits/test.tsv"

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
echo "Host:        $(hostname)"
echo "Backend:     ${BACKEND}"
echo "Models:      ${MODELS[*]}"
echo "Query types: ${QUERY_TYPES[*]}"
echo "Input:       ${INPUT_DATA_PATH}"
echo "=============================================="

if (( ${#MODELS[@]} == 0 || ${#QUERY_TYPES[@]} == 0 )); then
    echo "ERROR: No models or query types selected."
    run "Validate experiment selection" false
fi

if [[ ! -f "${INPUT_DATA_PATH}" ]]; then
    echo "ERROR: Test dataset not found: ${INPUT_DATA_PATH}"
    run "Missing dataset: ${INPUT_DATA_PATH}" false
fi

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

for model in "${MODELS[@]}"; do

    if [[ ! -v MODEL_PORTS["$model"] ]]; then
        echo "ERROR: No port assigned to ${model}"
        run "Missing port assignment: ${model}" false
    fi

    PORT="${MODEL_PORTS[$model]}"
    SAFE_MODEL="${model//\//--}"

    for query_type in "${QUERY_TYPES[@]}"; do

        LOG_FILE="./logs/teks_filter_${SAFE_MODEL}_${query_type}_port${PORT}.log"

        echo "----------------------------------------------"
        echo "Model:      ${model}"
        echo "Query type: ${query_type}"
        echo "Port:       ${PORT}"
        echo "Input:      ${INPUT_DATA_PATH}"
        echo "Log:        ${LOG_FILE}"
        echo "----------------------------------------------"

        {
            echo ""
            echo "=============================================="
            echo "START: $(date)"
            echo "Model: ${model}"
            echo "Query type: ${query_type}"
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
    raise RuntimeError(
        f"Expected {expected}, found {available} on port {port}"
    )

print(f"Verified: {expected} on port {port}")
' "${PORT}" "${model}" >> "${LOG_FILE}" 2>&1

        # Run TEKS filter and write to model-specific log.
        run "TEKS Filter | ${model} | ${query_type} | Port ${PORT}" \
            python -u ./src/experiments/teks_filter.py \
                --input_data_path "${INPUT_DATA_PATH}" \
                --query_row "${query_type}" \
                --backend "${BACKEND}" \
                --model "${model}" \
                --port "${PORT}" \
                --n_rows "${N_ROWS}" \
                --save_every "${SAVE_EVERY}" \
                --temperature "${TEMPERATURE}" \
                --max_tokens "${MAX_TOKENS}" \
                --top_p "${TOP_P}" \
            >> "${LOG_FILE}" 2>&1

        echo "FINISHED: ${model} | ${query_type}"

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
Query types: ${QUERY_TYPES[*]}
Input: ${INPUT_DATA_PATH}
Logs: ./logs/teks_filter_*"
