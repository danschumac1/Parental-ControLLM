#!/usr/bin/env bash
# 2754623
# =============================================================================
# TEKS FILTER RAG EXPERIMENTS — vLLM
# =============================================================================
# Run:
#   mkdir -p ./logs
#   chmod +x ./bin/experiments/teks_filter_rag/s117/teks_filter_rag_vllm.sh
#   nohup bash ./bin/experiments/teks_filter_rag/s117/teks_filter_rag_vllm.sh > ./logs/teks_filter_rag_launcher.log 2>&1 &
#
# Monitor:
#   tail -f ./logs/teks_filter_rag_launcher.log
#
# Input:  data/splits/test.tsv
# Output: data/generated/teks_filter_rag_results/<model>/<query>/<corpus>/...
# Existing outputs are reused by the Python script to resume runs.
# Servers must already be running; this script does not launch them.
# =============================================================================
#region
set -o pipefail
mkdir -p ./logs
LOG_FILE="./logs/teks_filter_rag_launcher.log"
source ./bin/_slack_traps.sh

# =============================================================================
# vLLM SERVER COMMANDS (REFERENCE ONLY)
# =============================================================================
#region
: <<'SERVER_NOTES'
# GEMMA 4 12B
nohup env CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve google/gemma-4-12B-it \
        --host 127.0.0.1 --port 8002 --gpu-memory-utilization 0.90 \
    > ./logs/gemma4_12b_server.log 2>&1 &

# GEMMA 4 31B
nohup env CUDA_VISIBLE_DEVICES=2 VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve google/gemma-4-31B-it \
        --host 127.0.0.1 --port 8002 --gpu-memory-utilization 0.90 \
        --max-model-len 65536 \
    > ./logs/gemma4_31b_server.log 2>&1 &

# LLAMA 3.3 70B
nohup env CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve meta-llama/Llama-3.3-70B-Instruct \
        --host 127.0.0.1 --port 8005 --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.90 --max-model-len 65536 \
    > ./logs/llama3_70b_server.log 2>&1 &

# QWEN 3.5 4B / 9B (choose one server on port 8000)
nohup env CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve Qwen/Qwen3.5-9B \
        --host 127.0.0.1 --port 8000 \
        --gpu-memory-utilization 0.90 \
    > ./logs/qwen35_9b_server.log 2>&1 &

# MINISTRAL 3 8B / 14B (choose one server on port 8000)
nohup env CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve mistralai/Ministral-3-8B-Instruct-2512 \
        --host 127.0.0.1 --port 8000 \
        --gpu-memory-utilization 0.90 --max-model-len 32768 \
        --tokenizer-mode mistral --config-format mistral --load-format mistral \
    > ./logs/ministral3_8b_server.log 2>&1 &

SERVER_NOTES
#endregion

# =============================================================================
# BACKEND, MODELS & PORTS
# =============================================================================
BACKEND="vllm"

MODELS=(
    # "google/gemma-4-12B-it"
    # "google/gemma-4-31B-it"
    # "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen3.5-4B"
    # "Qwen/Qwen3.5-9B"
    # "mistralai/Ministral-3-8B-Instruct-2512"
    # "mistralai/Ministral-3-14B-Instruct-2512"
    # "microsoft/phi-4"
)

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
# EXPERIMENT CONFIG
# =============================================================================
INPUT_DATA_PATH="./data/splits/test.tsv"

QUERY_TYPES=(
    "situation"
    # "eduq"
)

RETRIEVAL_CORPORA=(
    # "split"
    "full"
)

TOP_K_VALUES=(
    # 5
    # 10
    20
    # 50
)

EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
N_ROWS=-1
SAVE_EVERY=1
TEMPERATURE=0.0
MAX_TOKENS=1024
TOP_P=1.0

# =============================================================================
# SETUP
# =============================================================================
echo "=============================================="
echo "STARTING vLLM TEKS FILTER RAG EXPERIMENTS"
echo "=============================================="
echo "Host:        $(hostname)"
echo "Models:      ${MODELS[*]}"
echo "Query types: ${QUERY_TYPES[*]}"
echo "Corpora:     ${RETRIEVAL_CORPORA[*]}"
echo "Top-K:       ${TOP_K_VALUES[*]}"
echo "Embedding:   ${EMBEDDING_MODEL}"
echo "Input:       ${INPUT_DATA_PATH}"
echo "=============================================="

if (( ${#MODELS[@]} == 0 || ${#QUERY_TYPES[@]} == 0 || ${#RETRIEVAL_CORPORA[@]} == 0 || ${#TOP_K_VALUES[@]} == 0 )); then
    echo "ERROR: At least one experiment selection is empty."
    run "Validate RAG experiment selection" false
fi

if [[ ! -f "${INPUT_DATA_PATH}" ]]; then
    echo "ERROR: Dataset not found: ${INPUT_DATA_PATH}"
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

    # Verify the server once per model, before its experiments.
    SERVER_LOG="./logs/teks_filter_rag_${SAFE_MODEL}_server_check.log"
    run "Verify server | ${model} | Port ${PORT}" \
        python -c '
import json, sys, urllib.request
port, expected = sys.argv[1:]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as response:
    available = [m["id"] for m in json.load(response)["data"]]
if expected not in available:
    raise RuntimeError(f"Expected {expected}, found {available} on port {port}")
print(f"Verified: {expected} on port {port}")
' "${PORT}" "${model}" > "${SERVER_LOG}" 2>&1

    for query_type in "${QUERY_TYPES[@]}"; do
        for corpus in "${RETRIEVAL_CORPORA[@]}"; do
            for top_k in "${TOP_K_VALUES[@]}"; do
                MODEL_LOG="./logs/teks_filter_rag_${SAFE_MODEL}_${query_type}_${corpus}_top${top_k}_port${PORT}.log"

                echo "----------------------------------------------"
                echo "START: $(date)"
                echo "Model:  ${model} | Port: ${PORT}"
                echo "Query:  ${query_type} | Corpus: ${corpus} | Top-K: ${top_k}"
                echo "Log:    ${MODEL_LOG}"
                echo "----------------------------------------------"

                run "TEKS RAG | ${model} | ${query_type} | ${corpus} | K=${top_k}" \
                    python -u ./src/experiments/teks_filter_rag.py \
                        --input_data_path "${INPUT_DATA_PATH}" \
                        --query_row "${query_type}" \
                        --retrieval_corpus "${corpus}" \
                        --backend "${BACKEND}" \
                        --model "${model}" \
                        --port "${PORT}" \
                        --n_r_contexts "${top_k}" \
                        --embedding_model "${EMBEDDING_MODEL}" \
                        --n_rows "${N_ROWS}" \
                        --save_every "${SAVE_EVERY}" \
                        --temperature "${TEMPERATURE}" \
                        --max_tokens "${MAX_TOKENS}" \
                        --top_p "${TOP_P}" \
                    > "${MODEL_LOG}" 2>&1

                echo "FINISHED: $(date) | ${model} | ${query_type} | ${corpus} | K=${top_k}"
                sleep 2
            done
        done
    done
done

# =============================================================================
# SUCCESS NOTIFICATION
# =============================================================================
echo "=============================================="
echo "vLLM TEKS FILTER RAG EXPERIMENTS COMPLETE"
echo "=============================================="
LOG_FILE="./logs/teks_filter_rag_launcher.log"

./bin/_notify_slack.sh "SUCCESS" \
"✅ vLLM TEKS Filter RAG experiments completed successfully
Script: $(basename "$0")
Host: $(hostname)
Models: ${MODELS[*]}
Query types: ${QUERY_TYPES[*]}
Retrieval corpora: ${RETRIEVAL_CORPORA[*]}
Top-K: ${TOP_K_VALUES[*]}
Input: ${INPUT_DATA_PATH}
Logs: ./logs/teks_filter_rag_*"
