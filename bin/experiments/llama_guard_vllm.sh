
#!/bin/bash
# chmod +x ./bin/experiments/llama_guard_vllm.sh
# ./bin/experiments/llama_guard_vllm.sh
# nohup ./bin/experiments/llama_guard_vllm.sh > ./logs/llama_guard_vllm.log 2>&1 &
# tail -f ./logs/llama_guard_vllm.log

# -------------------------
# VLLM SERVER SETUP
# -------------------------

# nohup env \
#     CUDA_VISIBLE_DEVICES=0,1 \
#     VLLM_USE_FLASHINFER_SAMPLER=0 \
#     vllm serve meta-llama/Llama-Guard-4-12B \
#         --host 127.0.0.1 \
#         --port 8002 \
#         --tensor-parallel-size 2 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 65536 \
#     > ./logs/llama_guard_server.log 2>&1 &
#
# tail -f ./logs/llama_guard_server.log

# -------------------------
# BACKEND & MODELS
# -------------------------

BACKEND="vllm"
VLLM_BASE_URL="http://localhost:8002/v1"

MODELS=(
    "meta-llama/Llama-Guard-4-12B"
)

# -------------------------
# EXPERIMENT CONFIG
# -------------------------

INPUT_DATA_PATH="./data/generated/sample.tsv"
PROMPT_PATH="./data/prompts/llama_guard/hecat_expectations.yaml"
SCRIPT_PATH="./src/experiments/llama_guard_filter.py"

QUERY_COLUMN="situation"
# QUERY_COLUMN="eduQ"

N_ROWS=3
SAVE_EVERY=1

TEMPERATURE=0.0
MAX_TOKENS=64
TOP_P=1.0

# -------------------------
# LOGGING
# -------------------------

mkdir -p ./logs

echo "Starting Llama Guard 4 Experiments"
echo "Backend: ${BACKEND}"
echo "Input: ${INPUT_DATA_PATH}"
echo "Query column: ${QUERY_COLUMN}"
echo "Prompt: ${PROMPT_PATH}"
echo "-----------------------------------"

# -------------------------
# RUN EXPERIMENTS
# -------------------------

for model in "${MODELS[@]}"; do

    echo "----------------------------------------------------------------"
    echo "STARTING: ${model}"
    echo "Backend: ${BACKEND}"
    echo "----------------------------------------------------------------"

    python "$SCRIPT_PATH" \
        --input_data_path "$INPUT_DATA_PATH" \
        --backend "$BACKEND" \
        --model "$model" \
        --vllm_base_url "$VLLM_BASE_URL" \
        --prompt_path "$PROMPT_PATH" \
        --query_column "$QUERY_COLUMN" \
        --n_rows "$N_ROWS" \
        --save_every "$SAVE_EVERY" \
        --temperature "$TEMPERATURE" \
        --max_tokens "$MAX_TOKENS" \
        --top_p "$TOP_P"

    if [ $? -ne 0 ]; then
        echo "ERROR: ${model}"
    else
        echo "FINISHED: ${model}"
    fi

    echo "----------------------------------------------------------------"
    sleep 2
done

echo "Llama Guard 4 Experiments Complete"
