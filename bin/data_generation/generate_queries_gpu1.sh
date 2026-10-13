
#!/bin/bash
# 3237917
# chmod +x ./bin/data_generation/generate_queries_gpu1.sh
# ./bin/data_generation/generate_queries_gpu1.sh
# nohup ./bin/data_generation/generate_queries_gpu1.sh > ./logs/generate_queries_gpu1.log 2>&1 &
# tail -f ./logs/generate_queries_gpu1.log

set -euo pipefail
# 3230521
# nohup env \
#     CUDA_VISIBLE_DEVICES=1 \
#     VLLM_USE_FLASHINFER_SAMPLER=0 \
#     vllm serve microsoft/phi-4 \
#         --host 127.0.0.1 \
#         --port 8002 \
#         --tensor-parallel-size 1 \
#         --gpu-memory-utilization 0.90 \
#     > ./logs/phi_vllm_gpu0.log 2>&1 &


# -------------------------
# BACKEND CONFIG
# -------------------------

BACKEND="vllm"
MODELS=(
    # "google/gemma-4-12B-it"
    "microsoft/phi-4"
)

# BACKEND="openai"
# MODELS=(
#     "gpt-5.4-nano"
# )

VLLM_BASE_URL="http://localhost:8002/v1"
GPU_ID=1

# -------------------------
# GENERATION CONFIG
# -------------------------

GENERATION_TYPES=(
    "eduQ_generation"
    # "situation_generation"
)

PROMPT_TYPE="instruct"

TEMPERATURE=0.7
MAX_TOKENS=256
TOP_P=1.0
LIMIT=-1

# -------------------------
# PATHS
# -------------------------

INPUT_PATH="./data/cleaned/hecat_standards.tsv"
SCRIPT_PATH="./src/prompt_for_queries.py"

# -------------------------
# VALIDATION
# -------------------------

if [[ "$BACKEND" != "vllm" && "$BACKEND" != "openai" ]]; then
    echo "ERROR: Invalid backend: $BACKEND"
    exit 1
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: No models configured."
    exit 1
fi

if [[ ${#GENERATION_TYPES[@]} -eq 0 ]]; then
    echo "ERROR: No generation types configured."
    exit 1
fi

if [[ ! -f "$INPUT_PATH" ]]; then
    echo "ERROR: Input file not found: $INPUT_PATH"
    exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "ERROR: Python script not found: $SCRIPT_PATH"
    exit 1
fi

# -------------------------
# SETUP LOGGING
# -------------------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p ./logs

echo "==================================="
echo "Starting Query Generation"
echo "Timestamp: $TIMESTAMP"
echo "Backend: $BACKEND"
echo "Models: ${MODELS[*]}"
echo "Generation types: ${GENERATION_TYPES[*]}"
echo "Prompt type: $PROMPT_TYPE"
echo "Temperature: $TEMPERATURE"
echo "Max tokens: $MAX_TOKENS"
echo "Top-p: $TOP_P"
echo "Limit: $LIMIT"
echo "GPU: $GPU_ID"
echo "==================================="

# -------------------------
# RUN BATCH
# -------------------------

TOTAL_START=$SECONDS

for model in "${MODELS[@]}"; do
    for generation_type in "${GENERATION_TYPES[@]}"; do

        echo ""
        echo "-----------------------------------"
        echo "Model: $model"
        echo "Generation type: $generation_type"
        echo "Started: $(date)"
        echo "-----------------------------------"

        START=$SECONDS

        CMD=(
            env CUDA_VISIBLE_DEVICES="$GPU_ID"
            python -u "$SCRIPT_PATH"
            --input_path "$INPUT_PATH"
            --backend "$BACKEND"
            --model "$model"
            --generation_type "$generation_type"
            --prompt_type "$PROMPT_TYPE"
            --temperature "$TEMPERATURE"
            --max_tokens "$MAX_TOKENS"
            --top_p "$TOP_P"
        )

        if [[ "$LIMIT" -ge 0 ]]; then
            CMD+=(--limit "$LIMIT")
        fi

        if [[ "$BACKEND" == "vllm" ]]; then
            CMD+=(--vllm_base_url "$VLLM_BASE_URL")
        fi

        if "${CMD[@]}"; then
            ELAPSED=$((SECONDS - START))
            echo "Finished: $model ($generation_type)"
            echo "Elapsed: ${ELAPSED}s"
        else
            STATUS=$?
            echo "ERROR: $model ($generation_type)"
            echo "Exit code: $STATUS"
            exit "$STATUS"
        fi

        echo "-----------------------------------"
    done
done

# -------------------------
# COMPLETE
# -------------------------

TOTAL_ELAPSED=$((SECONDS - TOTAL_START))

echo ""
echo "==================================="
echo "Query Generation Complete"
echo "Finished: $(date)"
echo "Total elapsed: ${TOTAL_ELAPSED}s"
echo "==================================="
