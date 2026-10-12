#!/bin/bash
# chmod +x ./bin/data_generation/generate_queries.sh
# ./bin/data_generation/generate_queries.sh
# nohup ./bin/data_generation/generate_queries.sh > ./logs/generate_queries.log 2>&1 &
# tail -f ./logs/generate_queries.log

# -------------------------
# BACKEND CONFIG
# -------------------------

# BACKEND="vllm"
# MODELS=(
#     "google/gemma-4-12B-it"
#     # "Qwen/Qwen2.5-7B-Instruct"
# )

BACKEND="openai"
MODELS=(
    "gpt-5.4-nano"
)

VLLM_BASE_URL="http://localhost:8002/v1"

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
# SETUP LOGGING
# -------------------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p ./logs

echo "--- Starting Query Generation: $TIMESTAMP ---"
echo "Backend: $BACKEND"
echo "Models: ${MODELS[*]}"
echo "Generation types: ${GENERATION_TYPES[*]}"
echo "Prompt type: $PROMPT_TYPE"
echo "Temperature: $TEMPERATURE"
echo "Max tokens: $MAX_TOKENS"
echo "-----------------------------------"

# -------------------------
# RUN BATCH
# -------------------------

for model in "${MODELS[@]}"; do
    for generation_type in "${GENERATION_TYPES[@]}"; do

        echo "Model: $model"
        echo "Generation type: $generation_type"
        echo "-----------------------------------"

        CMD=(
            python "$SCRIPT_PATH"
            --input_path "$INPUT_PATH"
            --backend "$BACKEND"
            --model "$model"
            --generation_type "$generation_type"
            --prompt_type "$PROMPT_TYPE"
            --temperature "$TEMPERATURE"
            --max_tokens "$MAX_TOKENS"
            --top_p "$TOP_P"
        )

        if [ "$LIMIT" -ge 0 ]; then
            CMD+=(--limit "$LIMIT")
        fi

        if [ "$BACKEND" = "vllm" ]; then
            CMD+=(--vllm_base_url "$VLLM_BASE_URL")
        elif [ "$BACKEND" != "openai" ]; then
            echo "Invalid backend: $BACKEND"
            exit 1
        fi

        "${CMD[@]}"

        if [ $? -ne 0 ]; then
            echo "ERROR: $model ($generation_type)"
        else
            echo "Finished: $model ($generation_type)"
        fi

        echo "-----------------------------------"
    done
done

echo "--- Query Generation Complete: $(date) ---"