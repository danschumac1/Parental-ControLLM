#!/bin/bash
# chmod +x ./bin/curate_data.sh
# ./bin/curate_data.sh
# nohup ./bin/curate_data.sh > ./logs/curate_data.log 2>&1 &
# tail -f ./logs/curate_data.log

# -------------------------
# BACKEND CONFIG
# -------------------------

BACKEND="vllm"

MODELS=(
    # "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B"
)

GPU=0

# BACKEND="openai"
# MODELS=(
#     "gpt-4o-mini"
#     # "gpt-5-nano"
# )

# -------------------------
# GENERATION HYPERPARAMS
# -------------------------

SAMPLE_SIZE=30
TEMPERATURE=0
MAX_TOKENS=128
TOP_P=1.0

TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.90

# -------------------------
# PATHS
# -------------------------

STANDARDS_PATH="./data/cleaned/hecat_standards.tsv"
QUESTION_PROMPT_PATH="./data/prompts/dataset_curration/generate_question.yaml"
ANSWER_PROMPT_PATH="./data/prompts/dataset_curration/generate_answer.yaml"
OUTPUT_DIR="./data/generated"

# -------------------------
# SETUP LOGGING & OUTPUTS
# -------------------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p ./logs
mkdir -p "$OUTPUT_DIR"

echo "--- Starting Dataset Curation Batch: $TIMESTAMP ---"
echo "Backend: $BACKEND"
echo "Sample size: $SAMPLE_SIZE"
echo "Temperature: $TEMPERATURE"
echo "Max tokens: $MAX_TOKENS"
echo "Output dir: $OUTPUT_DIR"
echo "-----------------------------------"

# -------------------------
# RUN BATCH
# -------------------------

for model in "${MODELS[@]}"; do

    echo "Processing model: $model"
    echo "Backend: $BACKEND"
    echo "-----------------------------------"

    if [ "$BACKEND" = "vllm" ]; then

        echo "Running local vLLM offline inference on GPU=$GPU"

        VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=$GPU python ./src/create_dataset.py \
            --backend "$BACKEND" \
            --model "$model" \
            --standards_path "$STANDARDS_PATH" \
            --question_prompt_path "$QUESTION_PROMPT_PATH" \
            --answer_prompt_path "$ANSWER_PROMPT_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --sample_size "$SAMPLE_SIZE" \
            --temperature "$TEMPERATURE" \
            --max_tokens "$MAX_TOKENS" \
            --top_p "$TOP_P" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"

    elif [ "$BACKEND" = "openai" ]; then

        echo "Running OpenAI API inference"

        python ./src/create_dataset.py \
            --backend "$BACKEND" \
            --model "$model" \
            --standards_path "$STANDARDS_PATH" \
            --question_prompt_path "$QUESTION_PROMPT_PATH" \
            --answer_prompt_path "$ANSWER_PROMPT_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --sample_size "$SAMPLE_SIZE" \
            --temperature "$TEMPERATURE" \
            --max_tokens "$MAX_TOKENS"

    else
        echo "Invalid BACKEND: $BACKEND"
        echo "Expected BACKEND to be either 'vllm' or 'openai'."
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "Error encountered for model: $model"
        echo "Moving to next model..."
    else
        echo "Finished model: $model"
    fi

    echo "-----------------------------------"

done

echo "--- Dataset Curation Batch Complete: $(date) ---"