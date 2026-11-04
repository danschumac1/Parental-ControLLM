
#!/usr/bin/env bash

# =============================================================================
# TEKS FILTER EXPERIMENTS — OPENAI
# =============================================================================
# Run:
#   chmod +x ./bin/experiments/teks_filter_openai.sh
#   nohup ./bin/experiments/teks_filter_openai.sh > ./logs/teks_filter_openai.log 2>&1 &
#
# Monitor:
#   tail -f ./logs/teks_filter_openai.log
#
# Requires:
#   - OPENAI_API_KEY in environment
#   - OpenAI backend support in teks_filter.py
# =============================================================================

LOG_FILE="./logs/teks_filter_openai.log"
source ./bin/_slack_traps.sh

# =============================================================================
# CONFIG
# =============================================================================

BACKEND="openai"

MODELS=(
    "gpt-5.4-nano"
    # "gpt-5.4-mini"
)

DATASETS=(
    # "situations"
    "eduQ"
)

N_ROWS=-1
SAVE_EVERY=1

TEMPERATURE=0.0
MAX_TOKENS=1024
TOP_P=1.0

# =============================================================================
# SETUP
# =============================================================================

mkdir -p ./logs

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

echo "=============================================="
echo "STARTING OPENAI TEKS FILTER EXPERIMENTS"
echo "Models:   ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "=============================================="

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do

        INPUT_DATA_PATH="./data/generated/${dataset}.tsv"
        SAFE_MODEL="${model//\//--}"
        MODEL_LOG="./logs/teks_filter_${SAFE_MODEL}_${dataset}.log"

        echo "Running: ${model} | ${dataset}"
        echo "Log: ${MODEL_LOG}"

        run "TEKS Filter | ${model} | ${dataset}" \
            python -u ./src/experiments/teks_filter.py \
                --input_data_path "${INPUT_DATA_PATH}" \
                --backend "${BACKEND}" \
                --model "${model}" \
                --n_rows "${N_ROWS}" \
                --save_every "${SAVE_EVERY}" \
                --temperature "${TEMPERATURE}" \
                --max_tokens "${MAX_TOKENS}" \
                --top_p "${TOP_P}" \
            > "${MODEL_LOG}" 2>&1

        echo "FINISHED: ${model} | ${dataset}"

    done
done

# =============================================================================
# SUCCESS NOTIFICATION
# =============================================================================

echo "=============================================="
echo "OPENAI TEKS FILTER EXPERIMENTS COMPLETE"
echo "=============================================="

LOG_FILE="./logs/teks_filter_openai.log"

./bin/_notify_slack.sh "SUCCESS" \
"✅ OpenAI TEKS Filter experiments completed successfully
Script: $(basename "$0")
Host: $(hostname)
Models: ${MODELS[*]}
Datasets: ${DATASETS[*]}
Logs: ./logs/teks_filter_*"
