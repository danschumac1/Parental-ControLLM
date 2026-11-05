
#!/usr/bin/env bash
# 1824300
# =============================================================================
# TEKS FILTER RAG EXPERIMENTS — OPENAI
# =============================================================================
# Run:
#   chmod +x ./bin/experiments/teks_filter_rag/teks_filter_rag_openai.sh
#   nohup ./bin/experiments/teks_filter_rag/teks_filter_rag_openai.sh > ./logs/teks_filter_rag_openai.log 2>&1 &
#
# Monitor:
#   tail -f ./logs/teks_filter_rag_openai.log
#
# Requires:
#   - OPENAI_API_KEY in environment
#   - sentence-transformers installed
# =============================================================================

set -o pipefail

mkdir -p ./logs

LOG_FILE="./logs/teks_filter_rag_openai.log"
source ./bin/_slack_traps.sh

# =============================================================================
# CONFIG
# =============================================================================

BACKEND="openai"
INPUT_DATA_PATH="./data/splits/test.tsv"

MODELS=(
    "gpt-5.4-nano"
    # "gpt-5.4-mini"
)

QUERY_ROWS=(
    "situation"
    "eduq"
)

RETRIEVAL_CORPORA=(
    # "split"
    "full"
)

TOP_K_VALUES=(
    20
    # 5
    # 10
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

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

if [[ ! -f "${INPUT_DATA_PATH}" ]]; then
    echo "ERROR: Input file not found: ${INPUT_DATA_PATH}"
    exit 1
fi

echo "=============================================="
echo "STARTING OPENAI TEKS FILTER RAG EXPERIMENTS"
echo "Models:            ${MODELS[*]}"
echo "Query rows:        ${QUERY_ROWS[*]}"
echo "Retrieval corpora: ${RETRIEVAL_CORPORA[*]}"
echo "Top-K values:      ${TOP_K_VALUES[*]}"
echo "Embedding model:   ${EMBEDDING_MODEL}"
echo "Input:             ${INPUT_DATA_PATH}"
echo "=============================================="

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

for model in "${MODELS[@]}"; do
    for query_row in "${QUERY_ROWS[@]}"; do
        for corpus in "${RETRIEVAL_CORPORA[@]}"; do
            for top_k in "${TOP_K_VALUES[@]}"; do

                SAFE_MODEL="${model//\//--}"

                MODEL_LOG="./logs/teks_filter_rag_${SAFE_MODEL}_${query_row}_${corpus}_top${top_k}.log"

                echo "----------------------------------------------"
                echo "Model:   ${model}"
                echo "Query:   ${query_row}"
                echo "Corpus:  ${corpus}"
                echo "Top-K:   ${top_k}"
                echo "Log:     ${MODEL_LOG}"
                echo "----------------------------------------------"

                run "TEKS RAG | ${model} | ${query_row} | ${corpus} | K=${top_k}" \
                    python -u ./src/experiments/teks_filter_rag.py \
                        --input_data_path "${INPUT_DATA_PATH}" \
                        --query_row "${query_row}" \
                        --retrieval_corpus "${corpus}" \
                        --backend "${BACKEND}" \
                        --model "${model}" \
                        --n_r_contexts "${top_k}" \
                        --embedding_model "${EMBEDDING_MODEL}" \
                        --n_rows "${N_ROWS}" \
                        --save_every "${SAVE_EVERY}" \
                        --temperature "${TEMPERATURE}" \
                        --max_tokens "${MAX_TOKENS}" \
                        --top_p "${TOP_P}" \
                    > "${MODEL_LOG}" 2>&1

                echo "FINISHED: ${model} | ${query_row} | ${corpus} | K=${top_k}"

            done
        done
    done
done

# =============================================================================
# SUCCESS NOTIFICATION
# =============================================================================

echo "=============================================="
echo "OPENAI TEKS FILTER RAG EXPERIMENTS COMPLETE"
echo "=============================================="

./bin/_notify_slack.sh "SUCCESS" \
"✅ OpenAI TEKS Filter RAG experiments completed successfully
Script: $(basename "$0")
Host: $(hostname)
Models: ${MODELS[*]}
Query rows: ${QUERY_ROWS[*]}
Retrieval corpora: ${RETRIEVAL_CORPORA[*]}
Top-K: ${TOP_K_VALUES[*]}
Input: ${INPUT_DATA_PATH}
Logs: ./logs/teks_filter_rag_*"
