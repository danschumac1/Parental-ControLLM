
#!/bin/bash
# chmod +x ./bin/experiments/llama_guard.sh
# ./bin/experiments/llama_guard.sh
# nohup ./bin/experiments/llama_guard.sh > ./logs/llama_guard.log 2>&1 &
# tail -f ./logs/llama_guard.log

export CUDA_VISIBLE_DEVICES=0

MODEL="meta-llama/Llama-Guard-4-12B"
INPUT_DATA_PATH="./data/generated/sample.tsv"
SCRIPT_PATH="./src/experiments/llama_guard_filter.py"

QUERY_COLUMN="situation"
N_ROWS=-1
SAVE_EVERY=1
MAX_TOKENS=32

mkdir -p ./logs

echo "Starting Llama Guard 4 Transformers Experiment"

python "$SCRIPT_PATH" \
    --input_data_path "$INPUT_DATA_PATH" \
    --backend transformers \
    --model "$MODEL" \
    --query_column "$QUERY_COLUMN" \
    --n_rows "$N_ROWS" \
    --save_every "$SAVE_EVERY" \
    --max_tokens "$MAX_TOKENS"

if [ $? -ne 0 ]; then
    echo "ERROR: Llama Guard experiment failed."
    exit 1
fi

echo "Llama Guard experiment complete."
