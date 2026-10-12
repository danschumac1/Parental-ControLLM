#!/bin/bash
# chmod +x ./bin/cat_python_scripts.sh
# ./bin/cat_python_scripts.sh
# Get project root (one directory above this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="$ROOT_DIR/all_python_files.txt"

find "$ROOT_DIR" \
    \( -type d \( -name ".git" -o -name ".venv" -o -name "venv" -o -name "__pycache__" \) -prune \) -o \
    \( -type f -name "*.py" -print0 \) |
sort -z |
while IFS= read -r -d '' file; do
    echo "========== ${file#"$ROOT_DIR"/} =========="
    cat "$file"
    printf '\n\n'
done > "$OUTPUT_FILE"

echo "Python scripts exported to: $OUTPUT_FILE"