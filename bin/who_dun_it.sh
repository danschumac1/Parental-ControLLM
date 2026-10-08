#!/bin/bash
# chmod +x ./bin/who_dun_it.sh
# ./bin/who_dun_it.sh

# Step 1: Build a clean GPU_UUID → GPU_INDEX mapping
declare -A uuid_to_index
while IFS=',' read -r index uuid; do
  index=$(echo "$index" | xargs)
  uuid=$(echo "$uuid" | xargs)
  uuid_to_index["$uuid"]="GPU $index"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)

# Step 2: Look up processes and map GPU UUID to index
nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | while IFS=',' read -r uuid pid; do
  uuid=$(echo "$uuid" | xargs)
  pid=$(echo "$pid" | xargs)
  gpu=${uuid_to_index["$uuid"]}
  user=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
  cmd=$(ps -o args= -p "$pid" 2>/dev/null | xargs)
  if [[ -n "$user" && -n "$cmd" ]]; then
    echo "${gpu:-GPU UNKNOWN} | USER: $user | PID: $pid | CMD: $cmd"
  fi
done