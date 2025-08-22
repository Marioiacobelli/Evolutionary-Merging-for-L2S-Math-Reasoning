#!/bin/bash
set -ex

# prompt template
PROMPT_TYPE="deepseek-math" 
# max sequence length
MAX_TOKENS=10240          
# zero-shots or few-shots     
NUM_SHOTS=0
# benchmarks to evaluate 
DATASETS="aime24,gsm8k,math,minerva_math,olympiadbench,college_math" 
# path to the model to evaluate
MODEL_PATH="/home/marioiac/Documents/DLAI-project/mergenetic/models/DeepSeek-R1-Distill-Qwen-1.5B" 
# where to store jsonl outputs with results
OUTPUT_DIR="/DEEPSEEK-R1/1.5Bnew"   
# seed for reproducibility
SEED=0

# call l2s_eval.sh 
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_VISIBLE_DEVICES="0" bash "$SCRIPT_DIR/l2s_eval.sh" \
  "$PROMPT_TYPE" "$MODEL_PATH" "$MAX_TOKENS" "$NUM_SHOTS" "$DATASETS" "$OUTPUT_DIR" "$SEED"