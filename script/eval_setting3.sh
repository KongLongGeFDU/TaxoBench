#!/bin/bash

# ==========================================
# 1. Basic Configuration
# ==========================================
PYTHON_EXEC="/opt/miniconda3/envs/eval/bin/python"
PYTHON_SCRIPT="/mnt/TaxoBench/setting_pipeline/eval_setting3.py"
INPUT_PATH="/mnt/TaxoBench/dataset/data.jsonl"
TEMP=0
NUM_WORKERS=32

# ==========================================
# 2. Model Group Definition ("Provider Model")
# Format: "ProviderName ModelName"
# Each line represents a group. Comment/uncomment to select.
# ==========================================
MODEL_PAIRS=(
    "gpt gpt-5"
    "gemini gemini-3-pro-preview"
    "claude claude-sonnet-4-5-20250929"
    "qwen Qwen3-Max-Preview"
    "deepseek DeepSeek-V3.2"
    "kimi Kimi-K2-0905"
    "kimi Kimi-K2-thinking"
)

# ==========================================
# 3. Thinking Mode Configuration
# Empty string for disabled, --thinking for enabled
# ==========================================
THINKING_OPTIONS=(
    "" 
    "--thinking"
)

# ==========================================
# 4. Main Loop Logic
# ==========================================

# Outer loop: Iterate through model pairs
for pair in "${MODEL_PAIRS[@]}"; do
    # Split string into Provider and Model variables
    read -r PROVIDER MODEL <<< "$pair"

    # Inner loop: Iterate through Thinking modes
    for THINKING_FLAG in "${THINKING_OPTIONS[@]}"; do

        # ==========================================
        # Special filtering logic for Kimi models
        # ==========================================
        
        # Rule 1: Skip if Kimi-K2-0905 is set to thinking mode
        if [[ "$MODEL" == "Kimi-K2-0905" && "$THINKING_FLAG" == "--thinking" ]]; then
            continue
        fi

        # Rule 2: Skip if Kimi-K2-thinking is set to non-thinking mode
        if [[ "$MODEL" == "Kimi-K2-thinking" && "$THINKING_FLAG" == "" ]]; then
            continue
        fi

        # ==========================================
        
        # --- Construct file paths ---
        if [[ "$THINKING_FLAG" == "--thinking" ]];  then
            MODEL_MODE="${MODEL}-Thinking"
        else
            MODEL_MODE="${MODEL}-No-Thinking"
        fi

        # Define output paths (Assuming fixed directory structure)
        BASE_DIR="/mnt/TaxoBench/results/setting3"
        EVAL_OUTPUT="${BASE_DIR}/${MODEL_MODE}/${MODEL}.jsonl"
        LOG_DIR="${BASE_DIR}/log"
        LOG_FILE="${LOG_DIR}/${MODEL_MODE}.log"

        # Create directories
        mkdir -p "$LOG_DIR"
        EVAL_PATH_DIR=$(dirname "$EVAL_OUTPUT")
        mkdir -p "$EVAL_PATH_DIR"

        # --- Print task info to terminal ---
        echo "----------------------------------------"
        echo "🚀 Starting Task:"
        echo "   Provider: $PROVIDER"
        echo "   Model:    $MODEL"
        echo "   Mode:     $MODEL_MODE"
        echo "   Log:      $LOG_FILE"
        echo "----------------------------------------"

        # --- Log header information ---
        {
            echo "=============== Parameter List ===============" 
            echo "Time:     $(date)"
            echo "Provider: $PROVIDER"
            echo "Model:    $MODEL"
            echo "Thinking: $THINKING_FLAG"
            echo "TEMP:     $TEMP"
            echo "Num_workers: $NUM_WORKERS"
            echo "==============================================" 
        } >> "$LOG_FILE"

        # --- Execute Python Script ---
        # Redirect stderr to the log file as well (2>&1)
        "$PYTHON_EXEC" "$PYTHON_SCRIPT" \
            --path "$INPUT_PATH" \
            --eval_path "$EVAL_OUTPUT" \
            --num_processes "$NUM_WORKERS" \
            --provider "$PROVIDER" \
            --model "$MODEL" \
            --temperature "$TEMP" \
            $THINKING_FLAG \
            >> "$LOG_FILE" 2>&1
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "✅ Task Finished: $MODEL_MODE"
        else
            echo "❌ Task Failed: $MODEL_MODE (Check log: $LOG_FILE)"
        fi

    done
done

echo "🎉 All tasks completed."