#!/bin/bash

# ==========================================
# 1. Basic Configuration
# ==========================================
PYTHON_EXEC="/opt/miniconda3/envs/eval/bin/python"
PYTHON_SCRIPT="/mnt/TaxoBench/setting_pipeline/eval_setting2.py"
INPUT_PATH="/mnt/TaxoBench/dataset/data.jsonl"
TEMP=0
NUM_WORKERS=16

# ==========================================
# 2. Model Definitions
# Format: "Provider Model_Name"
# ==========================================
MODEL_PAIRS=(
    "gpt gpt-5"
    "gemini gemini-3-pro-preview"
    "claude claude-sonnet-4-5-20250929"
    "qwen Qwen3-Max-Preview"
    "deepseek DeepSeek-V3.2"
    "kimi Kimi-K2-0905"
)

# ==========================================
# 3. Thinking Mode Configuration
# Empty string disables; --thinking enables
# ==========================================
THINKING_OPTIONS=(
    "" 
    "--thinking"
)

# ==========================================
# 4. Main Execution Loop
# ==========================================

# Iterate through model pairs
for pair in "${MODEL_PAIRS[@]}"; do
    read -r PROVIDER MODEL <<< "$pair"

    # Iterate through thinking modes
    for THINKING_FLAG in "${THINKING_OPTIONS[@]}"; do
        
        # --- Construct File Paths ---
        if [[ "$THINKING_FLAG" == "--thinking" ]];  then
            MODEL_MODE="${MODEL}-Thinking"
        else
            MODEL_MODE="${MODEL}-No-Thinking"
        fi

        BASE_DIR="/mnt/TaxoBench/results/setting2"
        EVAL_OUTPUT="${BASE_DIR}/${MODEL_MODE}/${MODEL}.jsonl"
        LOG_DIR="${BASE_DIR}/log"
        LOG_FILE="${LOG_DIR}/${MODEL_MODE}.log"

        mkdir -p "$LOG_DIR"
        mkdir -p "$(dirname "$EVAL_OUTPUT")"

        # --- Print Task Info ---
        echo "----------------------------------------"
        echo "🚀 Starting Task:"
        echo "   Provider: $PROVIDER"
        echo "   Model:    $MODEL"
        echo "   Mode:     $MODEL_MODE"
        echo "   Log:      $LOG_FILE"
        echo "----------------------------------------"

        # --- Write Log Header ---
        {
            echo "=============== Parameters ===============" 
            echo "Time:        $(date)"
            echo "Provider:    $PROVIDER"
            echo "Model:       $MODEL"
            echo "Thinking:    $THINKING_FLAG"
            echo "Temperature: $TEMP"
            echo "Workers:     $NUM_WORKERS"
            echo "==========================================" 
        } >> "$LOG_FILE"

        # --- Execute Python Script ---
        "$PYTHON_EXEC" "$PYTHON_SCRIPT" \
            --path "$INPUT_PATH" \
            --eval_path "$EVAL_OUTPUT" \
            --num_workers "$NUM_WORKERS" \
            --provider "$PROVIDER" \
            --model "$MODEL" \
            --temperature "$TEMP" \
            $THINKING_FLAG \
            >> "$LOG_FILE" 2>&1
        
        # Check execution status
        if [ $? -eq 0 ]; then
            echo "✅ Task Finished: $MODEL_MODE"
        else
            echo "❌ Task Failed: $MODEL_MODE (Check log: $LOG_FILE)"
        fi

    done
done

echo "🎉 All tasks completed."