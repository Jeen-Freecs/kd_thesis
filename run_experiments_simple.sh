#!/bin/bash

###############################################################################
# Simple Training and Evaluation Script
# 
# Usage: ./run_experiments_simple.sh config1.yaml config2.yaml ...
# Or edit the CONFIGS array below and run: ./run_experiments_simple.sh
###############################################################################

# If arguments provided, use them; otherwise use default configs
if [ $# -gt 0 ]; then
    CONFIGS=("$@")
else
    # Default configs to run
    CONFIGS=(
        "configs/baseline_config.yaml"
        "configs/method2_dynamic_kd.yaml"
    )
fi

echo "=========================================="
echo "Running ${#CONFIGS[@]} experiment(s)"
echo "=========================================="
echo ""

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    num=$((i + 1))
    
    echo "=========================================="
    echo "Experiment ${num}/${#CONFIGS[@]}: ${config}"
    echo "=========================================="
    
    # Train
    echo "⚡ Training..."
    python scripts/train.py --config "${config}"
    
    if [ $? -eq 0 ]; then
        echo "✓ Training completed"
        
        # Extract run ID from last training output
        RUN_ID=$(grep "WandB Run ID:" logs/train.log | tail -1 | awk '{print $NF}')
        
        if [ ! -z "${RUN_ID}" ]; then
            echo "🔍 Evaluating run: ${RUN_ID}"
            
            # Evaluate
            python scripts/evaluate.py --wandb-run-id "${RUN_ID}" --config "${config}" --split test
            
            if [ $? -eq 0 ]; then
                echo "✓ Evaluation completed"
            else
                echo "✗ Evaluation failed"
            fi
        else
            echo "⚠ Could not find run ID, skipping evaluation"
        fi
    else
        echo "✗ Training failed"
    fi
    
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

