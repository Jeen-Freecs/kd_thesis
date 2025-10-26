#!/bin/bash
#
# Run All Knowledge Distillation Experiments
# This script runs all 12 experiments from the AML Final Project
#
# Usage: bash run_all_experiments.sh
# Or: nohup ./run_all_experiments.sh > all_experiments.log 2>&1 &

set -e  # Exit on error

echo "========================================"
echo "Knowledge Distillation Experiments"
echo "Starting all experiments..."
echo "========================================"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kd-env

# Create results directory
mkdir -p results
mkdir -p logs

# Log start time
START_TIME=$(date +%s)
echo "Start time: $(date)"
echo ""

# Array of experiments with descriptions
declare -A experiments
experiments=(
    ["configs/baseline_config.yaml"]="Baseline - Student from scratch (66.29%)"
    ["configs/single_teacher_densenet.yaml"]="Single Teacher DenseNet-121 α=0.50 (75.38%) ⭐ BEST"
    ["configs/single_teacher_resnet50.yaml"]="Single Teacher ResNet-50 α=0.75 (73.75%)"
    ["configs/single_teacher_vit.yaml"]="Single Teacher ViT α=0.25 (74.00%)"
    ["configs/method1_ca_wkd.yaml"]="Method 1: CA-WKD Diverse Ensemble (73.33%)"
    ["configs/method2_diverse_ensemble.yaml"]="Method 2: α-Guided Diverse Ensemble (71.38%)"
    ["configs/method2_dynamic_kd.yaml"]="Method 2: α-Guided 3 ResNets"
    ["configs/method3_densenet_resnet_temp8.yaml"]="Method 3: DenseNet+ResNet temp=8.0 (74.00%) ⭐ BEST MULTI"
    ["configs/method3_densenet_resnet_temp16.yaml"]="Method 3: DenseNet+ResNet temp=16.0 (73.96%)"
    ["configs/method3_adaptive_alpha.yaml"]="Method 3: 3 ResNets (72.90%)"
    ["configs/method3_diverse_ensemble.yaml"]="Method 3: Diverse Ensemble (72.10%)"
    ["configs/method3_3resnets_no_ce.yaml"]="Method 3: 3 ResNets No CE Loss (71.89%)"
)

# Counter
TOTAL=${#experiments[@]}
CURRENT=0
SUCCESSFUL=0
FAILED=0

# Run each experiment
for config in "${!experiments[@]}"; do
    CURRENT=$((CURRENT + 1))
    DESCRIPTION="${experiments[$config]}"
    
    echo "=========================================="
    echo "Experiment $CURRENT of $TOTAL"
    echo "Config: $config"
    echo "Description: $DESCRIPTION"
    echo "=========================================="
    
    # Extract experiment name from config path
    EXP_NAME=$(basename "$config" .yaml)
    
    # Run experiment
    if python scripts/train.py --config "$config"; then
        echo "✅ SUCCESS: $EXP_NAME"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "❌ FAILED: $EXP_NAME"
        FAILED=$((FAILED + 1))
        # Continue to next experiment even if this one failed
        continue
    fi
    
    echo ""
    echo "Progress: $CURRENT/$TOTAL completed ($SUCCESSFUL successful, $FAILED failed)"
    echo ""
    
    # Small pause between experiments
    sleep 5
done

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "End time: $(date)"
echo ""
echo "Results saved in: lightning_logs/"
echo "View on wandb: https://wandb.ai"
echo "=========================================="

# Deactivate conda
conda deactivate

