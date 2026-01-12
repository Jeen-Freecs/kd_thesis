#!/bin/bash

###############################################################################
# Automated Training and Evaluation Pipeline
# 
# This script:
# 1. Trains models with different configurations sequentially
# 2. Evaluates each trained model on test set
# 3. Logs all results
# 4. Handles errors gracefully
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_DIR="logs"
mkdir -p "${LOG_DIR}/training" "${LOG_DIR}/evaluation" "${LOG_DIR}/results" "${LOG_DIR}/archive"
RESULTS_FILE="${LOG_DIR}/results/all_results_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================" | tee -a "${RESULTS_FILE}"
echo "Starting Experiment Pipeline" | tee -a "${RESULTS_FILE}"
echo "Date: $(date)" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"

###############################################################################
# Configuration List
# Add or remove configs as needed
###############################################################################

CONFIGS=(
    # Baseline (no KD)
    # "configs/baseline_config.yaml"
    
    # # Single-teacher Dynamic KD (Method 2)
    # "configs/single_teacher_densenet.yaml"
    # "configs/single_teacher_resnet50.yaml"
    # "configs/single_teacher_vit.yaml"
    
    # PAT: Perspective-Aware Teaching (arXiv:2501.08885)
    # For heterogeneous architectures (CNN ↔ ViT)
    "configs/pat_densenet.yaml"
    "configs/pat_resnet50.yaml"
    "configs/pat_vit.yaml"
)

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

extract_run_id() {
    # Extract WandB run ID from training log
    local log_file=$1
    local run_id=$(grep -oE "WandB Run ID: [a-z0-9]+" "${log_file}" | tail -1 | cut -d' ' -f4)
    echo "${run_id}"
}

extract_checkpoint_dir() {
    # Extract checkpoint directory from training log
    local log_file=$1
    local checkpoint_dir=$(grep -oE "Checkpoints will be saved to: .+" "${log_file}" | tail -1 | cut -d':' -f2- | xargs)
    echo "${checkpoint_dir}"
}

###############################################################################
# Main Experiment Loop
###############################################################################

TOTAL_CONFIGS=${#CONFIGS[@]}
CURRENT=0
SUCCESSFUL=0
FAILED=0

for config in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    print_header "Experiment ${CURRENT}/${TOTAL_CONFIGS}: ${config}"
    echo "" | tee -a "${RESULTS_FILE}"
    echo "----------------------------------------" | tee -a "${RESULTS_FILE}"
    echo "Config ${CURRENT}/${TOTAL_CONFIGS}: ${config}" | tee -a "${RESULTS_FILE}"
    echo "----------------------------------------" | tee -a "${RESULTS_FILE}"
    
    # Check if config exists
    if [ ! -f "${config}" ]; then
        print_error "Config file not found: ${config}"
        echo "SKIPPED: Config file not found" | tee -a "${RESULTS_FILE}"
        echo "" | tee -a "${RESULTS_FILE}"
        continue
    fi
    
    # Create temporary log file for this run
    TRAIN_LOG="${LOG_DIR}/training/train_$(basename ${config%.yaml})_$(date +%Y%m%d_%H%M%S).log"
    EVAL_LOG="${LOG_DIR}/evaluation/eval_$(basename ${config%.yaml})_$(date +%Y%m%d_%H%M%S).log"
    
    ###########################################################################
    # Step 1: Training
    ###########################################################################
    
    print_info "Starting training with ${config}..."
    echo "Training started at: $(date)" | tee -a "${RESULTS_FILE}"
    
    if python scripts/train.py --config "${config}" 2>&1 | tee "${TRAIN_LOG}"; then
        print_success "Training completed successfully"
        echo "Training completed at: $(date)" | tee -a "${RESULTS_FILE}"
        
        # Extract run information
        RUN_ID=$(extract_run_id "${TRAIN_LOG}")
        CHECKPOINT_DIR=$(extract_checkpoint_dir "${TRAIN_LOG}")
        
        if [ -z "${RUN_ID}" ]; then
            print_error "Could not extract run ID from training log"
            echo "ERROR: Could not extract run ID" | tee -a "${RESULTS_FILE}"
            FAILED=$((FAILED + 1))
            echo "" | tee -a "${RESULTS_FILE}"
            continue
        fi
        
        print_info "Run ID: ${RUN_ID}"
        print_info "Checkpoint directory: ${CHECKPOINT_DIR}"
        echo "Run ID: ${RUN_ID}" | tee -a "${RESULTS_FILE}"
        echo "Checkpoint directory: ${CHECKPOINT_DIR}" | tee -a "${RESULTS_FILE}"
        
        ###########################################################################
        # Step 2: Evaluation
        ###########################################################################
        
        print_info "Starting evaluation..."
        echo "Evaluation started at: $(date)" | tee -a "${RESULTS_FILE}"
        
        # Find the best checkpoint (saved as "best.ckpt" by ModelCheckpoint)
        BEST_CHECKPOINT="${CHECKPOINT_DIR}/best.ckpt"
        
        if [ -f "${BEST_CHECKPOINT}" ]; then
            echo "" | tee -a "${RESULTS_FILE}"
            echo "📦 CHECKPOINT BEING EVALUATED:" | tee -a "${RESULTS_FILE}"
            echo "   Path: ${BEST_CHECKPOINT}" | tee -a "${RESULTS_FILE}"
            echo "   Size: $(du -h "${BEST_CHECKPOINT}" | cut -f1)" | tee -a "${RESULTS_FILE}"
            echo "   Modified: $(stat -c %y "${BEST_CHECKPOINT}" 2>/dev/null || stat -f %Sm "${BEST_CHECKPOINT}" 2>/dev/null)" | tee -a "${RESULTS_FILE}"
            echo "" | tee -a "${RESULTS_FILE}"
            
            print_info "Using local checkpoint: ${BEST_CHECKPOINT}"
            
            # Evaluate on test set
            if python scripts/evaluate.py --checkpoint "${BEST_CHECKPOINT}" --config "${config}" --split test 2>&1 | tee "${EVAL_LOG}"; then
                print_success "Evaluation completed successfully"
                echo "Evaluation completed at: $(date)" | tee -a "${RESULTS_FILE}"
                
                # Extract and save results
                grep -A 10 "EVALUATION RESULTS" "${EVAL_LOG}" | tee -a "${RESULTS_FILE}"
                SUCCESSFUL=$((SUCCESSFUL + 1))
            else
                print_error "Evaluation failed"
                echo "ERROR: Evaluation failed" | tee -a "${RESULTS_FILE}"
                FAILED=$((FAILED + 1))
            fi
        else
            print_error "Best checkpoint not found at ${BEST_CHECKPOINT}"
            echo "ERROR: Best checkpoint not found at ${BEST_CHECKPOINT}" | tee -a "${RESULTS_FILE}"
            
            # Try to evaluate from WandB
            print_info "Attempting to download from WandB artifact..."
            echo "" | tee -a "${RESULTS_FILE}"
            echo "📦 CHECKPOINT BEING EVALUATED:" | tee -a "${RESULTS_FILE}"
            echo "   Source: WandB artifact (model-${RUN_ID}:best)" | tee -a "${RESULTS_FILE}"
            echo "" | tee -a "${RESULTS_FILE}"
            
            if python scripts/evaluate.py --wandb-run-id "${RUN_ID}" --config "${config}" --split test 2>&1 | tee "${EVAL_LOG}"; then
                print_success "Evaluation completed (from WandB)"
                echo "Evaluation completed at: $(date)" | tee -a "${RESULTS_FILE}"
                
                # Extract and save results
                grep -A 10 "EVALUATION RESULTS" "${EVAL_LOG}" | tee -a "${RESULTS_FILE}"
                SUCCESSFUL=$((SUCCESSFUL + 1))
            else
                print_error "Evaluation failed"
                echo "ERROR: Evaluation failed" | tee -a "${RESULTS_FILE}"
                FAILED=$((FAILED + 1))
            fi
        fi
        
    else
        print_error "Training failed for ${config}"
        echo "ERROR: Training failed" | tee -a "${RESULTS_FILE}"
        FAILED=$((FAILED + 1))
    fi
    
    echo "" | tee -a "${RESULTS_FILE}"
    
    # Optional: Add delay between experiments
    if [ ${CURRENT} -lt ${TOTAL_CONFIGS} ]; then
        print_info "Waiting 5 seconds before next experiment..."
        sleep 5
    fi
    
done

###############################################################################
# Summary
###############################################################################

echo "" | tee -a "${RESULTS_FILE}"
print_header "Pipeline Complete!"
echo "" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "Summary" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "Total experiments: ${TOTAL_CONFIGS}" | tee -a "${RESULTS_FILE}"
echo "Successful: ${SUCCESSFUL}" | tee -a "${RESULTS_FILE}"
echo "Failed: ${FAILED}" | tee -a "${RESULTS_FILE}"
echo "Completion time: $(date)" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"
echo "Results saved to: ${RESULTS_FILE}" | tee -a "${RESULTS_FILE}"

if [ ${SUCCESSFUL} -eq ${TOTAL_CONFIGS} ]; then
    print_success "All experiments completed successfully! 🎉"
    exit 0
elif [ ${SUCCESSFUL} -gt 0 ]; then
    print_info "Some experiments completed successfully"
    exit 1
else
    print_error "All experiments failed"
    exit 1
fi
