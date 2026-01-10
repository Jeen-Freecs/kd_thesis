#!/bin/bash

###############################################################################
# Evaluation-Only Pipeline for Multi-Teacher KD Experiments
# 
# This script:
# 1. Runs evaluation ONLY (no training) for all method configs
# 2. Skips baseline and single-teacher configs
# 3. Uses proper wandb naming from config files
###############################################################################

set -e  # Exit on error
set -o pipefail  # Catch errors in piped commands

# Python executable path (venv with torch installed)
PYTHON="/venv/kd-env/bin/python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_DIR="logs"
mkdir -p "${LOG_DIR}/evaluation" "${LOG_DIR}/results"
RESULTS_FILE="${LOG_DIR}/results/eval_results_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================" | tee -a "${RESULTS_FILE}"
echo "Evaluation-Only Pipeline" | tee -a "${RESULTS_FILE}"
echo "Date: $(date)" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"

###############################################################################
# Configuration List - All method configs (excluding baseline & single teacher)
###############################################################################

CONFIGS=(
    "configs/method1_ca_wkd.yaml"
    "configs/method2_diverse_ensemble.yaml"
    "configs/method2_dynamic_kd.yaml"
    "configs/method3_3resnets_no_ce.yaml"
    "configs/method3_adaptive_alpha.yaml"
    "configs/method3_densenet_resnet_temp16.yaml"
    "configs/method3_densenet_resnet_temp8.yaml"
    "configs/method3_diverse_ensemble.yaml"
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

extract_wandb_name() {
    # Extract wandb.name from config file
    local config_file=$1
    local name=$(grep -A5 "^wandb:" "${config_file}" | grep "name:" | head -1 | sed 's/.*name:[[:space:]]*"\?\([^"]*\)"\?.*/\1/' | xargs)
    echo "${name}"
}

find_checkpoint() {
    # Find the best.ckpt for a given experiment name
    local experiment_name=$1
    local checkpoint=$(find checkpoints -type f -name "best.ckpt" -path "*${experiment_name}*" 2>/dev/null | head -1)
    echo "${checkpoint}"
}

###############################################################################
# Main Evaluation Loop
###############################################################################

TOTAL_CONFIGS=${#CONFIGS[@]}
CURRENT=0
SUCCESSFUL=0
FAILED=0
SKIPPED=0

for config in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    print_header "Evaluation ${CURRENT}/${TOTAL_CONFIGS}: ${config}"
    echo "" | tee -a "${RESULTS_FILE}"
    echo "----------------------------------------" | tee -a "${RESULTS_FILE}"
    echo "Config ${CURRENT}/${TOTAL_CONFIGS}: ${config}" | tee -a "${RESULTS_FILE}"
    echo "----------------------------------------" | tee -a "${RESULTS_FILE}"
    
    # Check if config exists
    if [ ! -f "${config}" ]; then
        print_error "Config file not found: ${config}"
        echo "SKIPPED: Config file not found" | tee -a "${RESULTS_FILE}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # Extract experiment name from config
    EXPERIMENT_NAME=$(extract_wandb_name "${config}")
    
    if [ -z "${EXPERIMENT_NAME}" ]; then
        print_error "Could not extract experiment name from config"
        echo "SKIPPED: Could not extract experiment name" | tee -a "${RESULTS_FILE}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    print_info "Experiment: ${EXPERIMENT_NAME}"
    echo "Experiment: ${EXPERIMENT_NAME}" | tee -a "${RESULTS_FILE}"
    
    # Find checkpoint for this experiment
    CHECKPOINT=$(find_checkpoint "${EXPERIMENT_NAME}")
    
    if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
        print_error "Checkpoint not found for: ${EXPERIMENT_NAME}"
        echo "SKIPPED: Checkpoint not found" | tee -a "${RESULTS_FILE}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    print_info "Checkpoint: ${CHECKPOINT}"
    echo "Checkpoint: ${CHECKPOINT}" | tee -a "${RESULTS_FILE}"
    
    # Create evaluation log file
    EVAL_LOG="${LOG_DIR}/evaluation/eval_$(basename ${config%.yaml})_$(date +%Y%m%d_%H%M%S).log"
    
    ###########################################################################
    # Run Evaluation
    ###########################################################################
    
    print_info "Starting evaluation on test set..."
    echo "Evaluation started at: $(date)" | tee -a "${RESULTS_FILE}"
    
    echo "" | tee -a "${RESULTS_FILE}"
    echo "📦 CHECKPOINT BEING EVALUATED:" | tee -a "${RESULTS_FILE}"
    echo "   Path: ${CHECKPOINT}" | tee -a "${RESULTS_FILE}"
    echo "   Size: $(du -h "${CHECKPOINT}" | cut -f1)" | tee -a "${RESULTS_FILE}"
    echo "   Modified: $(stat -c %y "${CHECKPOINT}" 2>/dev/null || stat -f %Sm "${CHECKPOINT}" 2>/dev/null)" | tee -a "${RESULTS_FILE}"
    echo "" | tee -a "${RESULTS_FILE}"
    
    if ${PYTHON} scripts/evaluate.py --checkpoint "${CHECKPOINT}" --config "${config}" --split test 2>&1 | tee "${EVAL_LOG}"; then
        print_success "Evaluation completed successfully"
        echo "Evaluation completed at: $(date)" | tee -a "${RESULTS_FILE}"
        
        # Extract and save results
        grep -A 15 "EVALUATION RESULTS" "${EVAL_LOG}" | tee -a "${RESULTS_FILE}"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        print_error "Evaluation failed"
        echo "ERROR: Evaluation failed" | tee -a "${RESULTS_FILE}"
        FAILED=$((FAILED + 1))
    fi
    
    echo "" | tee -a "${RESULTS_FILE}"
    
    # Small delay between evaluations
    if [ ${CURRENT} -lt ${TOTAL_CONFIGS} ]; then
        sleep 2
    fi
    
done

###############################################################################
# Summary
###############################################################################

echo "" | tee -a "${RESULTS_FILE}"
print_header "Evaluation Pipeline Complete!"
echo "" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "Summary" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "Total experiments: ${TOTAL_CONFIGS}" | tee -a "${RESULTS_FILE}"
echo "Successful: ${SUCCESSFUL}" | tee -a "${RESULTS_FILE}"
echo "Failed: ${FAILED}" | tee -a "${RESULTS_FILE}"
echo "Skipped: ${SKIPPED}" | tee -a "${RESULTS_FILE}"
echo "Completion time: $(date)" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"
echo "Results saved to: ${RESULTS_FILE}" | tee -a "${RESULTS_FILE}"

if [ ${SUCCESSFUL} -eq ${TOTAL_CONFIGS} ]; then
    print_success "All evaluations completed successfully! 🎉"
    exit 0
elif [ ${SUCCESSFUL} -gt 0 ]; then
    print_info "Some evaluations completed successfully"
    exit 1
else
    print_error "All evaluations failed"
    exit 1
fi

