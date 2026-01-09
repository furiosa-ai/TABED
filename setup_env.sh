#!/bin/bash
# TABED Environment Setup Script
#
# This script sets up the environment variables for TABED experiments.
# Source this file to configure your environment:
#   source setup_env.sh
#
# Or add these exports to your ~/.bashrc or ~/.zshrc

# =============================================================================
# TABED Path Configuration
# =============================================================================

# Root directory for all TABED data
# Previously hardcoded as: /pvc/home-mjlee
export TABED_ROOT="${TABED_ROOT:-${HOME}/data/tabed}"

# Directory for datasets
# Previously hardcoded as: /pvc/home-mjlee/data/MSD/datasets
export TABED_DATA_DIR="${TABED_DATA_DIR:-${TABED_ROOT}/datasets}"

# Directory for experiment results
# Previously hardcoded as: /pvc/home-mjlee/data/MSD/MLLMSD_Results
export TABED_RESULTS_DIR="${TABED_RESULTS_DIR:-${TABED_ROOT}/results}"

# Directory for model checkpoints
# Previously hardcoded as: /pvc/home-mjlee/data/MSD/checkpoint
export TABED_CHECKPOINT_DIR="${TABED_CHECKPOINT_DIR:-${TABED_ROOT}/checkpoint}"

# Directory for numpy output files
# Previously hardcoded as: /pvc/home-mjlee/data/MSD/npy
export TABED_NPY_DIR="${TABED_NPY_DIR:-${TABED_ROOT}/npy}"

# =============================================================================
# LLaVA Data Configuration
# =============================================================================

# Directory for LLaVA instruction data (M4-Instruct-Data)
# Previously hardcoded as: /pvc/home-mjlee/data/llava-next/llava_instruct
export LLAVA_DATA_DIR="${LLAVA_DATA_DIR:-${HOME}/data/llava-next/llava_instruct}"

# =============================================================================
# Create directories if they don't exist
# =============================================================================

mkdir -p "${TABED_ROOT}"
mkdir -p "${TABED_DATA_DIR}"
mkdir -p "${TABED_RESULTS_DIR}"
mkdir -p "${TABED_CHECKPOINT_DIR}"
mkdir -p "${TABED_NPY_DIR}"

# =============================================================================
# Print configuration
# =============================================================================

echo "TABED Environment Configured:"
echo "  TABED_ROOT:           ${TABED_ROOT}"
echo "  TABED_DATA_DIR:       ${TABED_DATA_DIR}"
echo "  TABED_RESULTS_DIR:    ${TABED_RESULTS_DIR}"
echo "  TABED_CHECKPOINT_DIR: ${TABED_CHECKPOINT_DIR}"
echo "  TABED_NPY_DIR:        ${TABED_NPY_DIR}"
echo "  LLAVA_DATA_DIR:       ${LLAVA_DATA_DIR}"
