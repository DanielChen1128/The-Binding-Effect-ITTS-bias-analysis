#!/bin/bash

# Script to run specific experiment JSON files on corresponding models
# Usage: bash run_experiments.sh

set -e  # Exit on error

# Paths can be overridden via environment variables; defaults resolve to this repo.
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTPUT_BASE="${OUTPUT_BASE:-./ITTS_audios}"

echo "=========================================="
echo "Starting TTS generation for experiments"
echo "=========================================="
echo ""

# Define model-experiment pairs
# Format: "experiment_file:model_name:conda_env"
declare -a EXPERIMENTS=(
    "PromptTTSpp_experiment.json:promptttspp:BindingBias"
    "Parler-L_experiment.json:parler-large:BindingBias"
    "Parler-M_experiment.json:parler-mini:BindingBias"
    "VoxInstruct_experiment.json:voxinstruct:voxinstruct"
)

# Run each experiment
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r json_file model_name conda_env <<< "$experiment"
    
    json_path="${SCRIPT_DIR}/${json_file}"
    json_basename=$(basename "$json_file" .json)
    
    # Check if JSON file exists
    if [ ! -f "$json_path" ]; then
        echo "[ERROR] File not found: ${json_path}"
        echo "Skipping..."
        echo ""
        continue
    fi
    
    echo "=========================================="
    echo "Processing: ${json_basename}"
    echo "Model: ${model_name}"
    echo "Conda Environment: ${conda_env}"
    echo "=========================================="
    echo ""
    
    # Run the model with appropriate conda environment
    conda run -n "${conda_env}" python "${SCRIPT_DIR}/generate_wav.py" \
        --model "${model_name}" \
        --json "${json_path}" \
        --output "${OUTPUT_BASE}/${model_name}/${json_basename}" \
        || echo "[WARNING] ${model_name} failed on ${json_basename}"
    
    echo ""
    echo "Completed: ${json_basename} with ${model_name}"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Output directory: ${OUTPUT_BASE}"
