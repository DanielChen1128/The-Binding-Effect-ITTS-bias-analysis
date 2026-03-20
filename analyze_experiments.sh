#!/bin/bash

# Script to analyze gender for specific experiment JSON files on corresponding models
# Usage: bash analyze_experiments.sh

set -e  # Exit on error

SCRIPT_DIR="/home/danielgy/ITTS"
DESC_DIR="${SCRIPT_DIR}/descriptions"
WAV_BASE="/home/danielgy/local-data/daniel/ITTS_audios"
ANALYSIS_BASE="${SCRIPT_DIR}/analysis"

echo "=========================================="
echo "Starting Gender Analysis for experiments"
echo "=========================================="
echo ""

# Define model-experiment pairs (matching run_experiments.sh)
# Format: "experiment_file:model_name:conda_env"
declare -a EXPERIMENTS=(
    "PromptTTSpp_experiment.json:promptttspp:BindingBias"
    "Parler-L_experiment.json:parler-large:BindingBias"
    "Parler-M_experiment.json:parler-mini:BindingBias"
    "VoxInstruct_experiment.json:voxinstruct:BindingBias"
)

# Analyze each experiment
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r json_file model_name conda_env <<< "$experiment"
    
    json_path="${SCRIPT_DIR}/${json_file}"
    json_basename=$(basename "$json_file" .json)
    wav_path="${WAV_BASE}/${model_name}/${json_basename}"
    output_path="${ANALYSIS_BASE}/${model_name}/${json_basename}"
    
    echo "=========================================="
    echo "Analyzing: ${json_basename}"
    echo "Model: ${model_name}"
    echo "Conda Environment: ${conda_env}"
    echo "=========================================="
    echo ""
    
    # Check if JSON file exists
    if [ ! -f "$json_path" ]; then
        echo "[ERROR] JSON file not found: ${json_path}"
        echo "Skipping..."
        echo ""
        continue
    fi
    
    # Check if WAV directory exists
    if [ ! -d "${wav_path}" ]; then
        echo "[WARNING] WAV directory not found: ${wav_path}"
        echo "          Skipping..."
        echo ""
        continue
    fi
    
    # Check if there are WAV files
    wav_count=$(find "${wav_path}" -name "*.wav" -type f | wc -l)
    if [ ${wav_count} -eq 0 ]; then
        echo "[WARNING] No WAV files found in: ${wav_path}"
        echo "          Skipping..."
        echo ""
        continue
    fi
    
    echo "------------------------------------------"
    echo "WAV Path: ${wav_path}"
    echo "WAV Count: ${wav_count}"
    echo "Output: ${output_path}"
    echo "------------------------------------------"
    echo ""
    
    # Run gender analysis
    conda run -n "${conda_env}" python "${SCRIPT_DIR}/analyze_gender.py" \
        --wav_path "${wav_path}" \
        --json "${json_path}" \
        --output "${output_path}" \
        || echo "[WARNING] Gender analysis failed for ${model_name}/${json_basename}"
    
    echo ""
    echo "Completed analysis: ${json_basename} with ${model_name}"
    echo ""
done

echo "=========================================="
echo "All experiment analysis completed!"
echo "=========================================="
echo "Results saved to: ${ANALYSIS_BASE}"
echo ""
echo "To view results:"
echo "  - Overall stats: ${ANALYSIS_BASE}/<model>/<experiment>/overall_gender_distribution.csv"
echo "  - By trait: ${ANALYSIS_BASE}/<model>/<experiment>/gender_by_trait.csv"
echo "  - By keyword: ${ANALYSIS_BASE}/<model>/<experiment>/gender_by_keyword.csv"
echo "  - Full results: ${ANALYSIS_BASE}/<model>/<experiment>/detection_results.csv"
echo ""
