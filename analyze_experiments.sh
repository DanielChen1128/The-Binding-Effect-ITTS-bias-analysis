#!/bin/bash
set -e

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "[INFO] Legacy per-model experiment JSONs are not distributed."
echo "[INFO] Analyzing the actual JSON files under ${DESC_DIR:-${SCRIPT_DIR}/descriptions}."
exec bash "${SCRIPT_DIR}/analyze_all_models.sh"
