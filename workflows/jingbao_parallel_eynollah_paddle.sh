#!/usr/bin/env bash
set -euo pipefail

WS="${1:-.}"

# Adjust if needed
ENV_EYNOLLAH="${2:?Missing env name for Eynollah processes}"
ENV_NON_EYNOLLAH="${3:?Missing env name for non-Eynollah processes}"
PROCESSOR_SCRIPT="./individual/jingbao_parallel_individual.sh"

# usage
if [[ "$WS" == "--help" || "$WS" == "-h" ]]; then
    echo "Usage: $0 <path_to_workspace> <eynollah_env_name> <non_eynollah_env_name>"
    exit 0
fi

echo "Running Eynollah inference step..."
conda run -n "$ENV_EYNOLLAH" \
    bash "$PROCESSOR_SCRIPT" \
    "$WS" \
    ocrd-eynollah-inference \
    OCR-D-IMG \
    OCR-D-EYNOLLAH \
    -P model eynollah-scale-bin-20260325-artbound-noheadings

echo "Running ECPO segmentation step..."
conda run -n "$ENV_NON_EYNOLLAH" \
    bash "$PROCESSOR_SCRIPT" \
    "$WS" \
    ocrd-ecpo-segment \
    OCR-D-EYNOLLAH \
    OCR-D-ECPO