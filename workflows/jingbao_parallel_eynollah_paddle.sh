#!/usr/bin/env bash
set -euo pipefail

WS="${1:-.}"

# Adjust if needed
ENV_EYNOLLAH="ecpo_eyollah"
ENV_NON_EYNOLLAH="ecpo_non_eynollah"
PROCESSOR_SCRIPT="./individual/jingbao_parallel_individual.sh"

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