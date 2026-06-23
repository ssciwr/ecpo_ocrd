#!/bin/bash
set -euo pipefail

WS="${1:-.}"

# Adjust if needed
ENV_EYNOLLAH="${2:?Missing env name for Eynollah processes}"
ENV_NON_EYNOLLAH="${3:?Missing env name for non-Eynollah processes}"
# Adjust if needed, this path works when running from the root of the repository
PROCESSOR_SCRIPT="workflows/individual/jingbao_parallel_individual.sh"

# usage
if [[ "$WS" == "--help" || "$WS" == "-h" ]]; then
    echo "Usage: $0 <path_to_workspace> <eynollah_gpu_env_name> <non_eynollah_gpu_env_name>"
    exit 0
fi

echo "Running Eynollah inference step..."
start_eynollah=$(date +%s)
conda run --no-capture-output -n "$ENV_EYNOLLAH" \
    bash "$PROCESSOR_SCRIPT" \
    "$WS" \
    ocrd-eynollah-inference \
    OCR-D-IMG \
    OCR-D-EYNOLLAH \
    -P model eynollah-scale-bin-20260325-artbound-noheadings
end_eynollah=$(date +%s)
runtime_eynollah=$((end_eynollah - start_eynollah))
echo "Eynollah inference step completed in $runtime_eynollah seconds"

echo "Running ECPO segmentation step..."
start_ecpo=$(date +%s)
conda run --no-capture-output -n "$ENV_NON_EYNOLLAH" \
    bash "$PROCESSOR_SCRIPT" \
    "$WS" \
    ocrd-ecpo-segment \
    OCR-D-EYNOLLAH \
    OCR-D-ECPO
end_ecpo=$(date +%s)
runtime_ecpo=$((end_ecpo - start_ecpo))
echo "ECPO segmentation step completed in $runtime_ecpo seconds"
echo "Again, Eynollah inference step completed in $runtime_eynollah seconds"
echo "Total runtime: $((runtime_eynollah + runtime_ecpo)) seconds"