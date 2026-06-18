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
    echo "Usage: $0 <path_to_workspace> <eynollah_env_name> <non_eynollah_env_name>"
    exit 0
fi

run_live() {
    local cmd="$*"
    script -q -c "$cmd" /dev/null
}

echo "Running Eynollah inference step..."
start_eynollah=$(date +%s)
run_live conda run -n "$ENV_EYNOLLAH" \
    bash "$PROCESSOR_SCRIPT" \
    "$WS" \
    ocrd-eynollah-inference \
    OCR-D-IMG \
    OCR-D-EYNOLLAH \
    -P model eynollah-scale-bin-20260325-artbound-noheadings
end_eynollah=$(date +%s)
echo "Eynollah inference step completed in $((end_eynollah - start_eynollah)) seconds"

echo "Running ECPO segmentation step..."
start_ecpo=$(date +%s)
run_live conda run -n "$ENV_NON_EYNOLLAH" \
    bash "$PROCESSOR_SCRIPT" \
    "$WS" \
    ocrd-ecpo-segment \
    OCR-D-EYNOLLAH \
    OCR-D-ECPO
end_ecpo=$(date +%s)
echo "ECPO segmentation step completed in $((end_ecpo - start_ecpo)) seconds"