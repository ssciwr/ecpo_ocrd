#!/bin/bash
set -euo pipefail

# solution from codex on ocrd-core

# get path to workspace from the command line argument
# defaults to current directory if not provided
WS="$(realpath "${1:-.}")"
PROCESSOR="${2:?Missing processor name}"
INPUT_GROUP="${3:?Missing input file group}"
OUTPUT_GROUP="${4:?Missing output file group}"

# remove the first 4 arguments
# so that the remaining ones can be passed to the processor
shift 4

METS="$WS/mets.xml"
SOCK=/tmp/ocrd-mets-$$.sock

# The server command is blocking, so run it in the background.
ocrd workspace -d "$WS" -U "$SOCK" server start &
METS_SERVER_PID=$!

cleanup() {
    ocrd workspace -d "$WS" -U "$SOCK" server stop >/dev/null 2>&1 || true
    wait "$METS_SERVER_PID" >/dev/null 2>&1 || true
}

trap cleanup EXIT

# Wait until the Unix socket is actually created
echo "Waiting for METS server socket: $SOCK"

for i in $(seq 1 100); do
    if [ -S "$SOCK" ]; then
        echo "METS server ready"
        break
    fi

    # Fail early if the server died during startup
    if ! kill -0 "$METS_SERVER_PID" 2>/dev/null; then
        echo "METS server exited unexpectedly"
        exit 1
    fi

    sleep 0.1
done

# Verify readiness
if [ ! -S "$SOCK" ]; then
    echo "Timed out waiting for METS server socket"
    exit 1
fi

export OCRD_MAX_PARALLEL_PAGES=8
export CUDA_VISIBLE_DEVICES=0
export OCRD_EXISTING_OUTPUT=ABORT
export OCRD_MISSING_OUTPUT=ABORT

# run the specified processor with the given arguments
"$PROCESSOR" \
    -m "$METS" \
    -U "$SOCK" \
    -I "$INPUT_GROUP" \
    -O "$OUTPUT_GROUP" \
    "$@"