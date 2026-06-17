# solution from codex on ocrd-core and ChatGPT
WS=/mnt/data/tle/ocrd_workspace_test
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

# --- safety settings for GPU + OCR-D ---
# Eynollah inference failed when OCRD_MAX_PARALLEL_PAGES > 1
export OCRD_MAX_PARALLEL_PAGES=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OCRD_EXISTING_OUTPUT=ABORT
export OCRD_MISSING_OUTPUT=ABORT


# --- collect pages ---
PAGES=$(ocrd workspace -d "$WS" -U "$SOCK" list-page)

echo "Found pages:"
echo "$PAGES"


# --- per-page pipeline function ---
process_page () {
    PAGE="$1"

    echo "====================================="
    echo "Processing page: $PAGE"
    echo "====================================="

    # 1) Eynollah inference
    ocrd-eynollah-inference -m "$METS" -U "$SOCK" \
        -I OCR-D-IMG -O OCR-D-EYNOLLAH-M \
        -g $PAGE \
        -P model eynollah-scale-bin-20260325-artbound-noheadings


    if [ $? -ne 0 ]; then
        echo "Eynollah failed on $PAGE"
        return 1
    fi

    # 2) ECPO segmentation
    # ocrd-ecpo-segment -m "$METS" -U "$SOCK" -I OCR-D-EYNOLLAH -O OCR-D-ECPO -g $PAGE

    # if [ $? -ne 0 ]; then
    #     echo "ECPO failed on $PAGE"
    #     return 1
    # fi

    echo "Done page: $PAGE"
}

export -f process_page
export WS METS SOCK


# --- SAFE PARALLEL EXECUTION (2 workers) ---
printf "%s\n" $PAGES | xargs -n 1 -P 2 -I {} bash -c 'process_page "$@"' _ {}