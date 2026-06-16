# solution from codex on ocrd-core
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

# eynollah inference failed when this was > 1
export OCRD_MAX_PARALLEL_PAGES=1

export CUDA_VISIBLE_DEVICES=0

# processor name should exclude prefix "ocrd-" since ocrd.core will add it back when looking up the processor class
# e.g. "eynollah-inference" instead of "ocrd-eynollah-inference"
ocrd-eynollah-inference -m "$METS" -U "$SOCK" -I OCR-D-IMG -O OCR-D-EYNOLLAH -P model eynollah-scale-bin-20260325-artbound-noheadings

ocrd-ecpo-segment -m "$METS" -U "$SOCK" -I OCR-D-EYNOLLAH -O OCR-D-ECPO