# solution from codex on ocrd-core

# get path to workspace from the command line argument

WS="$1"

if [ -z "$WS" ]; then
  echo "Usage: $0 <path_to_workspace>"
  exit 1
fi

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

# try to run Eynollah inference with more than one parallel worker,
# after moving model loading to inside process_page_pcgts
ocrd-eynollah-inference -m "$METS" -U "$SOCK" \
    -I OCR-D-IMG -O OCR-D-EYNOLLAH-TEST-8 \
    -P model eynollah-scale-bin-20260325-artbound-noheadings