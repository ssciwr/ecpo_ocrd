# solution from codex on ocrd-core
WS=/path/to/workspace
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

export OCRD_MAX_PARALLEL_PAGES=5

CUDA_VISIBLE_DEVICES=0 \
ocrd process -m "$METS" -U "$SOCK" \
'ocrd-eynollah-inference -I OCR-D-IMG -O OCR-D-EYNOLLAH -P model eynollah-scale-bin-20260325-artbound-noheadings' \
'ocrd-ecpo-segment -I OCR-D-EYNOLLAH -O OCR-D-ECPO -p "{\"labels\": [\"text\"]}"'