"""Check the VLM's answer to the reading order prompt.

Shared by vlm_ocr.py and make_vllm_request.py. The reading order prompt
allows exactly one of READING_ORDERS as answer. Anything else (e.g. an
explanation, a made-up option, or an answer stuck repeating itself) is
replaced by DEFAULT_READING_ORDER, so that the OCR prompt always gets a valid
reading order. The reading order request is therefore sent only once, without
repetition penalty and without retries.
"""

import re

READING_ORDERS = (
    "vertical, right-to-left",
    "horizontal, right-to-left",
    "horizontal, left-to-right",
)

# the most common layout of the newspaper's text
DEFAULT_READING_ORDER = "vertical, right-to-left"


def parse_reading_order(answer: str | None) -> str | None:
    """Return the reading order the answer names, None if it is not one of
    READING_ORDERS.

    Case, surrounding whitespace, a trailing full stop and spaces around the
    comma and hyphens are ignored, e.g. " Vertical , right-to-left. " gives
    "vertical, right-to-left".
    """
    if not answer:
        return None
    normalized = answer.strip().lower().rstrip(".。").strip()
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    return normalized if normalized in READING_ORDERS else None
