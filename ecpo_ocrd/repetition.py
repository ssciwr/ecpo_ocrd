"""Detect VLM answers that got stuck repeating themselves, and the repetition
penalties to retry them with.

Shared by vlm_ocr.py (retries with a higher repetition penalty),
make_vllm_request.py (the same, for single crops) and crop_and_analyze.py
(counts the regions that still repeat).
"""

import re

# a unit of up to 10 characters (including line breaks), repeated right after itself
REPEAT_RE = re.compile(r"(.{1,10}?)\1+", re.DOTALL)

# an answer counts as repetition if one unit repeats at least this many times
# in a row, the default of crop_and_analyze.py --min-repeat-times
MIN_REPEAT_TIMES = 50

# every request starts without penalty, each retry increases it by this step
REPETITION_PENALTY_START = 1.0
REPETITION_PENALTY_STEP = 0.05


def most_repeat(text: str) -> tuple[str, int]:
    """Return the unit repeated most often in a row, and how many times it occurs.

    The run ends where another unit appears or the text ends, e.g. "abc，，，，d"
    gives ("，", 4) and "一元二角\\n一元二角\\n" gives ("一元二角\\n", 2). Text
    without any unit directly repeated gives ("", 0).
    """
    unit, times = "", 0
    for match in REPEAT_RE.finditer(text):
        run_times = (match.end() - match.start()) // len(match.group(1))
        if run_times > times:
            unit, times = match.group(1), run_times
    return unit, times


def is_repetition(
    text: str, finish_reason: str | None, min_repeat_times: int = MIN_REPEAT_TIMES
) -> bool:
    """Whether the answer got stuck repeating itself: it was cut off at
    max_tokens (finish_reason "length", which also catches loops of units
    longer than 10 characters), or one unit repeats at least min_repeat_times
    times in a row."""
    return finish_reason == "length" or most_repeat(text)[1] >= min_repeat_times


def repetition_penalties(max_repetition_penalty: float) -> list[float]:
    """The penalties to try in order: 1.0, then increasing by 0.05 up to
    max_repetition_penalty, e.g. 1.2 gives [1.0, 1.05, 1.1, 1.15, 1.2]."""
    penalties = []
    i = 0
    while True:
        penalty = round(REPETITION_PENALTY_START + i * REPETITION_PENALTY_STEP, 4)
        if penalty > max_repetition_penalty + 1e-9:
            break
        penalties.append(penalty)
        i += 1
    return penalties or [REPETITION_PENALTY_START]
