from functools import lru_cache
from pathlib import Path
from typing import Optional

from ocrd import OcrdPage, OcrdPageResult, Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

import click


@lru_cache()
def modern_character_replacement_dict():
    """Load our replacements into a dictionary."""

    with open(Path(__file__).parent / "replacements.txt", "r", encoding="utf-8") as f:
        return {line[0]: line[1] for line in f.readlines()}


def normalize(text: str) -> str:
    for old, new in modern_character_replacement_dict().items():
        text = text.replace(old, new)
    return text


def normalize_text_equivs(segment):
    for text_equiv in segment.get_TextEquiv():
        text = text_equiv.get_Unicode()
        if text is not None:
            text_equiv.set_Unicode(normalize(text))


class YitiziProcessor(Processor):
    """OCR-D processor for normalizing Chinese variant characters in PAGE text."""

    def process_page_pcgts(
        self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None
    ) -> OcrdPageResult:
        assert input_pcgts
        assert input_pcgts[0]

        pcgts = input_pcgts[0]
        page = pcgts.get_Page()
        for region in page.get_AllRegions(classes=["Text"]):
            normalize_text_equivs(region)
            for line in region.get_TextLine():
                normalize_text_equivs(line)
                for word in line.get_Word():
                    normalize_text_equivs(word)
                    for glyph in word.get_Glyph():
                        normalize_text_equivs(glyph)
        return OcrdPageResult(pcgts)


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(YitiziProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
