import os
from pathlib import Path
from typing import Optional

from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd.processor.base import MissingInputFile
from ocrd_models.ocrd_file import OcrdFileType
from ocrd_modelfactory import page_from_file
from ocrd_utils import config, make_file_id

import click


def _first_unicode(segment) -> Optional[str]:
    for text_equiv in segment.get_TextEquiv():
        text = text_equiv.get_Unicode()
        if text:
            return text
    return None


def _word_text(word) -> Optional[str]:
    text = _first_unicode(word)
    if text:
        return text

    glyph_texts = [
        text for glyph in word.get_Glyph() if (text := _first_unicode(glyph))
    ]
    if glyph_texts:
        return "".join(glyph_texts)
    return None


def _line_text(line) -> Optional[str]:
    text = _first_unicode(line)
    if text:
        return text

    word_texts = [text for word in line.get_Word() if (text := _word_text(word))]
    if word_texts:
        return " ".join(word_texts)
    return None


def page_text(page) -> str:
    """Return the PAGE text content as one newline-separated string."""
    texts = []
    for region in page.get_AllRegions(classes=["Text"], order="reading-order"):
        line_texts = [
            text for line in region.get_TextLine() if (text := _line_text(line))
        ]
        if line_texts:
            texts.extend(line_texts)
            continue

        region_text = _first_unicode(region)
        if region_text:
            texts.append(region_text)

    return "\n".join(texts)


class TextsoupProcessor(Processor):
    """OCR-D processor to concatenate OCR results into text files for full text search"""

    def process_page_file(self, *input_files: Optional[OcrdFileType]) -> None:
        input_file = input_files[0]
        assert input_file
        page_id = input_file.pageId
        self.logger.info("exporting PAGE text soup for page %s", page_id)

        if not input_file.local_filename:
            self.logger.error("No local file exists for page %s", page_id)
            if config.OCRD_MISSING_INPUT == "ABORT":
                raise MissingInputFile(
                    self.input_file_grp, page_id, input_file.mimetype
                )
            return

        output_file_id = make_file_id(input_file, self.output_file_grp)
        if config.OCRD_EXISTING_OUTPUT != "OVERWRITE":
            if output_file := next(
                self.workspace.mets.find_files(ID=output_file_id), None
            ):
                raise FileExistsError(
                    f"A file with ID=={output_file_id} already exists {output_file}"
                    " and OCRD_EXISTING_OUTPUT != OVERWRITE"
                )

        pcgts = page_from_file(input_file)
        page = pcgts.get_Page()
        image_basename = Path(page.imageFilename or input_file.local_filename).stem
        output_local_filename = os.path.join(
            self.output_file_grp,
            f"{image_basename}.txt",
        )

        text = page_text(page)
        if text:
            text += "\n"

        self.workspace.add_file(
            file_id=output_file_id,
            file_grp=self.output_file_grp,
            page_id=page_id,
            local_filename=output_local_filename,
            mimetype="text/plain",
            content=text,
        )


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(TextsoupProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
