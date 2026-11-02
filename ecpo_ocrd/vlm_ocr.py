import asyncio
import base64
import logging
from io import BytesIO
from multiprocessing import BoundedSemaphore
from typing import List, Optional, Sequence, Tuple

import openai
from ocrd import OcrdPage, OcrdPageResult, Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_models.ocrd_page import TextEquivType

import click
from PIL import Image

OCR_PROMPT_TEMPLATE = """
OCR this image. This image is a crop of a text region from a historical Chinese newspaper, 
so it normally contains text. The text is mostly traditional Chinese; its reading order is {reading_order}.

Your result needs to fulfill **all** of these constraints:
* Transcribe exactly what is printed. Keep traditional and variant characters; 
do not convert to simplified or modern forms, and do not correct errors.
* Keep numbers as printed: Chinese numerals stay Chinese numerals, Arabic digits stay digits.
* Keep punctuation as printed; do not add any.
* Output one line per printed column (vertical text) or row (horizontal text), in the given reading order. 
For vertical right-to-left text, begin with the right-most column, even if it does not start at the top.
* Transcribe text in other languages (e.g. English, Japanese kana) as printed, without translating.
* Output only the transcription, with no comments or explanations.
* Only if no character is legible (e.g. the crop is blank or too blurry), output exactly one line: [OCR failed: <short reason in English>]

Double-check your response so that it fulfills all constraints.
"""

READING_ORDER_PROMPT = """
You are an expert in historic Chinese newspaper layouts.
Determine the reading order of the text in this image.

Answer with exactly one of these lines:
vertical, right-to-left
horizontal, right-to-left
horizontal, left-to-right

* vertical: the characters run top to bottom in columns, the columns are read from right to left.
* horizontal: the characters run in one or more rows, e.g. headlines. 
Use right-to-left unless the row clearly reads left to right (e.g. Latin letters or Arabic numbers).
* If the image shows only one character, answer: horizontal, right-to-left

Output only the answer line, without any explanation.
"""

REQUEST_SEMAPHORE_LIMIT = 100


class VLMOCRProcessor(Processor):
    """OCR-D processor for block-level VLM OCR."""

    def setup(self) -> None:
        self.client = openai.AsyncOpenAI(
            api_key=self.parameter["apikey"],
            base_url=self.parameter["server"],
        )
        self.model = self.parameter["model"]
        self.min_edge = self.parameter["min_edge"]
        self.repetition_penalty = self.parameter["repetition_penalty"]
        self.request_semaphore = BoundedSemaphore(REQUEST_SEMAPHORE_LIMIT)
        self.logger.info(
            "initialized VLM OCR client for model %s at %s with %d shared request slots",
            self.model,
            self.parameter["server"],
            REQUEST_SEMAPHORE_LIMIT,
        )

    def process_page_pcgts(
        self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None
    ) -> OcrdPageResult:
        assert input_pcgts
        assert input_pcgts[0]
        assert page_id

        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()
        self.logger.info("processing page %s with VLM block OCR", page_id)
        text_regions = page.get_AllRegions(classes=["Text"], order="reading-order")
        if not text_regions:
            self.logger.info("page %s has no text regions", page_id)
            return result

        # Read the original image instead of an AlternativeImage: the segmentation
        # steps store layout visualizations as AlternativeImage, and OCR-D would
        # use the most recent one, so the VLM would get crops with the region
        # outlines drawn over the text.
        page_image, page_coords, _ = self.workspace.image_from_page(
            page, page_id, feature_filter=self._alternative_image_features(page)
        )
        self.logger.debug(
            "page %s image prepared: size=%dx%d features=%s",
            page_id,
            page_image.width,
            page_image.height,
            page_coords.get("features", ""),
        )
        region_images = self._extract_text_region_images(
            text_regions, page_image, page_coords
        )
        self.logger.info(
            "page %s: submitting %d text blocks for VLM OCR",
            page_id,
            len(region_images),
        )
        region_texts = asyncio.run(self._ocr_region_images(region_images))

        for (region, _), text in zip(region_images, region_texts):
            region.set_TextEquiv([TextEquivType(Unicode=text)])

        self.logger.info(
            "page %s: finished OCR for %d text blocks", page_id, len(region_texts)
        )
        return result

    @staticmethod
    def _alternative_image_features(page) -> str:
        """Features (@comments) of all AlternativeImages of the page, to filter out.

        Features that OCR-D can apply to the original image itself (cropped,
        deskewed, rotated-*) are left out, otherwise OCR-D would skip them for
        the original image as well.
        """
        auto_features = {
            "cropped",
            "deskewed",
            "rotated-90",
            "rotated-180",
            "rotated-270",
        }
        features = set()
        for alternative_image in page.get_AlternativeImage():
            features.update((alternative_image.get_comments() or "").split(","))
        features -= auto_features
        features.discard("")
        return ",".join(sorted(features))

    def _extract_text_region_images(
        self, text_regions, page_image: Image.Image, page_coords: dict
    ) -> List[Tuple[object, Image.Image]]:
        region_images = []
        for region in text_regions:
            region_image, _ = self.workspace.image_from_segment(
                region, page_image, page_coords, fill="white"
            )
            self.logger.debug(
                "text region %s: extracted crop size=%dx%d",
                region.id,
                region_image.width,
                region_image.height,
            )
            region_images.append((region, region_image))

        return region_images

    async def _ocr_region_images(
        self, region_images: Sequence[Tuple[object, Image.Image]]
    ) -> List[str]:
        tasks = [
            asyncio.create_task(
                self._ocr_region_image(region.id or "<no-id>", region_image)
            )
            for region, region_image in region_images
        ]
        region_texts = await asyncio.gather(*tasks)
        self.logger.debug("completed all VLM OCR tasks")
        return region_texts

    async def _ocr_region_image(self, region_id: str, region_image: Image.Image) -> str:
        # The VLM cannot read crops with a very short edge (vLLM even rejects them
        # with status 400), so don't send them. The marker keeps them countable
        # for the analysis of failed regions.
        if min(region_image.size) < self.min_edge:
            self.logger.info(
                "text region %s (%dx%d): shorter edge below min_edge %d, skipping OCR",
                region_id,
                region_image.width,
                region_image.height,
                self.min_edge,
            )
            return (
                f"[OCR skipped: one edge too small "
                f"({region_image.width}x{region_image.height} px)]"
            )
        self.logger.debug(
            "text region %s: encoding crop for VLM request (%dx%d)",
            region_id,
            region_image.width,
            region_image.height,
        )
        image_data_url = self._image_data_url(region_image)
        # Handle failed requests per region: otherwise one bad region fails the
        # whole page, and openai's exceptions cannot even be passed back from the
        # parallel page workers, which hides the actual error message.
        try:
            reading_order = await self._vllm_request(
                READING_ORDER_PROMPT, image_data_url, "reading-order", region_id
            )
            self.logger.debug(
                "text region %s: detected reading order %r", region_id, reading_order
            )
            text = await self._vllm_request(
                OCR_PROMPT_TEMPLATE.format(reading_order=reading_order),
                image_data_url,
                "ocr",
                region_id,
            )
        except openai.APIStatusError as e:
            self.logger.error(
                "text region %s (%dx%d): VLM request failed with status %d: %s",
                region_id,
                region_image.width,
                region_image.height,
                e.status_code,
                e.message,
            )
            return f"[OCR failed: request error {e.status_code}]"
        except openai.APIError as e:
            # e.g. the server is not reachable: stop instead of marking every
            # region as failed, with an exception that the page workers can pass on
            raise RuntimeError(
                f"VLM request for region {region_id} failed: {e}"
            ) from None
        self.logger.debug(
            "text region %s: OCR returned %d characters", region_id, len(text)
        )
        return text

    async def _vllm_request(
        self, prompt: str, image_data_url: str, request_kind: str, region_id: str
    ) -> str:
        await self._acquire_request_slot()
        try:
            self.logger.debug(
                "text region %s: sending %s request to VLM", region_id, request_kind
            )
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            },
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.0,
                extra_body={
                    "repetition_penalty": self.repetition_penalty,
                    "top_k": 0,
                    "top_p": 1.0,
                },
            )
            self.logger.debug(
                "text region %s: completed %s request", region_id, request_kind
            )
            return response.choices[0].message.content
        finally:
            self.request_semaphore.release()

    async def _acquire_request_slot(self) -> None:
        waited = False
        while not self.request_semaphore.acquire(block=False):
            if not waited:
                self.logger.debug("waiting for shared VLM request slot")
                waited = True
            await asyncio.sleep(0.01)
        if waited:
            self.logger.debug("acquired shared VLM request slot")

    def _image_data_url(self, image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(VLMOCRProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
