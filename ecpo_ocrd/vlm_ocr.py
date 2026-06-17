import asyncio
import base64
from io import BytesIO
from multiprocessing import BoundedSemaphore
from typing import Dict, List, Optional, Sequence, Tuple

import openai
from ocrd import OcrdPage, OcrdPageResult, Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_models.ocrd_page import TextEquivType
from ocrd_utils import bbox_from_polygon, coordinates_of_segment, crop_image

import click
from PIL import Image, ImageDraw, ImageStat


OCR_PROMPT_TEMPLATE = """
Your task is to OCR this image written in traditional chinese.
The reading order is {reading_order}.

Your result needs to fulfill **all** of these constraints:
* Give the result **exactly** as it appears on the image
* Keep line breaks from the original
* Do not modify to modern chinese, keep exactly as is.
* Denote numbers exactly like in the image, not in english writing.
* The result needs to follow the {reading_order} reading order.
* If the text does not start in the top-right corner, start it in the right-most column.

Double-check your response so that it fulfills all constraints.
"""


READING_ORDER_PROMPT = """
You are an expert in classical Chinese paleography and historic East Asian page layouts.
Analyze the image and determine the correct reading order of the text.

Tasks:
* Identify the text line orientation: "vertical" or "horizontal"
* Identify column reading direction: "right-to-left" or "left-to-right" or "top-to-bottom"

Output format (no explanations, no reasoning):

<orientation>,  <direction>

Only produce the structured fields. No commentary or explanation.
"""


HOLE_SUFFIX = "_hole"
REQUEST_SEMAPHORE_LIMIT = 100


class VLMOCRProcessor(Processor):
    """OCR-D processor for block-level VLM OCR."""

    def setup(self) -> None:
        self.client = openai.AsyncOpenAI(
            api_key=self.parameter["apikey"],
            base_url=self.parameter["server"],
        )
        self.model = self.parameter["model"]
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

        page_image, page_coords, _ = self.workspace.image_from_page(page, page_id)
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

    def _extract_text_region_images(
        self, text_regions, page_image: Image.Image, page_coords: dict
    ) -> List[Tuple[object, Image.Image]]:
        holes_by_region_id = self._collect_holes(text_regions)
        num_holes = sum(len(holes) for holes in holes_by_region_id.values())
        text_regions = [region for region in text_regions if not self._is_hole(region)]
        self.logger.debug(
            "extracted %d text shell regions and %d text hole regions",
            len(text_regions),
            num_holes,
        )

        return [
            (
                region,
                self._image_from_text_region(
                    region,
                    holes_by_region_id.get(region.id, []),
                    page_image,
                    page_coords,
                ),
            )
            for region in text_regions
        ]

    def _collect_holes(self, text_regions) -> Dict[str, list]:
        holes_by_region_id = {}
        for region in text_regions:
            if self._is_hole(region):
                self.logger.debug(
                    "attaching hole region %s to %s",
                    region.id,
                    region.id[: -len(HOLE_SUFFIX)],
                )
                holes_by_region_id.setdefault(
                    region.id[: -len(HOLE_SUFFIX)], []
                ).append(region)
        return holes_by_region_id

    def _is_hole(self, region) -> bool:
        return bool(region.id and region.id.endswith(HOLE_SUFFIX))

    def _image_from_text_region(
        self, region, holes, page_image: Image.Image, page_coords: dict
    ) -> Image.Image:
        polygon = coordinates_of_segment(region, page_image, page_coords)
        bbox = bbox_from_polygon(polygon)
        hole_polygons = [
            coordinates_of_segment(hole, page_image, page_coords) for hole in holes
        ]
        self.logger.debug(
            "text region %s: bbox=%s holes=%d",
            region.id,
            bbox,
            len(hole_polygons),
        )
        masked_image = self._image_from_polygon_with_holes(
            page_image, polygon, hole_polygons
        )
        cropped_image = crop_image(masked_image, box=bbox)
        self.logger.debug(
            "text region %s: crop size=%dx%d",
            region.id,
            cropped_image.width,
            cropped_image.height,
        )
        return cropped_image

    def _image_from_polygon_with_holes(
        self, image: Image.Image, polygon, hole_polygons
    ) -> Image.Image:
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        self._draw_polygon(draw, polygon, fill=255)
        for hole_polygon in hole_polygons:
            self._draw_polygon(draw, hole_polygon, fill=0)

        background = Image.new(image.mode, image.size, self._background_color(image))
        return Image.composite(image, background, mask)

    def _draw_polygon(self, draw: ImageDraw.ImageDraw, polygon, fill: int) -> None:
        draw.polygon([tuple(point) for point in polygon], fill=fill)

    def _background_color(self, image: Image.Image):
        median = ImageStat.Stat(image).median
        if len(median) > 1:
            return tuple(median)
        return median[0]

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
        self.logger.debug(
            "text region %s: encoding crop for VLM request (%dx%d)",
            region_id,
            region_image.width,
            region_image.height,
        )
        image_data_url = self._image_data_url(region_image)
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
                    "repetition_penalty": 1.0,
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
