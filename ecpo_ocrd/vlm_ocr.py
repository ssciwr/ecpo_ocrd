import asyncio
import base64
from io import BytesIO
from multiprocessing import BoundedSemaphore
from typing import List, Optional, Sequence

import openai
from ocrd import OcrdPage, OcrdPageResult, Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_models.ocrd_page import TextEquivType

import click


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


class VLMOCRProcessor(Processor):
    """OCR-D processor for block-level VLM OCR."""

    def setup(self) -> None:
        self.client = openai.AsyncOpenAI(
            api_key=self.parameter["apikey"],
            base_url=self.parameter["server"],
        )
        self.model = self.parameter["model"]
        self.request_semaphore = BoundedSemaphore(100)

    def process_page_pcgts(
        self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None
    ) -> OcrdPageResult:
        assert input_pcgts
        assert input_pcgts[0]
        assert page_id

        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()
        text_regions = page.get_TextRegion()
        if not text_regions:
            return result

        page_image, page_coords, _ = self.workspace.image_from_page(page, page_id)
        region_images = [
            self.workspace.image_from_segment(region, page_image, page_coords)[0]
            for region in text_regions
        ]
        region_texts = asyncio.run(self._ocr_region_images(region_images))

        for region, text in zip(text_regions, region_texts):
            region.set_TextEquiv([TextEquivType(Unicode=text)])

        return result

    async def _ocr_region_images(self, region_images: Sequence) -> List[str]:
        tasks = [
            asyncio.create_task(self._ocr_region_image(region_image))
            for region_image in region_images
        ]
        return await asyncio.gather(*tasks)

    async def _ocr_region_image(self, region_image) -> str:
        image_data_url = self._image_data_url(region_image)
        reading_order = await self._vllm_request(READING_ORDER_PROMPT, image_data_url)
        return await self._vllm_request(
            OCR_PROMPT_TEMPLATE.format(reading_order=reading_order),
            image_data_url,
        )

    async def _vllm_request(self, prompt: str, image_data_url: str) -> str:
        await asyncio.to_thread(self.request_semaphore.acquire)
        try:
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
            return response.choices[0].message.content
        finally:
            self.request_semaphore.release()

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
