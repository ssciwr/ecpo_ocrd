# use paddleocr to refine Eynollah's inference results
# based on the implementation of https://github.com/dokempf/ecpo-new-pipeline

from ocrd import Processor, OcrdPage, OcrdPageResult, OcrdPageResultImage
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from typing import Optional
import click
from shapely.geometry import Polygon


class ECPOInferenceProcessor(Processor):
    """OCR-D Processor for ECPO inference, using PaddleOCR to refine Eynollah's inference results"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        # TODO
        pass

    def shutdown(self) -> None:
        # TODO
        pass

    def process_page_pcgts(
        self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None
    ) -> OcrdPageResult:
        """Override process_page_pcgts of the Processor class to perform
        ECPO inference on the input page(s) and return the result as an OcrdPageResult.
        """
        assert input_pcgts
        assert input_pcgts[0]
        assert self.parameter  # default values or from CLI with -p or -P

        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()

        text_regions = page.get_TextRegion()

        # keep only text regions
        filtered_text_regions = [r for r in text_regions if r.get_type() == "text"]

        # convert coordinates of text regions into polygon format
        polygons = []
        for region in filtered_text_regions:
            points = region.get_Coords().get_points()
            polygon = Polygon(points)
            polygons.append(polygon)

        polygons = list(reversed(sorted(polygons, key=lambda p: p.area)))

        # TODO: move util functions to this repo

        return result


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(ECPOInferenceProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
