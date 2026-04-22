from typing import Optional
import click
from shapely.geometry import Polygon
from PIL import Image
import numpy as np
import os

from ocrd import Processor, OcrdPage, OcrdPageResult, OcrdPageResultImage
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_models.ocrd_page import AlternativeImageType

from ecpo_ocrd.refine import LayoutDetector, crop_polygon, translate, overlay_outline


class ECPOInferenceProcessor(Processor):
    """OCR-D Processor for ECPO inference, using PaddleOCR to refine Eynollah's inference results"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        """Override setup of the Processor class to initialize the layout detector."""
        self.layout_detector = LayoutDetector()

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

        # get the original image of the page
        img_filename = page.imageFilename
        img_filepath = os.path.join(self.workspace.directory, img_filename)
        image = Image.open(img_filepath)
        img_arr = np.array(image)

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

        result_text_regions = []
        for polygon in polygons:
            if polygon.area < 20:
                continue

            crop, mask, (xoff, yoff) = crop_polygon(img_arr, polygon)
            res = self.layout_detector.layout_detection(crop)

            for r in res:
                poly = polygon.intersection(translate(r, xoff=xoff, yoff=yoff))
                if poly.area > 20:
                    result_text_regions.append(poly)

        overlayed_img = overlay_outline(
            Image.fromarray(img_arr), {"text_polys": result_text_regions}
        )

        # record alternative image with overlayed text regions
        alt_img = AlternativeImageType(
            comments="Refine Eynollah layout detection with PaddleOCR",
        )
        page.add_AlternativeImage(alt_img)
        result.images.append(
            OcrdPageResultImage(np.array(overlayed_img), "refined", alt_img)
        )

        return result


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(ECPOInferenceProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
