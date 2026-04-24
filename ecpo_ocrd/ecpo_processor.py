from typing import Optional
import click
from shapely.geometry import Polygon, MultiPolygon, LineString
from PIL import Image
import numpy as np
import os

from ocrd import Processor, OcrdPage, OcrdPageResult, OcrdPageResultImage
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_models.ocrd_page import (
    AlternativeImageType,
    TextRegionType,
    ImageRegionType,
    LineDrawingRegionType,
    SeparatorRegionType,
    CoordsType,
)
from ocrd_utils import points_from_polygon

from ecpo_ocrd.refine import (
    LayoutDetector,
    crop_polygon,
    translate,
    overlay_outline,
    flatten_polys,
)


region_mapping = {
    # label: (subtype, region type, region label)
    "artificial_boundary": (None, LineDrawingRegionType, "LineDrawingRegion"),
    "text": ("paragraph", TextRegionType, "TextRegion"),
    "image": (None, ImageRegionType, "ImageRegion"),
    "heading": ("heading", TextRegionType, "TextRegion"),
    "separator": (None, SeparatorRegionType, "SeparatorRegion"),
}

hole_suffix = "_hole"  # hole suffix in PAGE XML


class ECPOInferenceProcessor(Processor):
    """OCR-D Processor for ECPO inference, using PaddleOCR to refine Eynollah's inference results"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        """Override setup of the Processor class to initialize the layout detector."""
        self.layout_detector = LayoutDetector()

        # labels to be refined
        labels = self.parameter.get("labels", [])
        if labels and "all" in labels:
            self.labels = set(region_mapping.keys())
        else:
            self.labels = set(labels)

    def shutdown(self) -> None:
        # TODO: check if we need to do something else
        if hasattr(self, "layout_detector"):
            del self.layout_detector
        if hasattr(self, "labels"):
            del self.labels

    def _build_hierarchy_regions(
        self,
        regions: list[
            TextRegionType
            | ImageRegionType
            | LineDrawingRegionType
            | SeparatorRegionType
        ],
    ):
        """Build a hierarchy of regions based on their IDs and coordinates.
        Regions with '_hole' suffix in their ID are considered hole regions of their parent (same index).
        """
        region_hier = {}
        for region in regions:
            region_id = region.id if region.id else ""
            region_idx = (
                int(region_id.split("_")[1]) if "_" in region_id else 0
            )  # e.g. region_xxx_text_hole, xxx starts from 1
            is_hole = region_id.endswith(hole_suffix)

            points = region.get_Coords().get_points()
            # parse "x,y x,y ..." into [(x, y), (x, y), ...]
            coords = [tuple(map(int, p.split(","))) for p in points.split()]

            if region_idx not in region_hier:
                region_hier[region_idx] = {
                    "shell": None,  # main region polygon
                    "holes": set(),  # set of hole coords
                }

            if is_hole:
                region_hier[region_idx]["holes"].add(tuple(coords))
            else:
                region_hier[region_idx]["shell"] = coords

        return region_hier

    def _convert_ocrd_regions_to_polygons(
        self,
        regions: list[
            TextRegionType
            | ImageRegionType
            | LineDrawingRegionType
            | SeparatorRegionType
        ],
    ) -> list[Polygon]:
        """Convert OCR-D regions to Shapely Polygons for further processing."""
        region_heir = self._build_hierarchy_regions(regions)
        polygons = []
        for idx, region in region_heir.items():
            shell = region["shell"]
            holes = region["holes"]
            if shell:
                polygon = Polygon(shell, holes)
                polygons.append(polygon)
        return polygons

    def _refine_regions(
        self,
        org_img_arr: np.ndarray,
        regions: list[
            TextRegionType
            | ImageRegionType
            | LineDrawingRegionType
            | SeparatorRegionType
        ],
    ) -> list[Polygon]:
        """Refine the input regions using the layout detector
        and return the refined regions as a list of Polygons.

        Here regions with '_hole' suffix in their id are hole regions in their parent (same index).
        These regions will be masked out before feeding into the layout detector.

        Args:
            org_img_arr (np.ndarray): The original image array of the page.
            regions (list): A list of text/image/line drawing/separator regions to be refined

        Returns:
            list[Polygon]: A list of refined regions in polygon format.
        """
        # convert coordinates of regions into polygon format
        polygons = self._convert_ocrd_regions_to_polygons(regions)

        polygons = list(reversed(sorted(polygons, key=lambda p: p.area)))

        refined_regions = []
        for polygon in polygons:
            if polygon.area < 20:
                continue

            crop, mask, (xoff, yoff) = crop_polygon(org_img_arr, polygon)
            res = self.layout_detector.layout_detection(crop)

            for r in res:
                poly = polygon.intersection(translate(r, xoff=xoff, yoff=yoff))
                if poly.area > 20:
                    refined_regions.append(poly)

        return refined_regions

    def process_page_pcgts(
        self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None
    ) -> OcrdPageResult:
        """Override process_page_pcgts of the Processor class to perform
        ECPO inference on the input page(s) and return the result as an OcrdPageResult.
        """
        assert input_pcgts
        assert input_pcgts[0]

        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()

        # get the original image of the page
        img_filename = page.imageFilename
        img_filepath = os.path.join(self.workspace.directory, img_filename)
        image = Image.open(img_filepath)
        img_arr = np.array(image)

        overlayed_polys = {}
        for label in self.labels:
            if label not in region_mapping:
                self.log(f"Label '{label}' is not recognized. Skipping.")
                continue

            subtype, region_type_cls, region_type_name = region_mapping[label]
            selected_regions = getattr(page, f"get_{region_type_name}")()
            # text and heading have the same class type, filter if needed
            if subtype:
                selected_regions = [
                    r for r in selected_regions if r.get_type() == subtype
                ]

            refined_polys = self._refine_regions(img_arr, selected_regions)

            if refined_polys:
                overlayed_polys[label] = refined_polys

                # update PAGE XML with the refined regions
                getattr(page, f"set_{region_type_name}")(
                    []
                )  # clear existing regions of this type
                for i, poly in enumerate(refined_polys):
                    poly_iter = flatten_polys(poly)

                    for p in poly_iter:
                        points = points_from_polygon(p.exterior.coords)
                        new_region = region_type_cls(
                            id=f"refined_{label}_region_{i+1}",
                            Coords=CoordsType(points=points),
                        )
                        getattr(page, f"add_{region_type_name}")(new_region)

                        for h in p.interiors:
                            hole_points = points_from_polygon(h.coords)
                            hole_region = region_type_cls(
                                id=f"refined_{label}_region_{i+1}_hole",
                                Coords=CoordsType(points=hole_points),
                            )
                            getattr(page, f"add_{region_type_name}")(hole_region)

        overlayed_img = overlay_outline(Image.fromarray(img_arr), overlayed_polys)

        # record alternative image with overlayed text regions
        alt_img = AlternativeImageType(
            comments="Refine Eynollah layout detection with PaddleOCR",
        )
        page.add_AlternativeImage(alt_img)
        result.images.append(OcrdPageResultImage(overlayed_img, "refined", alt_img))

        return result


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(ECPOInferenceProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
