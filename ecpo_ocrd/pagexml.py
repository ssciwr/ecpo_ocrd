from __future__ import annotations

from typing import Any, Iterable, Optional

from shapely import make_valid
from shapely.geometry import Polygon

from ocrd_utils import coordinates_of_segment

from ecpo_ocrd.polygon import flatten_polygon_geometry


def region_coordinates(
    region: Any,
    page_image: Optional[Any] = None,
    page_coords: Optional[dict] = None,
) -> list[tuple[int, int]]:
    if page_image is not None and page_coords is not None:
        return [
            tuple(map(int, point))
            for point in coordinates_of_segment(region, page_image, page_coords)
        ]

    points = region.get_Coords().get_points()
    return [tuple(map(int, point.split(","))) for point in points.split()]


def ocrd_regions_to_polygons(
    regions: Iterable[Any],
    page_image: Optional[Any] = None,
    page_coords: Optional[dict] = None,
) -> list[Polygon]:
    polygons = []
    for region in regions:
        coords = region_coordinates(region, page_image, page_coords)
        if len(set(coords)) < 3:
            continue
        polygons.extend(flatten_polygon_geometry(make_valid(Polygon(coords))))
    return polygons
