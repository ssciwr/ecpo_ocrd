from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from shapely.geometry import Polygon

from ocrd_utils import coordinates_of_segment


HOLE_SUFFIX = "_hole"


@dataclass(frozen=True)
class RegionPolygon:
    region: Any
    polygon: Polygon


def is_hole_region(region: Any) -> bool:
    return bool(region.id and region.id.endswith(HOLE_SUFFIX))


def region_topology_key(region: Any) -> tuple[type, Optional[str], int] | str:
    region_id = region.id or ""
    if is_hole_region(region):
        region_id = region_id[: -len(HOLE_SUFFIX)]

    parts = region_id.split("_")
    if len(parts) > 1 and parts[1].isdigit():
        return type(region), region_type(region), int(parts[1])
    return region_id


def region_type(region: Any) -> Optional[str]:
    if hasattr(region, "get_type"):
        return region.get_type()
    return getattr(region, "type_", None)


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


def build_region_hierarchy(
    regions: Iterable[Any],
    page_image: Optional[Any] = None,
    page_coords: Optional[dict] = None,
) -> dict[tuple[type, Optional[str], int] | str, dict[str, Any]]:
    region_hierarchy: dict[tuple[type, Optional[str], int] | str, dict[str, Any]] = {}

    for region in regions:
        key = region_topology_key(region)
        entry = region_hierarchy.setdefault(
            key,
            {
                "region": None,
                "shell": None,
                "holes": [],
            },
        )
        coords = region_coordinates(region, page_image, page_coords)

        if is_hole_region(region):
            entry["holes"].append(coords)
        else:
            entry["region"] = region
            entry["shell"] = coords

    return region_hierarchy


def ocrd_regions_to_region_polygons(
    regions: Iterable[Any],
    page_image: Optional[Any] = None,
    page_coords: Optional[dict] = None,
) -> list[RegionPolygon]:
    region_polygons = []
    for entry in build_region_hierarchy(regions, page_image, page_coords).values():
        if entry["shell"]:
            polygon = Polygon(entry["shell"], entry["holes"])
            region_polygons.append(RegionPolygon(entry["region"], polygon))
    return region_polygons


def ocrd_regions_to_polygons(
    regions: Iterable[Any],
    page_image: Optional[Any] = None,
    page_coords: Optional[dict] = None,
) -> list[Polygon]:
    return [
        region_polygon.polygon
        for region_polygon in ocrd_regions_to_region_polygons(
            regions, page_image, page_coords
        )
    ]
