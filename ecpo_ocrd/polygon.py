import math

import numpy as np
from PIL import Image, ImageDraw
from shapely import make_valid, set_precision
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


def flatten_polygon_geometry(geometry) -> list[Polygon]:
    """Return all polygon components from a Shapely geometry."""
    if geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):
        result = []
        for geom in geometry.geoms:
            result.extend(flatten_polygon_geometry(geom))
        return result

    return []


def _integer_ring_points(coords) -> list[tuple[int, int]]:
    points = []
    for coord in coords:
        x, y = coord[:2]
        point = (int(round(x)), int(round(y)))
        if not points or points[-1] != point:
            points.append(point)

    if len(points) > 1 and points[0] == points[-1]:
        points.pop()

    return points


def page_points_from_coords(coords) -> str:
    """Convert Shapely coordinates to PAGE integer points without closure."""
    return " ".join("%i,%i" % point for point in _integer_ring_points(coords))


def _polygon_from_integer_rings(poly: Polygon, min_area: float) -> Polygon | None:
    shell = _integer_ring_points(poly.exterior.coords)
    if len(set(shell)) < 3:
        return None

    holes = []
    for interior in poly.interiors:
        hole = _integer_ring_points(interior.coords)
        if len(set(hole)) < 3:
            continue
        hole_poly = Polygon(hole)
        if not hole_poly.is_valid or hole_poly.area < min_area:
            continue
        holes.append(hole)

    return Polygon(shell, holes)


def polygons_for_pagexml(geometry, min_area: float = 20.0) -> list[Polygon]:
    """Normalize a geometry for safe PAGE XML serialization.

    PAGE coordinates are integer-valued, so near-collinear Shapely coordinates can
    collapse or self-intersect when serialized. Quantize to the PAGE grid, repair
    topology, and drop degenerate components before writing regions.
    """
    result = []

    for poly in flatten_polygon_geometry(make_valid(geometry)):
        quantized = set_precision(poly, 1.0)
        repaired = make_valid(quantized)
        repaired = set_precision(repaired, 1.0)

        for candidate in flatten_polygon_geometry(make_valid(repaired)):
            candidate = _polygon_from_integer_rings(candidate, min_area)
            if candidate is None:
                continue

            for final in flatten_polygon_geometry(make_valid(candidate)):
                final = _polygon_from_integer_rings(final, min_area)
                if final is None:
                    continue

                if final.area < min_area:
                    continue
                minx, miny, maxx, maxy = final.bounds
                if math.ceil(maxx) <= math.floor(minx):
                    continue
                if math.ceil(maxy) <= math.floor(miny):
                    continue
                if final.is_valid:
                    result.append(final)

    return result


def rasterize_polygon_to_mask(
    image_shape: tuple[int, int], polygon: Polygon | MultiPolygon
) -> np.ndarray:
    """Rasterize a Shapely polygon to a boolean mask."""
    H, W = image_shape[0], image_shape[1]
    if isinstance(polygon, Polygon):
        polygons = [polygon]
    elif isinstance(polygon, MultiPolygon):
        polygons = list(polygon.geoms)
    else:
        raise TypeError("polygon must be shapely Polygon or MultiPolygon")

    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)

    for poly in polygons:
        exterior_coords = [
            (int(round(x)), int(round(y))) for x, y in poly.exterior.coords
        ]
        draw.polygon(exterior_coords, outline=255, fill=255)

        for interior in poly.interiors:
            interior_coords = [
                (int(round(x)), int(round(y))) for x, y in interior.coords
            ]
            draw.polygon(interior_coords, outline=0, fill=0)

    mask = np.array(mask_img, dtype=np.uint8)
    return mask != 0


def crop_polygon(
    image: np.ndarray, polygon: Polygon
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Crop an image to a polygon bbox and whiten pixels outside the polygon."""
    H, W = image.shape[0], image.shape[1]
    mask = rasterize_polygon_to_mask((H, W), polygon)

    minx, miny, maxx, maxy = polygon.bounds
    minx = max(int(math.floor(minx)), 0)
    miny = max(int(math.floor(miny)), 0)
    maxx = min(int(math.ceil(maxx)), W)
    maxy = min(int(math.ceil(maxy)), H)

    mask_cropped = mask[miny:maxy, minx:maxx]

    if image.ndim == 3:
        cropped_img = image[miny:maxy, minx:maxx].copy()
        mask_3c = np.repeat(
            mask_cropped[:, :, np.newaxis], cropped_img.shape[2], axis=2
        )
        cropped_img[~mask_3c] = 255
    else:
        cropped_img = image[miny:maxy, minx:maxx].copy()
        cropped_img[~mask_cropped] = 255

    return cropped_img, mask_cropped, (minx, miny)
