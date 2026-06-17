import math

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon


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
