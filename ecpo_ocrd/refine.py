# use paddleocr to refine Eynollah's inference results
# based on the implementation of https://github.com/dokempf/ecpo-new-pipeline

from shapely.geometry import Polygon, MultiPolygon
from PIL import Image, ImageDraw
import numpy as np
import math
from skimage.filters import threshold_otsu

import paddleocr


def rasterize_polygon_to_mask(
    image_shape: tuple[int, int], polygon: Polygon | MultiPolygon
) -> np.ndarray:
    """Rasterize a shapely Polygon or MultiPolygon to a boolean mask of shape (H, W).
    Holes are handled. Coordinates are rounded to integer pixel coordinates
    with a consistent floor/ceil strategy.

    Args:
        image_shape (tuple[int, int]): (H, W) of the output mask
        polygon (Polygon): shapely Polygon or MultiPolygon to rasterize

    Returns:
        np.ndarray: boolean mask of shape (H, W).
            True inside the polygon, False outside
    """
    H, W = image_shape[0], image_shape[1]
    if isinstance(polygon, Polygon):
        polygons = [polygon]
    elif isinstance(polygon, MultiPolygon):
        polygons = list(polygon.geoms)
    else:
        raise TypeError("polygon must be shapely Polygon or MultiPolygon")

    # Create mask image (L mode gives 0..255 values)
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)

    for poly in polygons:
        # Exterior (rounded to nearest integer). We use rounding to nearest pixel.
        exterior_coords = [
            (int(round(x)), int(round(y))) for x, y in poly.exterior.coords
        ]
        draw.polygon(exterior_coords, outline=255, fill=255)

        # Interiors -> holes: draw them with fill=0 to erase
        for interior in poly.interiors:
            interior_coords = [
                (int(round(x)), int(round(y))) for x, y in interior.coords
            ]
            draw.polygon(interior_coords, outline=0, fill=0)

    mask = np.array(mask_img, dtype=np.uint8)  # 0 or 255
    mask_bool = mask != 0  # boolean mask: True inside polygon
    return mask_bool


def crop_polygon(
    image: np.ndarray, polygon: Polygon
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Cut out the part of the image inside the polygon, and set the rest to white.
    The cropping box is computed from polygon.bounds and clamped to image.

    Args:
        image (np.ndarray): image of shape (H, W) or (H, W, C)
        polygon (Polygon): shapely Polygon to crop

    Returns:
        np.ndarray: cropped image of shape (H, W) or (H, W, C),
            where the part outside the polygon is set to white (255).
        np.ndarray: boolean mask of shape (H, w) for the cropped image,
            where True indicates pixels inside the polygon
        tuple[int, int]: (minx, miny) the top-left coordinate of the cropped part
            in the original image
    """
    H, W = image.shape[0], image.shape[1]
    mask = rasterize_polygon_to_mask((H, W), polygon)

    # Compute integer bbox: floor(min), ceil(max) and clamp
    minx, miny, maxx, maxy = polygon.bounds
    minx = max(int(math.floor(minx)), 0)
    miny = max(int(math.floor(miny)), 0)
    maxx = min(int(math.ceil(maxx)), W)
    maxy = min(int(math.ceil(maxy)), H)

    # Crop both image and mask
    mask_cropped = mask[miny:maxy, minx:maxx]

    # Prepare cropped image: zero outside polygon in the bbox
    if image.ndim == 3:
        cropped_img = image[miny:maxy, minx:maxx].copy()
        # Broadcast mask to channels
        mask_3c = np.repeat(
            mask_cropped[:, :, np.newaxis], cropped_img.shape[2], axis=2
        )
        cropped_img[~mask_3c] = 255
    else:
        cropped_img = image[miny:maxy, minx:maxx].copy()
        cropped_img[~mask_cropped] = 255

    return cropped_img, mask_cropped, (minx, miny)


def box_to_polygon(x0: float, y0: float, x1: float, y1: float):
    """Convert a bounding box to a Shapely Polygon."""
    return Polygon(
        [
            (x0, y0),  # top-left
            (x1, y0),  # top-right
            (x1, y1),  # bottom-right
            (x0, y1),  # bottom-left
        ]
    )


def black_content(binary: np.ndarray, poly: Polygon) -> int:
    """Count of black pixels in the polygon.

    Args:
        binary (np.ndarray): binarized image of shape (H, W)
        poly (Polygon): polygon to count black pixels in

    Returns:
        int: count of black pixels in the polygon
    """
    if poly.area == 0.0:
        return 0

    cropped_img, mask_cropped, _ = crop_polygon(binary, poly)
    return np.sum(mask_cropped & (cropped_img == 0))


class LayoutDetector:
    def _poor_mans_defaultdict(
        self, default: float, args: dict[int, float]
    ) -> dict[int, float]:
        """Workaround for PaddleOCR's inability to use collections.defaultdict
        This should be a default dict, but PaddleOCR does not use the passed
        dictionary according to its API, and therefore does not instantiate the defaults.

        Args:
            default (float): the default value for each key
            args (dict[int, float]): the explicitly set values to update the default dict with

        Returns:
            dict[int, float]: a dictionary with keys from 0 to 24, where the values are
                taken from args if present, otherwise set to default

        """
        result = {i: default for i in range(25)}
        result.update(args)
        return result

    def __init__(self):
        # Global instance of the detector. The parameters do not matter too much, as we
        # can always pass them to inference without performance penalties. It is however
        # important to have this as a singleton, as it allocates the GPU memory.
        self.detector = paddleocr.LayoutDetection(
            threshold=self._poor_mans_defaultdict(
                1.0,
                {
                    0: 0.01,
                    1: 0.25,
                    2: 0.01,
                },
            ),
            layout_merge_bboxes_mode="union",
        )

    def otsu_binarization(self, img: np.ndarray) -> np.ndarray:
        """Apply binarization to an image.
        Not refined enough for OCR, but well worth it for layout detection.
        """
        # convert to grayscale
        img = np.array(Image.fromarray(img).convert("L"))
        thr = threshold_otsu(img)
        # if pixels are above the threshold, set to 255 (white), else 0 (black)
        return ((img > thr) * 255).astype(np.uint8)

    def impl_layout_detection(self, img: np.ndarray, text_threshold: float = 0.05):
        """TODO: docstring"""
        # Run the PaddleOCR layout detection
        layout = self.detector.predict(
            img,
            threshold=self._poor_mans_defaultdict(
                1.0,
                {
                    0: text_threshold,
                    2: text_threshold,
                },
            ),
        )
        boxes = layout[0]["boxes"]

        # Separate text and image boxes and convert them to polygons
        # TODO: check if we need other classes as well,
        # e.g. figure_title, reference, doc_title, footnote, header, footer, aside_text, reference_content
        text_polys = [
            box_to_polygon(*b["coordinate"]) for b in boxes if b["cls_id"] in (0, 2)
        ]  # 0 = paragraph_title, 2 = text, see yml file of the PaddleOCR model for details

        # Drop any polygons that do not contain more than 10 black pixels
        text_polys = [p for p in text_polys if black_content(img, p) > 10]

        # Filter polygons that do not add value
        text_polys = filter_redundant_polys(
            text_polys, overlap_threshold_function(text_polys)
        )
        # The following one would be better, but is way too slow right now
        # text_polys = filter_redundant_polys(
        #     text_polys, black_overlap_function(img, text_polys)
        # )

        # This happened in practice.
        # TODO: investigate why this is even possible.
        if len(text_polys) == 0:
            return text_polys

        # Look for disjoint groups of text polygons to apply a divide and conquer approach.
        # We have a choice between making this with an exact disjoint criterion or a fuzzy
        # one. So far, I have been switching back and forth.
        # poly_groups = disjoint_groups(text_polys, fuzzy_disjoint_criterion(0.95))
        poly_groups = disjoint_groups(text_polys, exact_disjoint_criterion)

        # We found a trivial split, so we can do divide and conquer
        if len(poly_groups) > 1:
            print(
                f"Dividing into {len(poly_groups)} groups of size {', '.join([str(len(pg)) for pg in poly_groups])}"
            )
            # Crop the image according to the polygon groups
            crops = [crop_polygon(img, unary_union(list(pg))) for pg in poly_groups]

            # Recursively call this function for each group and combine the results
            results = []
            for cimg, _, (xoff, yoff) in crops:
                for cpoly in impl_layout_detection(cimg):
                    results.append(translate(cpoly, xoff=xoff, yoff=yoff))

            return results

        # If we reach this, all polygons were connected and we need to find correct
        # polygons by selecting a subset. However, our algorithm to do so is of exponential
        # complexity, we therefore can only run in up to a certain threshold around
        # 20 polygons. We "fix" this by increasing the threshold value for detection
        # until we drop below this threshold.
        if len(text_polys) > 20:
            print(
                f"Restarting algorithm (found {len(text_polys)} polygons) with threshold {text_threshold + 0.01}"
            )
            return impl_layout_detection(img, text_threshold=text_threshold + 0.01)

        # As a preparation, we build some data structures
        atomics, poly_atomics = calculate_atomics(text_polys)
        atomics_values = [black_content(img, a) for a in atomics]
        edges = intersection_edges(text_polys)

        # Run the C++ brute-force algorithm with decreasing threshold
        def _bruteforce(threshold):
            if threshold < 0.75:
                # If we reduce the threshold, so drastically, something is terribly wrong.
                # We should investigate this, but for now, I just return the union of polygons
                # as one.
                return [list(range(len(text_polys)))]

            res = _cover_heuristic.find_optimal_cover(
                threshold, edges, poly_atomics, atomics_values
            )
            if len(res) == 0:
                print(f"Restarting brute-forcing with threshold {threshold - 0.01}")
                return _bruteforce(threshold - 0.01)
            return res

        res = _bruteforce(0.98)

        # Between multiple optimal solutions, we select the most square one
        best = max(res, key=average_squaricity_criterion(text_polys))

        # Now join all polygons that are part of the same connected component
        groups = disjoint_groups(
            [text_polys[b] for b in best], exact_disjoint_criterion
        )

        return [unary_union(list(g)) for g in groups]

    def layout_detection(self, img: np.ndarray) -> list[Polygon]:
        """Entry point for full layout detection."""

        # Binarize once in the beginning.
        binarized = self.otsu_binarization(img)

        # Dispatch to an impl function, as this function might be called recursively
        # with additional parameters.
        return self.impl_layout_detection(binarized)
