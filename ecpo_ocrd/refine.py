# use paddleocr to refine Eynollah's inference results
# based on the implementation of https://github.com/dokempf/ecpo-new-pipeline

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from PIL import Image, ImageDraw
import numpy as np
import math
from skimage.filters import threshold_otsu
import networkx as nx
from shapely.ops import unary_union, polygonize
from shapely.affinity import translate
import itertools
from typing import Iterable, Any, Callable

import paddleocr

import _cover_heuristic


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


def overlap_threshold_function(
    polys: list[Polygon], threshold: float = 0.9
) -> Callable[[int, int], bool]:
    """Overlap percentage thresholding for two polygons.

    Values are always in [0, 1] with 1 for identical polygons.
    """

    def _func(i, j):
        return (
            polys[i].intersection(polys[j]).area / polys[i].union(polys[j]).area
            > threshold
        )

    return _func


def filter_redundant_polys(
    polys: list[Polygon], criterion: Callable[[int, int], bool]
) -> list[Polygon]:
    """Filter polygons that are almost identical with others.

    Args:
        polys (list[Polygon]): list of polygons to filter
        criterion (Callable[[int, int], bool]): function that takes two indices i,
            j and returns True if polys[i] and polys[j] are considered redundant
            and should be merged, False otherwise

    Returns:
        list[Polygon]: list of polygons after merging redundant ones
    """
    # use networkx UnionFind to build a graph implicitly
    uf = nx.utils.UnionFind(polys)

    for i, p in enumerate(polys):
        for j, q in enumerate(polys):
            if i < j:
                if criterion(i, j):
                    uf.union(
                        p, q
                    )  # find connected components with criterion as edge condition

    # merge polygons in the same connected component by unary_union
    return [unary_union(list(s)) for s in uf.to_sets()]


def calculate_atomics(polys: list[Polygon]) -> tuple[list[Polygon], list[list[int]]]:
    """Given a number of polygons, calculate the set of composing atomics.

    A set of atomics for a set of polygons is defined such that every
    polygon is the disjoint union of a subset of the atomics. Allows
    for additive calculations without resorting to geometry calculations
    every time.

    Args:
        polys (list[Polygon]): list of polygons to calculate atomics for

    Returns:
        tuple[list[Polygon], list[list[int]]]: a tuple of (atomics, poly_atomics) where:
            - atomics is a list of Polygons that are the atomic components of the input polygons
            - poly_atomics is a list of lists,
                where poly_atomics[i] is the list of indices of atomics that compose polys[i]
    """
    # Create all atomics as polygons
    boundaries = [p.boundary for p in polys if not p.is_empty]
    merged = unary_union(boundaries)
    atomics = list(polygonize(merged))

    # For each atomic, find out which polygons it belongs to
    atomic_covers = []
    for cell in atomics:
        pt = cell.representative_point()
        covered = [i for i, p in enumerate(polys) if p.covers(pt)]
        atomic_covers.append(covered)

    # Invert that mapping: Which atomics compose each polygon
    poly_atomics = [[] for p in polys]
    for i, ac in enumerate(atomic_covers):
        for p in ac:
            poly_atomics[p].append(i)

    return atomics, poly_atomics


def black_overlap_function(
    binary: np.ndarray, polys: list[Polygon], threshold=0.98
) -> Callable[[int, int], bool]:
    """Overlap percentage of the black pixels of two polygons.

    Values are always in [0, 1] with 1 for all black content in the overlap.
    This would essentially mean that we can pick any polygon without losing
    anything.

    Args:
        binary (np.ndarray): binarized image of shape (H, W)
        polys (list[Polygon]): list of polygons to calculate black overlaps for
        threshold (float): threshold for considering two polygons redundant
            based on black pixel overlap

    Returns:
        Callable[[int, int], bool]: function that takes two indices i, j and returns
            True if the black pixel overlap of polys[i] and polys[j] is above the threshold,
            False otherwise.
    """

    atomics, poly_atomics = calculate_atomics(polys)
    atomics_values = [black_content(binary, a) for a in atomics]

    def _func(i, j):
        seti = set(poly_atomics[i])
        setj = set(poly_atomics[j])
        intersection = seti.intersection(setj)
        union = seti.union(setj)

        return (
            sum((atomics_values[i] for i in intersection), 0)
            / sum((atomics_values[i] for i in union), 0)
            > threshold
        )

    return _func


def exact_disjoint_criterion(p: Polygon, q: Polygon) -> bool:
    """True if two polygons are disjoint"""
    return p.intersection(q).area == 0


def fuzzy_disjoint_criterion(
    threshold: float = 0.95,
) -> Callable[[Polygon, Polygon], bool]:
    """Two polygons are considered disjoint if their intersection
    is smaller than a certain percentage of the smaller polygon."""

    def _func(p, q):
        return p.intersection(q).area < threshold * min(p.area, q.area)

    return _func


def disjoint_groups(items: Iterable[Any], is_disjoint) -> list[set[Any]]:
    """Find groups of items that are not disjoint from each other.

    Args:
        items (Iterable[Any]): Iterable of items to group
        is_disjoint (Callable[[Any, Any], bool]): function that takes two items,
            a and b, and returns True if they are disjoint (i.e. should NOT be in the same group),
            False otherwise

    Returns:
        list[set[Any]]: list of sets, each set contains items that are NOT disjoint from each other
    """
    uf = nx.utils.UnionFind(items)

    # Merge pairs that are NOT disjoint
    for a, b in itertools.combinations(items, 2):
        if not is_disjoint(a, b):
            uf.union(a, b)

    return list(uf.to_sets())


def intersection_edges(polys: list[Polygon]) -> list[tuple[int, int]]:
    """Calculate the intersection graph for a set of polygons."""

    edges = []
    for i, p in enumerate(polys):
        for j, q in enumerate(polys):
            if i != j:
                if not exact_disjoint_criterion(p, q):
                    # if not disjoint_criterion(p, q):
                    edges.append((i, j))

    return edges


def squaricity(poly: Polygon) -> float:
    """A measure for how square-like a polygon is.

    Values are always in [0, 1] with 1 is an actual square.
    """
    return 16 * poly.area / (poly.length * poly.length)


def average_squaricity_criterion(polys: list[Polygon]) -> Callable[[list[int]], float]:
    """Sorting criterion for average squaricity for a set of polygons."""

    def _average_squaricity_criterion(indices):
        return sum((squaricity(polys[i]) for i in indices), 0.0) / len(indices)

    return _average_squaricity_criterion


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
        """Implementation of the layout detection algorithm.
        Goal: Find a set of polygons that cover the text boxes with
        minimal redundancy and good squaricity.

        Algorithm outline:
        1. Run the PaddleOCR layout detection to get initial text boxes.
        2. Convert text boxes to polygons
            and filter out those with too little black content or too overlapping ones.
        3. If there are disjoint groups of polygons,
            apply the algorithm recursively to each group (divide and conquer).
        4. At this point, all polygons are connected. If there are too many polygons (> 20),
            increase the threshold and restart the algorithm to get fewer polygons.
        5. Calculate the atomic polygons and their black content,
            and run a brute-force algorithm to find the optimal cover
            of the text boxes with a certain threshold.
        6. Among multiple optimal covers, select the one with the highest
            average squaricity of the resulting polygons.
        7. Return the union of polygons in each group as the final detected text polygons.

        Args:
            img (np.ndarray): input image. In this project, the image is cropped
                to the bounding box of the original Eynollah polygon, but it can be any image.
            text_threshold (float): threshold for filtering text boxes based on their confidence score.
                Higher threshold means fewer boxes and less redundancy,
                    but also higher risk of missing text.
                This is a parameter for the PaddleOCR layout detection,
                    and we can increase it if we find too many polygons in step 4.

        Returns:
            list[Polygon]: list of detected text polygons in the image.
        """
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
                for cpoly in self.impl_layout_detection(cimg):
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
            return self.impl_layout_detection(img, text_threshold=text_threshold + 0.01)

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
        """Entry point for full layout detection.

        Args:
            img (np.ndarray): input image. In this project, the image is cropped
                to the bounding box of the original Eynollah polygon, but it can be any image.

        Returns:
            list[Polygon]: list of detected text polygons in the image.
        """

        # Binarize once in the beginning.
        binarized = self.otsu_binarization(img)

        # Dispatch to an impl function, as this function might be called recursively
        # with additional parameters.
        return self.impl_layout_detection(binarized)


def flatten_polys(poly):
    if poly.is_empty:
        return []

    if isinstance(poly, Polygon):
        return [poly]

    if isinstance(poly, MultiPolygon):
        return list(poly.geoms)

    if isinstance(poly, GeometryCollection):
        result = []
        for g in poly.geoms:
            result.extend(flatten_polys(g))
        return result

    return []  # ignore other geometry types


def overlay_outline(
    image: Image.Image, result: dict[str, list[Polygon]]
) -> Image.Image:
    """ "Overlay the detected polygons on the original image for visualization.

    Args:
        image (Image.Image): original image to overlay on
        result (dict[str, list[Polygon]]): dictionary containing the detected polygons,
            with possible keys:
                "artificial_boundary", "text", "image", "heading", and "separator"

    Returns:
        Image.Image: image with overlaid polygons.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGBA")

    def _draw_polygons(polys, line_color, width=4, fill_color=None):
        for poly in polys:
            poly_iter = flatten_polys(poly)

            for p in poly_iter:
                # create a grayscale mask for the polygon
                mask = Image.new("L", image.size, 0)
                mask_draw = ImageDraw.Draw(mask)

                # draw outer polygon
                mask_draw.polygon(
                    p.exterior.coords,
                    fill=255,  # visible area
                )

                # cut out holes
                for hole in p.interiors:
                    mask_draw.polygon(
                        hole.coords,
                        fill=0,  # invisible area
                    )

                # apply the fill color using mask
                if fill_color is not None:
                    r, g, b, a = fill_color
                    base_layer = Image.new("RGBA", image.size, (r, g, b, 0))
                    alpha_mask = mask.point(lambda p: int(p * (a / 255)))
                    base_layer.putalpha(alpha_mask)
                    image.alpha_composite(base_layer)

                # draw the border on top of the fill
                border_draw = ImageDraw.Draw(image, "RGBA")

                border_draw.polygon(
                    p.exterior.coords,
                    outline=line_color,
                    width=width,
                    fill=None,  # no fill for border, only outline
                )

                for hole in p.interiors:
                    border_draw.polygon(
                        hole.coords,
                        outline=line_color,
                        width=width,
                        fill=None,  # no fill for border, only outline
                    )

    # fill color with alpha for better visualization of overlaps
    fill_color_artificial_boundary = (0, 204, 0, 77)
    border_color_artificial_boundary = (0, 204, 0, 255)
    fill_color_text = (231, 76, 60, 77)
    border_color_text = (231, 76, 60, 255)
    fill_color_image = (52, 152, 219, 77)
    border_color_image = (52, 152, 219, 255)
    fill_color_heading = (230, 126, 34, 77)
    border_color_heading = (230, 126, 34, 255)
    fill_color_separator = (155, 89, 182, 77)
    border_color_separator = (155, 89, 182, 255)

    # draw in priority order
    # according to order from LabelStudio annotation
    # from low to highter hierarchy level:
    # artificial_boundary -> text -> image -> heading -> separator

    # artificial boundary polygons (green)
    if "artificial_boundary" in result:
        _draw_polygons(
            result["artificial_boundary"],
            border_color_artificial_boundary,
            width=4,
            fill_color=fill_color_artificial_boundary,
        )
    # text polygons (red)
    if "text" in result:
        _draw_polygons(
            result["text"], border_color_text, width=4, fill_color=fill_color_text
        )
    # image polygons (blue)
    if "image" in result:
        _draw_polygons(
            result["image"],
            border_color_image,
            width=4,
            fill_color=fill_color_image,
        )
    # heading polygons (yellow)
    if "heading" in result:
        _draw_polygons(
            result["heading"],
            border_color_heading,
            width=4,
            fill_color=fill_color_heading,
        )
    # separator polygons (purple)
    if "separator" in result:
        _draw_polygons(
            result["separator"],
            border_color_separator,
            width=4,
            fill_color=fill_color_separator,
        )

    return image
