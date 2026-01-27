from ocrd import Processor, Workspace, OcrdPage, OcrdPageResult, OcrdPageResultImage
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_utils import points_from_polygon
from ocrd_models.ocrd_page import BorderType, CoordsType, AlternativeImageType
from typing import Optional

import numpy as np
import numpy.typing as npt
from ecpo_ocrd import utils
from pathlib import Path
from typing import List, Tuple, Callable
from paddleocr import TextDetection
from PIL import Image, ImageDraw
import ruptures as rpt
import click
from copy import deepcopy


# ----------------------------
# Step 1: use PaddleOCR to detect text
# ----------------------------
def get_text_detections_paddleocr(
    img: Path | np.ndarray, ocr_model: str, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Get text detections using PaddleOCR.

    Note: using PaddleOCR TextDetection predict directly with file path yields better
    Dynamic Programming results than using with a pre-loaded image (as numpy array).

    Args:
        img (Path | np.ndarray): Path to the input image or a pre-loaded image as a numpy array.
        ocr_model (str): PaddleOCR model name.
        device (str): Device to run the model on. e.g. "cpu", "gpu", "gpu:0".

    Returns:
        Tuple[np.ndarray, np.ndarray]: The input image and detected text polygons.
    """
    model = TextDetection(model_name=ocr_model, device=device)
    det_result = model.predict(img, batch_size=1)

    first_result = det_result[0]  # only one image
    in_img = first_result.get("input_img")
    det_polys = first_result.get("dt_polys")

    return in_img, det_polys


# ----------------------------
# Step 2: compute signal (column-wise projection) with text areas masked
# ----------------------------
def compute_signal(
    img: np.ndarray,
    dt_polys: np.ndarray,
    proj_func: Callable[[npt.ArrayLike], float] = np.mean,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a mask from text detection polygons and compute column-wise projection signal.

    Args:
        img (np.ndarray): Input image in RGB format (H, W, 3).
        dt_polys (np.ndarray): Detected text polygons, shape (N, 4, 2).
        proj_func (Callable): Function to compute projection, e.g. np.sum or np.mean.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The computed signal and the mask array.
    """
    # create a mask with background black (0)
    h, w = img.shape[:2]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # draw text areas as white (255)
    for poly in dt_polys:
        points = [tuple(point) for point in poly]
        draw.polygon(points, fill=255)

    # compute signal
    mask_array = np.array(mask)
    signal = proj_func(mask_array, axis=0)  # column-wise

    return signal, mask_array


# ----------------------------
# Step 3: find split points using Dynamic Programming
# ----------------------------
def find_split_points(
    signal: np.ndarray,
    num_bkps: int = 4,
    close_thres: float = 1e-3,
    num_segments: int = 2,
    fallback: bool = True,
) -> Tuple[List[int], bool, List[int]]:
    """Find breakpoints in the signal using Dynamic Programming,
    and return points within the resulting segments where the signal is close to zero.

    Args:
        signal (np.ndarray): 1D array representing the signal.
        num_bkps (int): Number of breakpoints to find with DP.
        close_thres (float): Threshold to consider a point as "close to zero".
        num_segments (int): Number of segments (pages) to split the image into.
        fallback (bool): Whether to fallback to center split if no breakpoints found.

    Returns:
        Tuple[List[int], bool, List[int]]: List of refined breakpoints,
            a flag indicating if fallback was used, and the original breakpoints from DP.
    """
    # use ruptures to find breakpoints
    algo = rpt.Dynp(model="l2").fit(signal)
    bkps = algo.predict(n_bkps=num_bkps)
    bkps = bkps[:-1]  # remove the last point which is length of signal

    # find near-zero points within segments
    near_zero_points = []
    for start, stop in zip(bkps[:-1], bkps[1:]):
        segment = signal[start:stop]
        near_zero_mask = np.isclose(segment, 0.0) | (segment <= close_thres)
        # got index where mask is True
        near_zero_ps = np.where(near_zero_mask)[0] + start
        near_zero_points.extend(near_zero_ps.tolist())

    # group near-zero points into groups of continuous indices
    groups = []
    current_group = []
    for i in range(len(near_zero_points)):
        if i == 0:
            current_group.append(near_zero_points[i])
        else:
            if near_zero_points[i] == near_zero_points[i - 1] + 1:
                current_group.append(near_zero_points[i])
            else:
                groups.append(current_group)
                current_group = [near_zero_points[i]]
    if current_group:
        groups.append(current_group)

    # record only the edges of each group as refined breakpoints
    center = signal.shape[0] / 2
    refined_bkps = []
    for group in groups:
        refined_bkps.append(group[0])
        refined_bkps.append(group[-1])

    assert len(refined_bkps) % 2 == 0, "Refined breakpoints should be in pairs."

    # get only breakpoints near the center to make sure we have num_segments - 1 splits
    near_center_bkps = sorted(
        refined_bkps,
        key=lambda x: abs(x - center),
    )[: num_segments - 1]
    near_center_bkps = sorted(near_center_bkps)

    use_fallback = False
    if not near_center_bkps and fallback:
        # fallback to center split
        w = signal.shape[0]
        near_center_bkps = [w // 2]
        use_fallback = True

    return near_center_bkps, use_fallback, bkps


# ----------------------------
# Step 4 & 5: Slice & save, no-OCR-D version
# ----------------------------
def no_ocrd_slice_and_save_to_files(
    img: np.ndarray,
    splits_internal: List[int],
    output_dir: Path,
    fname: str,
    unique_tag: str,
    segment_size: int = 300,
    jpeg_quality: int = 95,
):
    """Slice the image at the given split columns and save segments.
    Segments are saved as <fname>_pX.jpg where X is the segment index starting from 0.

    This function is not tied to OCR-D workspace, it's a standalone function.

    Args:
        img (np.ndarray): Input image in RGB format (H, W, 3).
        splits_internal (List[int]): List of internal split columns (x coordinates).
        output_dir (Path): Directory to save the segments.
        fname (str): filename of the image, used for naming output segments.
        unique_tag (str): unique tag to append to output image filenames.
        segment_size (int): Minimum size in pixels of segment to consider for splits.
        jpeg_quality (int): JPEG quality for saving images.
    """
    w = img.shape[1]
    cuts = [0] + splits_internal + [w]
    saved = []
    current_cut = cuts[0]
    i = 0
    for next_cut in cuts[1:]:
        if next_cut - current_cut <= segment_size:
            continue  # too narrow, skip

        # create an image of same size but mask other areas except the segment
        seg = img[:, current_cut:next_cut]
        out_img = np.full_like(img, 255)
        out_img[:, current_cut:next_cut] = seg

        # save the image with only the segment visible
        outname = f"{fname}_p{i}_{unique_tag}.jpg"
        outpath = output_dir / outname
        utils.save_jpeg(outpath, out_img, quality=jpeg_quality)
        saved.append((outpath, current_cut, next_cut))

        # update for next
        current_cut = next_cut
        i += 1

    return saved


# ----------------------------
# Step 4 & 5: Slice & save, OCR-D version
# ----------------------------
def slice_and_save_ocrd(
    img: np.ndarray,
    splits_internal: List[int],
    pcgts: OcrdPage,
    workspace: Workspace,
    page_id: Optional[str],
    segment_size: int = 300,
) -> OcrdPageResult:
    """Slice the image at the given split columns and save segments
    into the OCR-D workspace. Each segment is saved as a new file in the output file
    group, and a corresponding Page object is created for each segment.

    Args:
        img (np.ndarray): Input image in RGB format (H, W, 3).
        splits_internal (List[int]): List of internal split columns (x coordinates).
        pcgts (OcrdPage): The original OCR-D page object.
        workspace (Workspace): The OCR-D workspace to save new files.
        page_id (Optional[str]): The ID of the original page.
        segment_size (int): Minimum size in pixels of segment to consider for splits.

    Returns:
        OcrdPageResult: Result containing new OCR-D page objects for each segment.
    """
    h, w = img.shape[:2]
    cuts = [0] + splits_internal + [w]
    border_polygons = []
    current_cut = cuts[0]
    for next_cut in cuts[1:]:
        if next_cut - current_cut <= segment_size:
            continue  # too narrow, skip

        # create border polygon for the segment
        border_polygon = [
            (current_cut, 0),
            (next_cut, 0),
            (next_cut, h),
            (current_cut, h),
        ]
        border_polygons.append(border_polygon)

        # update for next
        current_cut = next_cut

    # create a wrapper for the output pages
    # refer to: https://github.com/OCR-D/ocrd_anybaseocr/pull/115
    results = OcrdPageResult(
        pcgts, *[deepcopy(pcgts) for _ in range(len(border_polygons) - 1)]
    )
    for i, (result, border_polygon) in enumerate(zip(results, border_polygons)):
        # set page border
        border = BorderType(
            Coords=CoordsType(points=points_from_polygon(border_polygon))
        )
        result.pcgts.Page.set_Border(border)

        # let OCR-D crop the image to the border
        cropped_image, cropped_coords, _ = workspace.image_from_page(
            result.pcgts.Page, page_id, fill="background", transparency=True
        )

        # record the coordinate transformation as AlternativeImage for downstream tools
        alt_image = AlternativeImageType(
            comments=cropped_coords["features"],
        )
        result.pcgts.Page.add_AlternativeImage(alt_image)

        # attach the cropped image to the result
        result.images.append(
            OcrdPageResultImage(cropped_image, f".IMG-SPLIT-{i}", alt_image)
        )

    return results


# use above functions for an ocrd processor class
class SplitPagesProcessor(Processor):
    """OCR-D Processor to split pages into multiple segments based on gutter detection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_count = 0
        self.fallback_page_ids = []

    def process_page_pcgts(
        self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None
    ) -> OcrdPageResult:
        """Override process_page_pcgts of Processor to split pages into multiple groups.

        For each physical page image, the processor will:

            1) use PaddleOCR to find text detection on the image
            2) compute signal (column-wise projection) to find vertical split points
                2.1) based on the text detection result to mask text areas, masked areas are white, background is black
                2.2) compute column-wise projection on the masked image, i.e. mean of pixel values per column (signal)
            3) find vertical split points based on the signal
                3.1) use Dynamic Programming to find vertical breakpoints of significant gaps
                3.2) find refined points between those breakpoints that their signal near zero (black), i.e. no text there
                3.3) only consider points near the center to ensure we have num_segments - 1 splits
            4) split into vertical segments at those refined points; always covers full width
            5) save segments into corresponding output file groups
        """
        assert input_pcgts
        assert input_pcgts[0]
        assert self.parameter  # default values or from CLI with -p or -P

        pcgts = input_pcgts[0]
        page = pcgts.get_Page()

        page_image, page_coords, page_info = self.workspace.image_from_page(
            page,
            page_id,
            feature_selector="deskewed",  # use deskewed image only
        )

        # # step 0: get path of the current page image
        # img_rel_path = page.get_imageFilename()
        # if not img_rel_path:
        #     raise RuntimeError(f"No ImageFilename found for page {page_id}.")
        # img_path = Path(self.workspace.directory) / img_rel_path
        # self.logger.info(f"Splitting page {page_id} with image {img_path}")

        # step 1: use PaddleOCR to detect text
        paddleocr_model = self.parameter.get("paddleocr_model", "PP-OCRv5_server_det")
        device = self.parameter.get("device", "cpu")
        in_img, dt_polys = get_text_detections_paddleocr(
            np.array(page_image.convert("RGB")),  # PaddleOCR needs RGB images
            ocr_model=paddleocr_model,
            device=device,
        )

        # step 2: compute signal
        proj_func_name = self.parameter.get("proj_func", "mean")
        proj_func = getattr(np, proj_func_name)
        signal, mask_array = compute_signal(in_img, dt_polys, proj_func=proj_func)

        # step 3: find split points
        number_breakpoints = self.parameter.get("number_breakpoints", 4)
        close_threshold = self.parameter.get("close_threshold", 3.0)
        fallback_to_center = self.parameter.get("fallback_to_center", True)
        num_segments = self.parameter.get("num_segments", 2)
        points, fallback, org_bkps = find_split_points(
            signal,
            num_bkps=number_breakpoints,
            close_thres=close_threshold,
            num_segments=num_segments,
            fallback=fallback_to_center,
        )

        if fallback:
            self.fallback_count += 1
            if page_id:
                self.fallback_page_ids.append(page_id)
                self.logger.info(f"Fallback to center for page {page_id}")

        # step 4 & 5: slice & save
        segment_size = self.parameter.get("segment_size", 400)
        results = slice_and_save_ocrd(
            in_img,
            points,
            pcgts,
            self.workspace,
            page_id,
            segment_size=segment_size,
        )

        return results


# ----------------------------
# CLI for OCR-D pipeline
# ----------------------------
@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(SplitPagesProcessor, *args, **kwargs)
