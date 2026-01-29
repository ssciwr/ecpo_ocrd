import pytest
from ecpo_ocrd import split_pages as sp
from PIL import Image, ImageDraw, ImageFont
import numpy as np


@pytest.fixture
def get_img():
    # Image size and gap configuration
    img_w, img_h = 1000, 800
    gap = 300

    # Create blank white image
    img = Image.new("RGB", (img_w, img_h), color="white")
    draw = ImageDraw.Draw(img)

    # Text content
    text_left = (
        "This is the LEFT text block.\n"
        "It is used to test OCR detection.\n"
        "There will be a blank space in the center."
    )

    text_right = (
        "This is the RIGHT text block.\n"
        "It mirrors the layout of the left side.\n"
        "Both sides leave a 300 px gap."
    )

    # Load font (optional)
    font = ImageFont.load_default()

    # Measure each text block
    bbox_left = draw.multiline_textbbox((0, 0), text_left, font=font, align="left")
    bbox_right = draw.multiline_textbbox((0, 0), text_right, font=font, align="left")

    text_w_left = bbox_left[2] - bbox_left[0]
    text_h_left = bbox_left[3] - bbox_left[1]
    text_w_right = bbox_right[2] - bbox_right[0]
    text_h_right = bbox_right[3] - bbox_right[1]

    # Compute positions
    total_text_width = text_w_left + gap + text_w_right
    start_x = (img_w - total_text_width) / 2  # horizontally center the pair
    y = (img_h - max(text_h_left, text_h_right)) / 2  # vertical center

    # Draw left text
    draw.multiline_text((start_x, y), text_left, fill="black", font=font, align="left")

    # Draw right text
    draw.multiline_text(
        (start_x + text_w_left + gap, y),
        text_right,
        fill="black",
        font=font,
        align="left",
    )

    return img


@pytest.fixture
def get_img_path(get_img, tmp_path):
    img_path = tmp_path / "test_image.png"
    get_img.save(img_path, format="PNG")
    return img_path


def test_get_text_detections_paddleocr(get_img_path):
    ocr_model = "PP-OCRv5_server_det"
    in_img, det_polys = sp.get_text_detections_paddleocr(
        str(get_img_path), ocr_model, device="cpu"
    )
    assert in_img.shape == (800, 1000, 3)
    assert len(det_polys) > 0


def test_compute_signal(get_img_path):
    in_img, det_polys = sp.get_text_detections_paddleocr(
        str(get_img_path), ocr_model="PP-OCRv5_server_det", device="cpu"
    )
    signal, mask_array = sp.compute_signal(in_img, det_polys, proj_func=np.mean)
    assert len(signal) == in_img.shape[1]  # signal length should match image width
    assert all(0 <= val <= 255 for val in signal)  # signal values should be in [0, 255]
    assert mask_array.shape == (in_img.shape[0], in_img.shape[1])


def test_find_split_points(get_img_path):
    in_img, det_polys = sp.get_text_detections_paddleocr(
        str(get_img_path), ocr_model="PP-OCRv5_server_det", device="cpu"
    )
    signal, _ = sp.compute_signal(in_img, det_polys, proj_func=np.mean)
    split_points, fallback, org_bkps = sp.find_split_points(
        signal,
        num_bkps=4,
        close_thres=0.0,
        fallback=True,
    )
    assert len(split_points) == 2
    assert fallback is False
    assert len(org_bkps) == 4


@pytest.fixture()
def get_polygons():
    polygons = [
        [(0, 0), (10, 0), (10, 50), (0, 50)],
        [(10, 0), (30, 0), (30, 50), (10, 50)],
        [(30, 0), (35, 0), (35, 50), (30, 50)],
        [(35, 0), (70, 0), (70, 50), (35, 50)],
    ]
    return polygons


def test_merge_polygons_special_cases():
    with pytest.raises(ValueError):
        sp._merge_polygons([], -1)

    refined_polys = sp._merge_polygons([], 0)
    assert refined_polys == []


def test_merge_polygons(get_polygons):

    # resulting segments are fewer than expected
    expected_polygon_num = len(get_polygons) + 1

    refined_polys = sp._merge_polygons(get_polygons, expected_polygon_num)
    assert refined_polys == get_polygons

    # resulting segments match expected
    expected_polygon_num = len(get_polygons)
    refined_polys = sp._merge_polygons(get_polygons, expected_polygon_num)
    assert refined_polys == get_polygons

    # resulting segments are more than expected
    expected_polygon_num = len(get_polygons) - 1
    refined_polys = sp._merge_polygons(get_polygons, expected_polygon_num)
    assert len(refined_polys) == expected_polygon_num
    assert refined_polys[0] == get_polygons[0]
    assert refined_polys[1] == [
        get_polygons[1][0],
        get_polygons[2][1],
        get_polygons[2][2],
        get_polygons[1][3],
    ]
    assert refined_polys[2] == get_polygons[3]

    expected_polygon_num = len(get_polygons) - 2
    refined_polys = sp._merge_polygons(get_polygons, expected_polygon_num)
    assert len(refined_polys) == expected_polygon_num
    assert refined_polys[0] == [
        get_polygons[0][0],
        get_polygons[2][1],
        get_polygons[2][2],
        get_polygons[0][3],
    ]
    assert refined_polys[1] == get_polygons[3]
    # all merged into one
    expected_polygon_num = 1
    refined_polys = sp._merge_polygons(get_polygons, expected_polygon_num)
    assert len(refined_polys) == expected_polygon_num
    assert refined_polys[0] == [
        get_polygons[0][0],
        get_polygons[3][1],
        get_polygons[3][2],
        get_polygons[0][3],
    ]


def test_refine_border_polygons_no_middle_two_sides(get_polygons):
    # Test case where resulting segments are fewer than expected
    expected_num_segments = len(get_polygons) + 1
    refined_polys, fewer_flag = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=35.0
    )
    assert refined_polys == get_polygons
    assert fewer_flag is True

    # Test case where resulting segments match expected
    expected_num_segments = len(get_polygons)
    refined_polys, fewer_flag = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=35.0
    )
    assert refined_polys == get_polygons
    assert fewer_flag is False

    # Test case where resulting segments are more than expected
    expected_num_segments = len(get_polygons) - 1
    refined_polys, fewer_flag = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=35.0
    )
    assert len(refined_polys) == expected_num_segments
    assert refined_polys[0] == get_polygons[0]
    assert refined_polys[1] == [
        get_polygons[1][0],
        get_polygons[2][1],
        get_polygons[2][2],
        get_polygons[1][3],
    ]
    assert refined_polys[2] == get_polygons[3]
    assert fewer_flag is False

    expected_num_segments = len(get_polygons) - 2
    refined_polys, fewer_flag = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=35.0
    )
    assert len(refined_polys) == expected_num_segments
    assert refined_polys[0] == [
        get_polygons[0][0],
        get_polygons[2][1],
        get_polygons[2][2],
        get_polygons[0][3],
    ]
    assert refined_polys[1] == get_polygons[3]
    assert fewer_flag is False

    # all merged into one
    expected_num_segments = 1
    refined_polys, fewer_flag = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=35.0
    )
    assert len(refined_polys) == expected_num_segments
    assert refined_polys[0] == [
        get_polygons[0][0],
        get_polygons[3][1],
        get_polygons[3][2],
        get_polygons[0][3],
    ]
    assert fewer_flag is False


def test_refine_border_polygons_with_middle(get_polygons):
    # only check cases where resulting segments are more than expected
    center_x = 32.0  # goes through the 3rd polygon

    expected_num_segments = len(get_polygons) - 1
    refined_polys, fewer_flag = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=center_x
    )
    assert len(refined_polys) == expected_num_segments
    assert refined_polys[0] == [
        get_polygons[0][0],
        get_polygons[1][1],
        get_polygons[1][2],
        get_polygons[0][3],
    ]
    assert refined_polys[1] == get_polygons[2]
    assert refined_polys[2] == get_polygons[3]
    assert fewer_flag is False

    expected_num_segments = len(get_polygons) - 2
    refined_polys, _ = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=center_x
    )
    assert len(refined_polys) == expected_num_segments
    assert refined_polys[0] == [
        get_polygons[0][0],
        get_polygons[2][1],
        get_polygons[2][2],
        get_polygons[0][3],
    ]
    assert refined_polys[1] == get_polygons[3]

    # all merged into one
    expected_num_segments = 1
    refined_polys, _ = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=center_x
    )
    assert len(refined_polys) == expected_num_segments
    assert refined_polys[0] == [
        get_polygons[0][0],
        get_polygons[3][1],
        get_polygons[3][2],
        get_polygons[0][3],
    ]


def test_refine_border_polygons_one_side_cases(get_polygons):
    # these are rare cases where polygons are only on one side of the center
    center_x = 0.0  # all polygons are on the right side
    expected_num_segments = len(get_polygons) - 2
    refined_polys, _ = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=center_x
    )
    assert len(refined_polys) == expected_num_segments
    assert refined_polys[0] == [
        get_polygons[0][0],
        get_polygons[2][1],
        get_polygons[2][2],
        get_polygons[0][3],
    ]
    assert refined_polys[1] == get_polygons[3]

    center_x = 75.0  # all polygons are on the left side
    expected_num_segments = len(get_polygons) - 1
    refined_polys, _ = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=center_x
    )
    assert len(refined_polys) == expected_num_segments
    assert refined_polys[0] == get_polygons[0]
    assert refined_polys[1] == [
        get_polygons[1][0],
        get_polygons[2][1],
        get_polygons[2][2],
        get_polygons[1][3],
    ]
    assert refined_polys[2] == get_polygons[3]

    center_x = 5.0  # through the first polygon
    expected_num_segments = len(get_polygons) - 1
    refined_polys, _ = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=center_x
    )
    assert len(refined_polys) == expected_num_segments
    assert refined_polys[0] == get_polygons[0]
    assert refined_polys[1] == [
        get_polygons[1][0],
        get_polygons[2][1],
        get_polygons[2][2],
        get_polygons[1][3],
    ]
    assert refined_polys[2] == get_polygons[3]

    center_x = 68.0  # through the last polygon
    expected_num_segments = len(get_polygons) - 2
    refined_polys, _ = sp.refine_border_polygons(
        get_polygons, expected_num_segments, center_x=center_x
    )
    assert len(refined_polys) == expected_num_segments
    assert refined_polys[0] == [
        get_polygons[0][0],
        get_polygons[2][1],
        get_polygons[2][2],
        get_polygons[0][3],
    ]
    assert refined_polys[1] == get_polygons[3]
