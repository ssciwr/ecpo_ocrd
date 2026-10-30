"""Cut out the regions where VLM OCR failed, to inspect them.
Code suggestioned by Claude and checked by human.

Usage:
    python ecpo_ocrd/crop_and_analyze.py <xml_folder> <output_folder>
        [--min-size 10]
        [--white-threshold 0.95]
        [--black-threshold 0.95]
        [--jobs 4]
        [--image-folder <path to sds_ecpo>]
        [--min-repeat-times 50]
        [--show-unfailed-regions false]

For each PAGE-XML file in <xml_folder>, the page image is downloaded from the
ECPO IIIF server into <output_folder>/<image name>/,
* every text region whose text starts with "[OCR failed" is saved there as <region id>.jpg.
This includes both the failures reported by the VLM and the failed requests
("[OCR failed: request error <status code>]").
* Regions whose text got stuck repeating a few characters (until another character is encountered
or the maximum output length is reached) are cut out as well,
if one unit repeats itself at least --min-repeat-times times in a row.

With --image-folder, the original images are read from that folder instead and
nothing is downloaded. Crops that already exist are not cut again.

The XML files are processed in parallel.

Afterwards, <output_folder>/failed_regions_<settings>.csv lists every failed region, with
flags:
* whether one of its edges is too small or
* it is almost white/black and
* the status code of failed requests or
* the repeated unit and how many times it repeats in a row, and
* the OCR text

<output_folder>/failed_regions_stats_<settings>.png shows the statistics.
<settings> holds the options the results depend on, e.g.
"minsize10_white095_black095_repeat50",
so that runs with different options don't overwrite each other.
"""

import csv
import os
import re
import urllib.parse
import urllib.request
from io import BytesIO
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from PIL import Image, ImageDraw

BASE_LINK = "https://ecpo.cats.uni-heidelberg.de/fcgi-bin/iipsrv.fcgi?IIIF=/ecpo/images/{}/full/full/0/default.jpg"
FAILED_PREFIX = "[OCR failed"
# written by vlm_ocr.py when the request to the VLM fails with an HTTP status
REQUEST_ERROR_RE = re.compile(r"\[OCR failed: request error (\d+)\]")
# a unit of up to 10 characters (including line breaks), repeated right after itself
REPEAT_RE = re.compile(r"(.{1,10}?)\1+", re.DOTALL)

# grey levels (0-255) from which a pixel counts as white or black;
# paper in the scans is ~235-255, printed ink only ~130-150
WHITE_LEVEL = 200
BLACK_LEVEL = 100

CSV_NAME = "failed_regions_{}.csv"
PLOT_NAME = "failed_regions_stats_{}.png"
CSV_FIELDS = [
    "image",
    "xml_file",
    "region_id",
    # centre of the region in % of the page, from the top/from the left, e.g. "70/50"
    "pos_top_left",
    "width",
    "height",
    "white_ratio",
    "black_ratio",
    "edge_too_small",
    "almost_white",
    "almost_black",
    # HTTP status code if the request failed, empty if the VLM reported the failure
    "request_status",
    # the unit repeated most often in a row, e.g. "，" repeated 1012 times;
    # both are only filled if the unit repeats often enough to count as
    # repetition
    "most_repeated_unit",
    "repeat_times",
    "ocr_text",
]


def settings_suffix(
    min_size: int, white_threshold: float, black_threshold: float, min_repeat_times: int
) -> str:
    """Options in a form usable in file names.

    E.g. "minsize10_white095_black095_repeat50".
    """
    white = f"{white_threshold:.2f}".replace(".", "")
    black = f"{black_threshold:.2f}".replace(".", "")
    return f"minsize{min_size}_white{white}_black{black}_repeat{min_repeat_times}"


def parse_points(points: str) -> list[tuple[int, int]]:
    """Parse PAGE-XML points "x1,y1 x2,y2 ..." into a list of (x, y) tuples."""
    return [tuple(int(float(v)) for v in p.split(",")) for p in points.split()]


def polygon_mask(points: list[tuple[int, int]]) -> tuple[tuple, Image.Image]:
    """Return the bounding box of the polygon and its mask in bounding box coordinates."""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    # +1 so that the right/bottom border of the polygon is included
    box = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
    mask = Image.new("L", (box[2] - box[0], box[3] - box[1]), 0)
    shifted = [(x - box[0], y - box[1]) for x, y in points]
    ImageDraw.Draw(mask).polygon(shifted, fill=255, outline=255)
    return box, mask


def crop_polygon(
    image: Image.Image, points: list[tuple[int, int]], fill="white"
) -> Image.Image:
    """Return the bounding box of the polygon, with everything outside it filled."""
    box, mask = polygon_mask(points)
    # crop before converting, converting the whole page for every region is slow
    region = image.crop(box).convert("RGB")
    background = Image.new("RGB", region.size, fill)
    return Image.composite(region, background, mask)


def relative_position(box: tuple, page_width: int, page_height: int) -> str:
    """Centre of the box in % of the page, as "<from top>/<from left>", e.g. "70/50"."""
    top = round(100 * (box[1] + box[3]) / 2 / page_height)
    left = round(100 * (box[0] + box[2]) / 2 / page_width)
    return f"{top}/{left}"


def white_black_ratio(crop: Image.Image, mask: Image.Image) -> tuple[float, float]:
    """Share of white and black pixels inside the polygon (the fill outside is ignored)."""
    hist = crop.convert("L").histogram(mask=mask)
    total = sum(hist)
    if total == 0:
        return 1.0, 0.0
    return sum(hist[WHITE_LEVEL:]) / total, sum(hist[: BLACK_LEVEL + 1]) / total


def request_status(text: str) -> str:
    """Return the HTTP status code of a failed request, "" if the VLM reported the failure."""
    match = REQUEST_ERROR_RE.match(text)
    return match.group(1) if match else ""


def most_repeat(text: str) -> tuple[str, int]:
    """Return the unit repeated most often in a row, and how many times it occurs.

    The run ends where another unit appears or the text ends, e.g. "abc，，，，d"
    gives ("，", 4) and "一元二角\\n一元二角\\n" gives ("一元二角\\n", 2). Text
    without any unit directly repeated gives ("", 0).
    """
    unit, times = "", 0
    for match in REPEAT_RE.finditer(text):
        run_times = (match.end() - match.start()) // len(match.group(1))
        if run_times > times:
            unit, times = match.group(1), run_times
    return unit, times


def image_url(image_filename: str) -> str:
    """Build the IIIF URL from the part of imageFilename starting at "jingbao"."""
    start = image_filename.find("jingbao/")
    if start == -1:
        raise ValueError(f"'jingbao/' not found in imageFilename {image_filename}")
    # quote e.g. spaces in "1933/Old files (corrected by LH)/..."
    return BASE_LINK.format(urllib.parse.quote(image_filename[start:]))


def local_image_path(image_filename: str, image_folder: Path) -> Path:
    """Resolve imageFilename against the folder holding the original images.

    E.g. "../sds_ecpo/cats-ecpo/ecpo/jingbao/.../x.tif" with image_folder
    "/data/sds_ecpo" gives "/data/sds_ecpo/cats-ecpo/ecpo/jingbao/.../x.tif".
    """
    parts = [part for part in Path(image_filename).parts if part not in ("..", ".")]
    # the first remaining part is the image folder itself, e.g. "sds_ecpo"
    return image_folder.joinpath(*parts[1:])


def region_texts(page) -> list[tuple[str, str, str]]:
    """Return (region id, points, text) of all text regions, text is "" if missing."""
    ns = page.tag.split("}")[0] + "}"
    result = []
    for region in page.iter(f"{ns}TextRegion"):
        unicode = region.find(f"{ns}TextEquiv/{ns}Unicode")
        text = (unicode.text or "") if unicode is not None else ""
        points = region.find(f"{ns}Coords").get("points")
        result.append((region.get("id"), points, text))
    return result


def is_failed(text: str, min_repeat_times: int) -> bool:
    """Whether the OCR of a region failed.

    Besides "[OCR failed ..." this includes texts where the VLM got stuck
    repeating the same few characters at least min_repeat_times times in a row.
    """
    return text.startswith(FAILED_PREFIX) or most_repeat(text)[1] >= min_repeat_times


class PageImage:
    """Download and open the page image only when a crop actually has to be made."""

    def __init__(self, page, image_dir: Path, xml_name: str, image_folder: Path):
        self.page = page
        self.xml_name = xml_name
        self.image = None
        self.download = image_folder is None
        if self.download:
            self.path = image_dir / f"{image_dir.name}.jpg"
        else:
            self.path = local_image_path(page.get("imageFilename"), image_folder)

    def get(self) -> Image.Image:
        if self.image is None:
            if not self.download and not self.path.exists():
                raise FileNotFoundError(f"original image {self.path} does not exist")
            if self.download and not self.path.exists():
                url = image_url(self.page.get("imageFilename"))
                click.echo(f"{self.xml_name}: downloading {url}")
                # download to a temporary file first, so that an interrupted
                # download is not mistaken for a complete image on the next run
                part = self.path.with_name(self.path.name + ".part")
                urllib.request.urlretrieve(url, part)
                os.replace(part, self.path)
            self.image = Image.open(self.path)
            expected = (
                int(self.page.get("imageWidth")),
                int(self.page.get("imageHeight")),
            )
            if self.image.size != expected:
                click.echo(
                    f"{self.xml_name}: warning, image size {self.image.size} differs "
                    f"from {expected} in the PAGE-XML, crops may be misplaced"
                )
        return self.image

    def close(self) -> None:
        if self.image is not None:
            self.image.close()


def process_xml(
    xml_path: Path,
    output_dir: Path,
    min_size: int,
    white_threshold: float,
    black_threshold: float,
    min_repeat_times: int,
    image_folder: Path = None,
    show_unfailed_regions: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Return the rows of the failed regions and, if show_unfailed_regions is set,
    the flags of the unfailed regions (regions with text that did not fail)."""
    root = ET.parse(xml_path).getroot()
    ns = root.tag.split("}")[0] + "}"
    page = root.find(f"{ns}Page")
    page_width = int(page.get("imageWidth"))
    page_height = int(page.get("imageHeight"))

    # one folder per image, e.g. jb_3918_1939-08-21_0005to0008
    image_dir = output_dir / Path(page.get("imageFilename")).stem
    image_dir.mkdir(parents=True, exist_ok=True)
    page_image = PageImage(page, image_dir, xml_path.name, image_folder)

    all_regions = region_texts(page)
    regions = [r for r in all_regions if is_failed(r[2], min_repeat_times)]
    rows = []
    unfailed_rows = []
    n_cropped = 0
    try:
        for region_id, points, text in regions:
            box, mask = polygon_mask(parse_points(points))
            crop_path = image_dir / f"{region_id}.jpg"

            crop = None
            if crop_path.exists():
                crop = Image.open(crop_path)
                if crop.size != mask.size:  # e.g. region changed, cut it again
                    crop = None
            if crop is None:
                crop = crop_polygon(page_image.get(), parse_points(points))
                part = crop_path.with_name(crop_path.name + ".part")
                crop.save(part, format="JPEG")
                os.replace(part, crop_path)
                # use the saved (JPEG compressed) crop for the statistics, so that
                # they don't depend on whether the crop existed already
                crop = Image.open(crop_path)
                n_cropped += 1

            white, black = white_black_ratio(crop, mask)
            unit, repeat_times = most_repeat(text)
            if text.startswith(FAILED_PREFIX) or repeat_times < min_repeat_times:
                # not a repetition so leave both empty
                unit, repeat_times = "", ""
            rows.append(
                {
                    "image": image_dir.name,
                    "xml_file": xml_path.name,
                    "region_id": region_id,
                    "pos_top_left": relative_position(box, page_width, page_height),
                    "width": mask.width,
                    "height": mask.height,
                    "white_ratio": round(white, 3),
                    "black_ratio": round(black, 3),
                    "edge_too_small": min(mask.size) < min_size,
                    "almost_white": white >= white_threshold,
                    "almost_black": black >= black_threshold,
                    "request_status": request_status(text),
                    # escape line breaks, so that the unit stays readable in the CSV
                    "most_repeated_unit": unit.replace("\n", "\\n"),
                    "repeat_times": repeat_times,
                    "ocr_text": text,
                }
            )

        if show_unfailed_regions:
            for _, points, text in all_regions:
                if not text or is_failed(text, min_repeat_times):
                    continue
                _, mask = polygon_mask(parse_points(points))
                # no crops are saved, but compress them like the saved crops of
                # the failed regions, so that both statistics are comparable
                buffer = BytesIO()
                crop_polygon(page_image.get(), parse_points(points)).save(
                    buffer, format="JPEG"
                )
                white, black = white_black_ratio(Image.open(buffer), mask)
                unfailed_rows.append(
                    {
                        "image": image_dir.name,
                        "edge_too_small": min(mask.size) < min_size,
                        "almost_white": white >= white_threshold,
                        "almost_black": black >= black_threshold,
                    }
                )
    finally:
        page_image.close()

    click.echo(
        f"{xml_path.name}: {len(regions)} failed regions, {n_cropped} newly cropped -> {image_dir}"
    )
    return rows, unfailed_rows


def category(row: dict) -> str:
    """One category per region, for the plot; the CSV keeps the individual flags."""
    if row["edge_too_small"]:
        return "One edge too small"
    if row["almost_white"]:
        return "Almost white"
    if row["almost_black"]:
        return "Almost black"
    return "Other (has content)"


def failure_source(row: dict) -> str:
    """How the OCR failed: reported by the VLM, a failed request, or repetition."""
    if row["request_status"]:
        return f"Request error {row['request_status']}"
    if row["most_repeated_unit"]:
        return "Repetition"
    return "Reported by VLM"


def plot_stats(
    rows: list[dict],
    min_size: int,
    white_threshold: float,
    black_threshold: float,
    path: Path,
    unfailed_rows: list[dict] = None,
) -> bool:
    """Plot the statistics, return False if matplotlib is not installed.

    With unfailed_rows, a last line shows how the unfailed regions are
    flagged, i.e. which of them a filter on the flags would skip.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        click.echo("matplotlib is not installed, skipping the plot", err=True)
        return False

    surface, ink, muted, grid, baseline = (
        "#fcfcfb",
        "#0b0b0b",
        "#898781",
        "#e1e0d9",
        "#c3c2b7",
    )
    # bar charts and histograms with one series
    color = "#2a78d6"
    # categorical colors for the pies and the stacked bars, assigned in this order
    # (dark blue, orange, aqua, yellow, magenta, green, violet, red); dark blue
    # instead of the blue above, so that the pies don't look like they belong to
    # the bar charts
    series_colors = [
        "#184f95",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ]
    dark_colors = {"#184f95", "#008300"}  # need white text on top
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": ink,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.edgecolor": baseline,
        }
    )

    # first line: the two pie charts, second line: the three bar charts/histograms
    # the cross-tab has one bar per source, give it more room when there are
    # many status codes
    crosstab_height = max(1, len({failure_source(row) for row in rows}) / 6)
    layout = [
        ["category", "category", "category", "source", "source", "source"],
        ["flags", "flags", "crosstab", "crosstab", "crosstab", "crosstab"],
        ["size", "size", "size", "per_image", "per_image", "per_image"],
    ]
    height_ratios = [1, crosstab_height, 1]
    if unfailed_rows:
        # last line: the unfailed regions
        layout.append(["unfailed_category"] * 3 + ["unfailed_flags"] * 3)
        height_ratios.append(1)
    fig, axes = plt.subplot_mosaic(
        layout,
        height_ratios=height_ratios,
        figsize=(18, 5 * sum(height_ratios)),
        facecolor=surface,
    )
    for name, ax in axes.items():
        ax.set_facecolor(surface)
        if name in ("category", "source", "unfailed_category"):
            continue
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    n = len(rows)
    n_images = len({row["image"] for row in rows})

    def pie(ax, labels: list[str], values: list[int], title: str) -> None:
        """Pie chart with the counts in the legend; slices below 1% get no label."""
        total = sum(values)
        # more slices than colors (many status codes): the colors start over,
        # hatched, so that neighbouring slices never look the same
        wedges, _ = ax.pie(
            values,
            colors=[series_colors[i % len(series_colors)] for i in range(len(values))],
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": surface, "linewidth": 2},
        )
        for wedge in wedges[len(series_colors) :]:
            wedge.set_hatch("//")
        n_outside = 0
        for wedge, value in zip(wedges, values):
            if value / total < 0.01:
                continue
            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            # the label does not fit into thin slices, put it outside of the pie;
            # alternate the distance, so that labels of neighbouring thin slices
            # don't overlap
            inside = value / total >= 0.05
            if inside:
                radius = 0.7
            else:
                radius = 1.12 if n_outside % 2 == 0 else 1.28
                n_outside += 1
            ax.text(
                radius * np.cos(angle),
                radius * np.sin(angle),
                f"{100 * value / total:.0f}%",
                ha="center",
                va="center",
                color="white" if inside else ink,
                fontweight="bold",
            )
        ax.legend(
            wedges,
            [
                f"{label}: {v:,} ({100 * v / total:.1f}%)"
                for label, v in zip(labels, values)
            ],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )
        # pad: room for the labels of thin slices outside of the pie
        ax.set_title(title, loc="left", fontsize=11, pad=24)
        ax.set_aspect("equal")

    # 1) why did the regions fail? one category per region
    categories = [
        "One edge too small",
        "Almost white",
        "Almost black",
        "Other (has content)",
    ]
    counts = Counter(category(row) for row in rows)
    pie(
        axes["category"],
        categories,
        [counts[label] for label in categories],
        f"Failed regions by category, one per region in this order\n"
        f"(shorter side < {min_size} px, ≥ {white_threshold:.0%} white "
        f"or ≥ {black_threshold:.0%} black pixels)",
    )

    # 2) did the VLM report the failure, did the request fail, or did the VLM
    # get stuck repeating itself?
    counts = Counter(failure_source(row) for row in rows)
    fixed = ["Reported by VLM", "Repetition"]
    request_errors = sorted(
        (label for label in counts if label not in fixed),
        key=lambda label: (-counts[label], int(label.split()[-1])),
    )
    # one slice per status code, most frequent first
    labels = fixed + request_errors
    values = [counts[label] for label in labels]
    pie(
        axes["source"],
        labels,
        values,
        "Failed regions by source\n(VLM's own failure message, repeated "
        "characters, HTTP status code of failed requests)",
    )
    sources = labels

    def flag_bars(ax, rows: list[dict], what: str) -> None:
        """Each flag on its own, a region can have several of them."""
        flags = [
            ("One edge too small", "edge_too_small"),
            ("Almost white", "almost_white"),
            ("Almost black", "almost_black"),
        ]
        labels = [label for label, _ in flags] + ["None of these"]
        values = [sum(row[key] for row in rows) for _, key in flags]
        values.append(sum(not any(row[key] for _, key in flags) for row in rows))
        bars = ax.barh(labels[::-1], values[::-1], color=color, height=0.6)
        ax.bar_label(
            bars,
            labels=[f"{v:,} ({100 * v / len(rows):.0f}%)" for v in values[::-1]],
            padding=4,
            color=ink,
        )
        ax.set_xlim(0, max(values) * 1.35)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_title(
            f"{what} per flag\n(a region can have several flags)",
            loc="left",
            fontsize=11,
        )
        ax.set_xlabel(what)

    # 3) each flag on its own, a region can have several of them
    flag_bars(axes["flags"], rows, "Failed regions")

    # 4) do the sources fail on different kinds of regions? same groups as
    # the source pie, same colors as the category pie
    ax = axes["crosstab"]
    source_counts = Counter()
    pairs = Counter()
    for row in rows:
        source = failure_source(row)
        source_counts[source] += 1
        pairs[source, category(row)] += 1
    y_labels = [f"{source}\n({source_counts[source]:,} regions)" for source in sources]
    left = np.zeros(len(sources))
    for cat, cat_color in zip(categories, series_colors):
        shares = np.array(
            [100 * pairs[source, cat] / source_counts[source] for source in sources]
        )
        bars = ax.barh(
            y_labels[::-1],
            shares[::-1],
            left=left[::-1],
            color=cat_color,
            height=0.6,
            edgecolor=surface,
            linewidth=2,
            label=cat,
        )
        # label only segments wide enough for the text
        ax.bar_label(
            bars,
            labels=[f"{v:.0f}%" if v >= 6 else "" for v in shares[::-1]],
            label_type="center",
            color="white" if cat_color in dark_colors else ink,
        )
        left += shares
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncol=len(categories),
        frameon=False,
        borderaxespad=0.3,
    )
    ax.set_title(
        "Category of failed regions per source\n\n",
        loc="left",
        fontsize=11,
    )
    ax.set_xlabel("Share of the source's failed regions (%)")

    # 5) how big are the failed regions?
    ax = axes["size"]
    cap = 100
    sizes = [min(min(row["width"], row["height"]), cap) for row in rows]
    ax.hist(sizes, bins=range(0, cap + 5, 5), color=color, rwidth=0.85)
    ax.axvline(min_size, color=ink, linestyle="--", linewidth=1)
    ax.annotate(
        f"min size = {min_size} px",
        xy=(min_size, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -12),
        textcoords="offset points",
        color=ink,
    )
    ax.set_title("Shorter side of failed regions", loc="left", fontsize=11)
    ax.set_xlabel(f"Shorter side in px (last bin: ≥ {cap} px)")
    ax.set_ylabel("Failed regions")

    # 6) how are the failures spread over the images?
    ax = axes["per_image"]
    per_image = list(Counter(row["image"] for row in rows).values())
    ax.hist(per_image, bins=20, color=color, rwidth=0.85)
    ax.set_title("Failed regions per image", loc="left", fontsize=11)
    ax.set_xlabel("Failed regions in one image")
    ax.set_ylabel("Images")

    # 7) and 8) would a filter on the flags skip unfailed regions?
    title = f"{n:,} failed regions in {n_images:,} images"
    if unfailed_rows:
        counts = Counter(category(row) for row in unfailed_rows)
        pie(
            axes["unfailed_category"],
            categories,
            [counts[label] for label in categories],
            "Unfailed regions by category, one per region in this order\n"
            "(would be skipped by a filter on these flags)",
        )
        flag_bars(axes["unfailed_flags"], unfailed_rows, "Unfailed regions")
        title += f", {len(unfailed_rows):,} unfailed regions"

    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    # leave room at the top for the suptitle
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    # "tight" so that the legends next to the pies are not cut off
    fig.savefig(path, dpi=150, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    return True


@click.command()
@click.argument(
    "xml_folder", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_folder", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--min-size",
    default=10,
    show_default=True,
    help="A region has one edge too small if its shorter side is below this many "
    "pixels, even if its area is large",
)
@click.option(
    "--white-threshold",
    type=click.FloatRange(0, 1),
    default=0.95,
    show_default=True,
    help=f"A region is almost white if at least this share of its pixels "
    f"is white (grey level >= {WHITE_LEVEL})",
)
@click.option(
    "--black-threshold",
    type=click.FloatRange(0, 1),
    default=0.95,
    show_default=True,
    help=f"A region is almost black if at least this share of its pixels "
    f"is black (grey level <= {BLACK_LEVEL})",
)
@click.option(
    "--jobs",
    "-j",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of XML files processed in parallel. Keep it moderate when "
    "downloading, every job downloads images from the ECPO server",
)
@click.option(
    "--image-folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Folder with the original images, e.g. the path to sds_ecpo for "
    'imageFilename="../sds_ecpo/cats-ecpo/...". If given, no images are downloaded',
)
@click.option(
    "--min-repeat-times",
    type=click.IntRange(min=2),
    default=50,
    show_default=True,
    help="A text counts as repetition if a unit of up to 10 characters "
    "repeats itself at least this many times in a row, until another unit "
    "appears or the text ends",
)
@click.option(
    "--show-unfailed-regions",
    type=bool,
    default=False,
    show_default=True,
    help="true/false. If true, also show how the unfailed regions are "
    "flagged, i.e. which of them a filter on the flags would skip",
)
def cli(
    xml_folder,
    output_folder,
    min_size,
    white_threshold,
    black_threshold,
    jobs,
    image_folder,
    min_repeat_times,
    show_unfailed_regions,
):
    """Crop failed OCR regions of all PAGE-XML files in XML_FOLDER into OUTPUT_FOLDER."""
    xml_files = sorted(xml_folder.glob("*.xml"))
    if not xml_files:
        raise click.UsageError(f"No .xml files found in {xml_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    rows_per_file = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                process_xml,
                xml_path,
                output_folder,
                min_size,
                white_threshold,
                black_threshold,
                min_repeat_times,
                image_folder,
                show_unfailed_regions,
            ): xml_path
            for xml_path in xml_files
        }
        for done, future in enumerate(as_completed(futures), start=1):
            xml_path = futures[future]
            try:
                rows_per_file[xml_path] = future.result()
            except Exception as e:
                click.echo(f"{xml_path.name}: failed ({e})", err=True)
            if done % 100 == 0 or done == len(xml_files):
                click.echo(f"Processed {done}/{len(xml_files)} XML files")

    # keep the CSV in file order, independent of which job finished first
    rows = [
        row for xml_path in xml_files for row in rows_per_file.get(xml_path, ([], []))[0]
    ]
    unfailed_rows = [
        row for xml_path in xml_files for row in rows_per_file.get(xml_path, ([], []))[1]
    ]

    suffix = settings_suffix(
        min_size, white_threshold, black_threshold, min_repeat_times
    )
    csv_path = output_folder / CSV_NAME.format(suffix)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    click.echo(f"Wrote {len(rows)} failed regions to {csv_path}")
    for source, count in sorted(Counter(failure_source(row) for row in rows).items()):
        click.echo(f"  {source}: {count}")

    if show_unfailed_regions:
        click.echo(f"Found {len(unfailed_rows)} unfailed regions")
        # the plot has an extra line, don't overwrite the plot without it
        suffix += "_unfailed"
    plot_path = output_folder / PLOT_NAME.format(suffix)
    if rows and plot_stats(
        rows,
        min_size,
        white_threshold,
        black_threshold,
        plot_path,
        unfailed_rows,
    ):
        click.echo(f"Saved statistics plot to {plot_path}")


if __name__ == "__main__":
    cli()
