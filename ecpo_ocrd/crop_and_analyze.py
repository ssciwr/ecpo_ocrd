"""Cut out the regions where VLM OCR failed, to inspect them.
Code suggestioned by Claude and checked by human.

Usage:
    python ecpo_ocrd/crop_and_analyze.py <xml_folder> <output_folder> [--min-size 10] [--white-threshold 0.95] [--black-threshold 0.95] [--jobs 4]
        [--image-folder <path to sds_ecpo>]

For each PAGE-XML file in <xml_folder>, the page image is downloaded from the
ECPO IIIF server into <output_folder>/<image name>/, and every text region whose
text starts with "[OCR failed" is saved there as <region id>.jpg. This
includes both the failures reported by the VLM and the failed requests
("[OCR failed: request error <status code>]"). With
--image-folder, the original images are read from that folder instead and
nothing is downloaded. Crops that
already exist are not cut again. The XML files are processed in parallel.

Afterwards, <output_folder>/failed_regions_<settings>.csv lists every failed region, with
flags whether one of its edges is too small or it is almost white/black and
the status code of failed requests, and
<output_folder>/failed_regions_stats_<settings>.png shows the statistics.
<settings> holds the options the flags depend on, e.g. "minsize10_white095_black095",
so that runs with different options don't overwrite each other.
"""

import csv
import os
import re
import urllib.parse
import urllib.request
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
    "ocr_text",
]


def settings_suffix(min_size: int, white_threshold: float, black_threshold: float) -> str:
    """Options in a form usable in file names, e.g. "minsize10_white095_black095"."""
    white = f"{white_threshold:.2f}".replace(".", "")
    black = f"{black_threshold:.2f}".replace(".", "")
    return f"minsize{min_size}_white{white}_black{black}"


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


def failed_regions(page) -> list[tuple[str, str, str]]:
    """Return (region id, points, text) of all text regions whose OCR failed."""
    ns = page.tag.split("}")[0] + "}"
    result = []
    for region in page.iter(f"{ns}TextRegion"):
        unicode = region.find(f"{ns}TextEquiv/{ns}Unicode")
        if unicode is not None and (unicode.text or "").startswith(FAILED_PREFIX):
            points = region.find(f"{ns}Coords").get("points")
            result.append((region.get("id"), points, unicode.text))
    return result


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
    image_folder: Path = None,
) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    ns = root.tag.split("}")[0] + "}"
    page = root.find(f"{ns}Page")
    page_width = int(page.get("imageWidth"))
    page_height = int(page.get("imageHeight"))

    # one folder per image, e.g. jb_3918_1939-08-21_0005to0008
    image_dir = output_dir / Path(page.get("imageFilename")).stem
    image_dir.mkdir(parents=True, exist_ok=True)
    page_image = PageImage(page, image_dir, xml_path.name, image_folder)

    regions = failed_regions(page)
    rows = []
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
                    "ocr_text": text,
                }
            )
    finally:
        page_image.close()

    click.echo(
        f"{xml_path.name}: {len(regions)} failed regions, {n_cropped} newly cropped -> {image_dir}"
    )
    return rows


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
    """Who reported the failure: the VLM itself, or the request with its status code."""
    if row["request_status"]:
        return f"Request error {row['request_status']}"
    return "Reported by VLM"


def plot_stats(
    rows: list[dict],
    min_size: int,
    white_threshold: float,
    black_threshold: float,
    path: Path,
) -> bool:
    """Plot the statistics, return False if matplotlib is not installed."""
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
    # (dark blue, orange, aqua, yellow, magenta, green); dark blue instead of the
    # blue above, so that the pies don't look like they belong to the bar charts
    series_colors = ["#184f95", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
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
    fig, axes = plt.subplot_mosaic(
        [
            ["category", "category", "category", "source", "source", "source"],
            ["flags", "flags", "crosstab", "crosstab", "crosstab", "crosstab"],
            ["size", "size", "size", "per_image", "per_image", "per_image"],
        ],
        figsize=(18, 15),
        facecolor=surface,
    )
    for name, ax in axes.items():
        ax.set_facecolor(surface)
        if name in ("category", "source"):
            continue
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    n = len(rows)
    n_images = len({row["image"] for row in rows})

    def pie(ax, labels: list[str], values: list[int], title: str) -> None:
        """Pie chart with the counts in the legend; slices below 1% get no label."""
        wedges, _ = ax.pie(
            values,
            colors=series_colors[: len(values)],
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": surface, "linewidth": 2},
        )
        for wedge, value in zip(wedges, values):
            if value / n < 0.01:
                continue
            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            # the label does not fit into thin slices, put it outside of the pie
            inside = value / n >= 0.05
            radius = 0.7 if inside else 1.12
            ax.text(
                radius * np.cos(angle),
                radius * np.sin(angle),
                f"{100 * value / n:.0f}%",
                ha="center",
                va="center",
                color="white" if inside else ink,
                fontweight="bold",
            )
        ax.legend(
            wedges,
            [f"{label}: {v:,} ({100 * v / n:.1f}%)" for label, v in zip(labels, values)],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )
        ax.set_title(title, loc="left", fontsize=11)
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

    # 2) did the VLM report the failure, or did the request fail?
    counts = Counter(failure_source(row) for row in rows)
    request_errors = sorted(
        (label for label in counts if label != "Reported by VLM"),
        key=lambda label: (-counts[label], int(label.split()[-1])),
    )
    # at most 6 slices: fold the rarest status codes into one slice
    labels = ["Reported by VLM"] + request_errors[:4]
    values = [counts[label] for label in labels]
    if len(request_errors) > 4:
        labels.append("Other request errors")
        values.append(sum(counts[label] for label in request_errors[4:]))
    pie(
        axes["source"],
        labels,
        values,
        "Failed regions by source\n(HTTP status code of failed requests)",
    )
    sources = labels

    # 3) each flag on its own, a region can have several of them
    ax = axes["flags"]
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
        labels=[f"{v:,} ({100 * v / n:.0f}%)" for v in values[::-1]],
        padding=4,
        color=ink,
    )
    ax.set_xlim(0, max(values) * 1.35)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_title(
        "Failed regions per flag\n(a region can have several flags)",
        loc="left",
        fontsize=11,
    )
    ax.set_xlabel("Failed regions")

    # 4) do the sources fail on different kinds of regions? same groups as
    # the source pie, same colors as the category pie
    ax = axes["crosstab"]
    source_counts = Counter()
    pairs = Counter()
    for row in rows:
        source = failure_source(row)
        if source not in sources:
            source = "Other request errors"
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

    fig.suptitle(
        f"{n:,} failed regions in {n_images:,} images", x=0.01, ha="left", fontsize=13
    )
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
def cli(
    xml_folder,
    output_folder,
    min_size,
    white_threshold,
    black_threshold,
    jobs,
    image_folder,
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
                image_folder,
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
    rows = [row for xml_path in xml_files for row in rows_per_file.get(xml_path, [])]

    suffix = settings_suffix(min_size, white_threshold, black_threshold)
    csv_path = output_folder / CSV_NAME.format(suffix)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    click.echo(f"Wrote {len(rows)} failed regions to {csv_path}")
    for source, count in sorted(Counter(failure_source(row) for row in rows).items()):
        click.echo(f"  {source}: {count}")

    plot_path = output_folder / PLOT_NAME.format(suffix)
    if rows and plot_stats(rows, min_size, white_threshold, black_threshold, plot_path):
        click.echo(f"Saved statistics plot to {plot_path}")


if __name__ == "__main__":
    cli()
