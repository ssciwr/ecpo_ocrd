"""Trace the regions where VLM OCR failed back through the pipeline, to see
whether the OCR or the segmentation (Eynollah or PaddleOCR) caused the failure.

Written by Claude, checked by human.

The pipeline is

    image -> Eynollah segment -> PaddleOCR refine (ecpo-segment) -> VLM OCR

ecpo-segment gives the refined regions new ids, so a failed region is traced
back by geometry instead: every refined region is the intersection of one
Eynollah region with a PaddleOCR box, so its parent is the Eynollah text
region it overlaps most.

Usage:
    python ecpo_ocrd/analysis/trace_failed_regions.py <workspace> <failed_csv> <output_folder>
        --ocr-grp <OCR file group>
        [--ecpo-grp <group>] [--eynollah-grp <group>]
        [--image-folder <path to sds_ecpo>]
        [--crops-folder <folder of crop_and_analyze.py>]
        [--scale 0.5]
        [--show-boundaries]
        [--jobs 4]

<failed_csv> is the failed_regions_*.csv written by crop_and_analyze.py for
the OCR file group. The file groups of the earlier steps are found from the
processing steps (mets:agent) recorded in mets.xml: starting from the OCR
group, each step's input group is followed back to ocrd-ecpo-segment and
ocrd-eynollah-inference. --ecpo-grp / --eynollah-grp override them.

For every page with failed regions, <output_folder>/<image>_trace.png shows

1. Eynollah: the page with the Eynollah text and image regions, the parents
   of failed regions in blue; with --show-boundaries also the artificial
   boundaries
2. PaddleOCR: the page with the refined (ECPO) regions, the failed ones filled
   in the colour of their failure type, their Eynollah parents outlined in blue
3. VLM crops: the crops sent to the VLM, framed in the colour of their failure
   type, grouped by Eynollah parent

Failure types and their colours: failed (red: the VLM reported failure or
the request failed), skipped (yellow: one edge below min_edge), repetition
(purple).

The crops are taken from --crops-folder (default: the folder of
<failed_csv>, where crop_and_analyze.py saved them as <image>/<region id>.png),
otherwise they are cut from the page image in the same way.

The page images are read from the image file group of the workspace, from
--image-folder or, if neither has them, downloaded from the ECPO IIIF server
into <output_folder>/images/.

<output_folder>/traced_regions.csv holds the rows of <failed_csv> plus
* page_id: METS page id
* failure_type: failed, skipped or repetition
* eynollah_parent_id: the Eynollah text region the failed region overlaps most
* parent_overlap: share of the failed region inside that parent, ~1 for a
  clear parent; low values mean the parent is unclear
* refined_share: area of the failed region / area of the parent
* n_siblings: other refined regions with the same parent
* refined: False if ecpo-segment kept the Eynollah region as it is
"""

import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import click
from PIL import Image, ImageDraw, ImageFont
from shapely import make_valid
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

try:
    from ecpo_ocrd.analysis.crop_and_analyze import (
        SKIPPED_SOURCE,
        PageImage,
        crop_polygon,
        failure_source,
        parse_points,
    )
    from ecpo_ocrd.polygon import flatten_polygon_geometry
except ImportError:  # run as a script without the package installed
    # the modules of the pipeline are in the parent folder
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crop_and_analyze import (
        SKIPPED_SOURCE,
        PageImage,
        crop_polygon,
        failure_source,
        parse_points,
    )
    from polygon import flatten_polygon_geometry

METS_NS = {
    "mets": "http://www.loc.gov/METS/",
    "xlink": "http://www.w3.org/1999/xlink",
}
OCRD_OPTION = "{https://ocr-d.de}option"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

ECPO_PROCESSOR = "ocrd-ecpo-segment"
EYNOLLAH_PROCESSOR = "ocrd-eynollah-inference"

FAILED, SKIPPED, REPETITION = "failed", "skipped", "repetition"
TYPE_COLORS = {
    FAILED: (214, 39, 40),
    SKIPPED: (245, 200, 0),
    REPETITION: (148, 103, 189),
}
# the label text of the crops; yellow text is hard to read on white
TYPE_TEXT_COLORS = {**TYPE_COLORS, SKIPPED: (170, 130, 0)}
PARENT_COLOR = (31, 119, 180)
REGION_COLOR = (120, 120, 120)
# magenta, to stand out against the blue parents
IMAGE_REGION_COLOR = (230, 30, 170)
# only drawn with --show-boundaries
BOUNDARY_COLOR = (44, 160, 44)

# a failed region whose best parent covers less of it than this has no clear
# parent
MIN_PARENT_OVERLAP = 0.5

# layout of the PNG, in pixels
GAP = 24
HEADER_HEIGHT = 70
CAPTION_HEIGHT = 34
CROP_BOX = 220
CELL_WIDTH = 260
LABEL_LINES = 3
LINE_HEIGHT = 17
CELL_HEIGHT = CROP_BOX + 16 + LABEL_LINES * LINE_HEIGHT + 14
MAX_UPSCALE = 8

TRACE_FIELDS = [
    "page_id",
    "failure_type",
    "eynollah_parent_id",
    "parent_overlap",
    "refined_share",
    "n_siblings",
    "refined",
]


# ---------------------------------------------------------------- METS


@dataclass
class Agent:
    """One processing step recorded in mets.xml."""

    processor: str
    input_groups: list[str]
    output_groups: list[str]
    parameter: str


@dataclass
class Mets:
    agents: list[Agent]
    # file group -> page id -> path relative to the workspace
    files: dict[str, dict[str, str]]
    # (file group, file name) -> page id
    page_of_file: dict[tuple[str, str], str]


def read_mets(path: Path) -> Mets:
    root = ET.parse(path).getroot()

    agents = []
    for agent in root.iterfind(".//mets:metsHdr/mets:agent", METS_NS):
        notes = {
            note.get(OCRD_OPTION): (note.text or "")
            for note in agent.iterfind("mets:note", METS_NS)
        }
        if "output-file-grp" not in notes:
            continue  # e.g. the ocrd/core agent that created the workspace
        name = agent.find("mets:name", METS_NS).text or ""
        agents.append(
            Agent(
                processor=name.split()[0],
                input_groups=notes.get("input-file-grp", "").split(","),
                output_groups=notes["output-file-grp"].split(","),
                parameter=notes.get("parameter", ""),
            )
        )

    file_page = {}
    for page in root.iterfind(
        ".//mets:structMap[@TYPE='PHYSICAL']//mets:div[@TYPE='page']", METS_NS
    ):
        for fptr in page.iterfind("mets:fptr", METS_NS):
            file_page[fptr.get("FILEID")] = page.get("ID")

    files = defaultdict(dict)
    page_of_file = {}
    for group in root.iterfind(".//mets:fileGrp", METS_NS):
        use = group.get("USE")
        for file in group.iterfind("mets:file", METS_NS):
            page_id = file_page.get(file.get("ID"))
            flocat = file.find("mets:FLocat", METS_NS)
            if page_id is None or flocat is None:
                continue
            href = flocat.get(XLINK_HREF)
            # several files of one page in a group (e.g. images saved by a
            # processor): keep the PAGE-XML or, in the image group, the image
            if use in files and page_id in files[use] and not href.endswith(".xml"):
                continue
            files[use][page_id] = href
            page_of_file[(use, Path(href).name)] = page_id
    return Mets(agents, dict(files), page_of_file)


@dataclass
class Chain:
    ocr: str
    ecpo: str
    eynollah: str
    image: str
    steps: list[Agent]


def trace_chain(
    mets: Mets, ocr_grp: str, ecpo_grp: str = None, eynollah_grp: str = None
) -> Chain:
    """Follow the input groups of the processing steps back from the OCR group.

    If several steps wrote the same group, the last one wrote its current files.
    """
    steps = []
    group = ocr_grp
    seen = set()
    while group not in seen:
        seen.add(group)
        writers = [a for a in mets.agents if group in a.output_groups]
        if not writers:
            break
        agent = writers[-1]
        steps.append(agent)
        if agent.processor == EYNOLLAH_PROCESSOR:
            break
        group = agent.input_groups[0]

    def output_of(processor):
        for agent in steps:
            if agent.processor == processor:
                return agent.output_groups[0]
        return None

    ecpo = ecpo_grp or output_of(ECPO_PROCESSOR)
    eynollah = eynollah_grp or output_of(EYNOLLAH_PROCESSOR)
    if ecpo is None or eynollah is None:
        found = (
            " <- ".join(f"{a.output_groups[0]} ({a.processor})" for a in steps)
            or ocr_grp
        )
        raise click.UsageError(
            f"Could not find the {'ECPO' if ecpo is None else 'Eynollah'} file "
            f"group in mets.xml, found only {found}. Pass --ecpo-grp / "
            f"--eynollah-grp."
        )
    image = next(
        (a.input_groups[0] for a in mets.agents if eynollah in a.output_groups),
        "OCR-D-IMG",
    )
    return Chain(ocr_grp, ecpo, eynollah, image, steps)


# ---------------------------------------------------------------- regions


@dataclass
class Region:
    id: str
    kind: str  # e.g. TextRegion, ImageRegion
    points: list[tuple[int, int]]
    geom: object

    @property
    def centre(self) -> tuple[float, float]:
        c = self.geom.centroid
        return c.x, c.y


def to_geom(points: list[tuple[int, int]]):
    """Shapely geometry with an area, also for degenerate polygons (e.g. 1 px wide)."""
    if len(set(points)) >= 3:
        geom = unary_union(flatten_polygon_geometry(make_valid(Polygon(points))))
        if geom.area > 0:
            return geom
    shape = LineString(points) if len(set(points)) > 1 else Point(points[0])
    return shape.buffer(0.5)


def read_regions(xml_path: Path) -> tuple[ET.Element, list[Region]]:
    root = ET.parse(xml_path).getroot()
    ns = root.tag.split("}")[0] + "}"
    page = root.find(f"{ns}Page")
    regions = []
    for element in page.iter():
        kind = element.tag.split("}")[-1]
        coords = element.find(f"{ns}Coords")
        if not kind.endswith("Region") or coords is None:
            continue
        points = parse_points(coords.get("points"))
        regions.append(Region(element.get("id"), kind, points, to_geom(points)))
    return page, regions


def text_regions(regions: list[Region]) -> list[Region]:
    return [r for r in regions if r.kind == "TextRegion"]


@dataclass
class Parent:
    id: str  # "" if no Eynollah text region overlaps the region
    overlap: float
    share: float
    refined: bool


def assign_parents(children: list[Region], parents: list[Region]) -> dict[str, Parent]:
    """The Eynollah text region each refined region overlaps most."""
    parent_by_id = {p.id: p for p in parents}
    tree = STRtree([p.geom for p in parents])
    result = {}
    for child in children:
        same = parent_by_id.get(child.id)
        if same is not None and same.points == child.points:
            # ecpo-segment kept the Eynollah region (label not refined)
            result[child.id] = Parent(child.id, 1.0, 1.0, False)
            continue
        best, best_overlap = None, 0.0
        for i in tree.query(child.geom):
            overlap = child.geom.intersection(parents[i].geom).area / child.geom.area
            if overlap > best_overlap:
                best, best_overlap = parents[i], overlap
        if best is None:
            result[child.id] = Parent("", 0.0, 0.0, True)
        else:
            share = child.geom.area / best.geom.area
            result[child.id] = Parent(best.id, best_overlap, share, True)
    return result


def failure_type(row: dict) -> str:
    source = failure_source(row)
    if source == SKIPPED_SOURCE:
        return SKIPPED
    if source == "Repetition":
        return REPETITION
    return FAILED


# ---------------------------------------------------------------- drawing


def font(size: int):
    return ImageFont.load_default(size=size)


def scaled(points, scale: float) -> list[tuple[float, float]]:
    return [(x * scale, y * scale) for x, y in points]


def draw_outlines(base: Image.Image, items: list[tuple], scale: float) -> Image.Image:
    """Draw (points, outline colour, line width, fill RGBA or None) in order."""
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for points, outline, width, fill in items:
        pts = scaled(points, scale)
        if fill is not None and len(pts) >= 3:
            draw.polygon(pts, fill=fill)
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=outline + (255,), width=width)
        else:
            draw.point(pts, fill=outline + (255,))
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def fit_crop(crop: Image.Image) -> Image.Image:
    """Scale the crop into the CROP_BOX, enlarging small crops (pixels stay sharp)."""
    factor = min(CROP_BOX / crop.width, CROP_BOX / crop.height)
    if factor >= 1:
        factor = min(factor, MAX_UPSCALE)
        resample = Image.NEAREST
    else:
        resample = Image.LANCZOS
    size = (max(1, round(crop.width * factor)), max(1, round(crop.height * factor)))
    return crop.convert("RGB").resize(size, resample)


def short_reason(row: dict, ftype: str) -> str:
    if ftype == SKIPPED:
        return "skipped: edge too small"
    if ftype == REPETITION:
        return f"repetition x{row['repeat_times']}"
    text = row["ocr_text"].strip()
    for prefix in ("[OCR failed:", "[OCR failed"):
        if text.startswith(prefix):
            text = text[len(prefix) :].rstrip("]").strip()
            break
    return text if len(text) <= 32 else text[:31] + "…"


def legend(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    counts: Counter,
    show_boundaries: bool,
) -> None:
    entries = [
        (TYPE_COLORS[t], f"{t} ({counts.get(t, 0)})")
        for t in (FAILED, SKIPPED, REPETITION)
    ] + [
        (PARENT_COLOR, "Eynollah parent of a failed region"),
        (REGION_COLOR, "other text regions"),
        (IMAGE_REGION_COLOR, "Eynollah image region"),
    ]
    if show_boundaries:
        entries.append((BOUNDARY_COLOR, "Eynollah artificial boundary"))
    for colour, label in entries:
        draw.rectangle([x, y, x + 18, y + 18], fill=colour)
        draw.text((x + 26, y), label, fill="black", font=font(16))
        x += 26 + int(draw.textlength(label, font=font(16))) + 28


def render_page(
    title: str,
    page_image: Image.Image,
    eynollah: list[Region],
    ecpo: list[Region],
    failed: list[dict],
    scale: float,
    show_boundaries: bool = False,
) -> Image.Image:
    """failed: dicts with region, type, parent, row and crop."""
    size = (round(page_image.width * scale), round(page_image.height * scale))
    base = page_image.convert("RGB").resize(size, Image.LANCZOS)
    line = max(1, round(4 * scale))

    # 1. Eynollah
    parent_ids = {f["parent"].id for f in failed}
    items = []
    for r in eynollah:
        if r.kind == "ImageRegion":
            items.append((r.points, IMAGE_REGION_COLOR, line, None))
        elif r.kind == "LineDrawingRegion" and show_boundaries:
            items.append((r.points, BOUNDARY_COLOR, line, None))
        elif r.kind == "TextRegion" and r.id not in parent_ids:
            items.append((r.points, REGION_COLOR, line, None))
    for r in eynollah:
        if r.kind == "TextRegion" and r.id in parent_ids:
            items.append((r.points, PARENT_COLOR, 2 * line + 1, PARENT_COLOR + (45,)))
    panel1 = draw_outlines(base, items, scale)

    # 2. PaddleOCR (ECPO)
    failed_ids = {f["region"].id for f in failed}
    items = [
        (r.points, REGION_COLOR, max(1, line - 1), None)
        for r in text_regions(ecpo)
        if r.id not in failed_ids
    ]
    # the Eynollah parents, without fill so that the refined regions in them
    # stay visible
    items += [
        (r.points, PARENT_COLOR, 2 * line + 1, None)
        for r in eynollah
        if r.kind == "TextRegion" and r.id in parent_ids
    ]
    # thick outline, so that tiny regions stay visible after scaling
    for ftype in (SKIPPED, FAILED, REPETITION):
        colour = TYPE_COLORS[ftype]
        items += [
            (f["region"].points, colour, 3, colour + (110,))
            for f in failed
            if f["type"] == ftype
        ]
    panel2 = draw_outlines(base, items, scale)

    # 3. VLM crops
    width = 2 * base.width + 3 * GAP
    columns = max(1, (width - GAP) // CELL_WIDTH)
    rows = -(-len(failed) // columns)
    grid_height = rows * CELL_HEIGHT

    height = (
        HEADER_HEIGHT
        + CAPTION_HEIGHT
        + base.height
        + GAP
        + CAPTION_HEIGHT
        + grid_height
        + GAP
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    counts = Counter(f["type"] for f in failed)
    draw.text((GAP, 10), title, fill="black", font=font(24))
    legend(draw, GAP, 42, counts, show_boundaries)

    y = HEADER_HEIGHT
    captions = [
        "1. Eynollah segmentation",
        "2. PaddleOCR refinement (ECPO), failed regions filled, Eynollah parents in blue",
    ]
    for i, (panel, caption) in enumerate(zip((panel1, panel2), captions)):
        x = GAP + i * (base.width + GAP)
        draw.text((x, y + 6), caption, fill="black", font=font(20))
        canvas.paste(panel, (x, y + CAPTION_HEIGHT))
    y += CAPTION_HEIGHT + base.height + GAP
    draw.text(
        (GAP, y + 6),
        "3. Crops sent to the VLM, grouped by Eynollah parent "
        "(right to left, top to bottom); label: region id @ position "
        "(% from top / % from left)",
        fill="black",
        font=font(20),
    )
    y += CAPTION_HEIGHT

    for i, f in enumerate(failed):
        cx = GAP + (i % columns) * CELL_WIDTH
        cy = y + (i // columns) * CELL_HEIGHT
        colour = TYPE_COLORS[f["type"]]
        crop = fit_crop(f["crop"])
        box_x = cx + (CELL_WIDTH - 16 - CROP_BOX) // 2
        draw.rectangle(
            [box_x - 4, cy - 4, box_x + CROP_BOX + 3, cy + CROP_BOX + 3],
            outline=colour,
            width=4,
        )
        canvas.paste(
            crop,
            (
                box_x + (CROP_BOX - crop.width) // 2,
                cy + (CROP_BOX - crop.height) // 2,
            ),
        )
        row, parent = f["row"], f["parent"]
        parent_text = (
            f"in {parent.id} ({parent.share:.0%})" if parent.id else "no parent"
        )
        if parent.id and parent.overlap < MIN_PARENT_OVERLAP:
            parent_text += f" ?{parent.overlap:.0%}"
        labels = [
            f"{f['region'].id} @{row['pos_top_left']}",
            f"{row['width']}x{row['height']} {parent_text}",
            short_reason(row, f["type"]),
        ]
        for j, label in enumerate(labels):
            draw.text(
                (cx, cy + CROP_BOX + 10 + j * LINE_HEIGHT),
                label,
                fill=TYPE_TEXT_COLORS[f["type"]] if j == 2 else "black",
                font=font(13),
            )
    return canvas


# ---------------------------------------------------------------- pages


@dataclass
class PageJob:
    workspace: Path
    page_id: str
    ocr_xml: Path
    ecpo_xml: Path
    eynollah_xml: Path
    image_href: str
    rows: list[dict]
    output_folder: Path
    crops_folder: Path
    image_folder: Path
    scale: float
    show_boundaries: bool


def open_page_image(job: PageJob, page: ET.Element, name: str):
    """Return the page image and the PageImage to close (or None)."""
    if job.image_folder is None and job.image_href:
        local = job.workspace / job.image_href
        if local.exists():
            return Image.open(local), None
    stem = Path(page.get("imageFilename")).stem
    cache = job.output_folder / "images" / stem
    cache.mkdir(parents=True, exist_ok=True)
    page_image = PageImage(page, cache, name, job.image_folder)
    return page_image.get(), page_image


def process_page(job: PageJob) -> tuple[list[dict], str]:
    """Return the traced rows and a log line."""
    _, ocr_regions = read_regions(job.ocr_xml)
    page, ecpo_regions = read_regions(job.ecpo_xml)
    _, eynollah_regions = read_regions(job.eynollah_xml)

    ecpo_text = text_regions(ecpo_regions)
    parents = assign_parents(ecpo_text, text_regions(eynollah_regions))
    n_children = Counter(p.id for p in parents.values() if p.id)
    ecpo_by_id = {r.id: r for r in ecpo_text}
    ocr_points = {r.id: r.points for r in ocr_regions}

    traced, failed, notes = [], [], []
    n_moved = 0
    for row in job.rows:
        region = ecpo_by_id.get(row["region_id"])
        if region is None:
            notes.append(f"{row['region_id']} not in {job.ecpo_xml.name}")
            continue
        if ocr_points.get(region.id) != region.points:
            n_moved += 1
        parent = parents[region.id]
        ftype = failure_type(row)
        traced.append(
            {
                **row,
                "page_id": job.page_id,
                "failure_type": ftype,
                "eynollah_parent_id": parent.id,
                "parent_overlap": round(parent.overlap, 3),
                "refined_share": round(parent.share, 4),
                "n_siblings": max(0, n_children[parent.id] - 1) if parent.id else 0,
                "refined": parent.refined,
            }
        )
        failed.append({"region": region, "type": ftype, "parent": parent, "row": row})
    if n_moved:
        notes.append(
            f"{n_moved} regions have other coordinates in the OCR XML than in "
            f"{job.ecpo_xml.name}, is the ECPO file group right?"
        )
    if not failed:
        return traced, f"{job.page_id}: no failed regions found, " + "; ".join(notes)

    # group the crops by parent, parents and regions right to left, top to bottom
    parent_regions = {r.id: r for r in text_regions(eynollah_regions)}

    def reading_key(region: Region):
        x, y = region.centre
        return (-x, y)

    def sort_key(f):
        parent = parent_regions.get(f["parent"].id)
        if parent is None:
            return (1, (0, 0), reading_key(f["region"]))
        return (0, reading_key(parent), reading_key(f["region"]))

    failed.sort(key=sort_key)

    image_name = traced[0]["image"]
    image, page_image = open_page_image(job, page, job.ocr_xml.name)
    try:
        for f in failed:
            crop_path = job.crops_folder / image_name / f"{f['region'].id}.png"
            if crop_path.exists():
                f["crop"] = Image.open(crop_path)
            else:
                f["crop"] = crop_polygon(image, f["region"].points)
        title = (
            f"{image_name}  ·  page {job.page_id}  ·  "
            f"{len(failed)} failed regions of {len(ecpo_text)} refined text "
            f"regions  ·  {len(text_regions(eynollah_regions))} Eynollah text regions"
        )
        canvas = render_page(
            title,
            image,
            eynollah_regions,
            ecpo_regions,
            failed,
            job.scale,
            job.show_boundaries,
        )
    finally:
        if page_image is not None:
            page_image.close()
        else:
            image.close()

    png = job.output_folder / f"{image_name}_trace.png"
    part = png.with_name(png.name + ".part")
    canvas.save(part, format="PNG", optimize=True)
    part.replace(png)
    log = f"{job.page_id}: {len(failed)} failed regions -> {png.name}"
    if notes:
        log += " (" + "; ".join(notes) + ")"
    return traced, log


# ---------------------------------------------------------------- CLI


@click.command()
@click.argument(
    "workspace", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "failed_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_folder", type=click.Path(file_okay=False, path_type=Path))
@click.option("--ocr-grp", required=True, help="File group of the OCR results")
@click.option(
    "--ecpo-grp",
    default=None,
    help="File group of ocrd-ecpo-segment, default: found in mets.xml",
)
@click.option(
    "--eynollah-grp",
    default=None,
    help="File group of ocrd-eynollah-inference, default: found in mets.xml",
)
@click.option(
    "--image-folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Folder with the original images, e.g. the path to sds_ecpo for "
    'imageFilename="../sds_ecpo/cats-ecpo/...". Default: the image file group '
    "of the workspace, or download from the ECPO IIIF server",
)
@click.option(
    "--crops-folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Output folder of crop_and_analyze.py with the crops as "
    "<image>/<region id>.png. Default: the folder of FAILED_CSV",
)
@click.option(
    "--scale",
    type=click.FloatRange(0.05, 1),
    default=0.5,
    show_default=True,
    help="Scale of the page panels relative to the original image",
)
@click.option(
    "--show-boundaries",
    is_flag=True,
    default=False,
    help="Also draw the artificial boundaries (LineDrawingRegion) found by "
    "Eynollah in panel 1",
)
@click.option(
    "--jobs",
    "-j",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of pages processed in parallel",
)
def cli(
    workspace,
    failed_csv,
    output_folder,
    ocr_grp,
    ecpo_grp,
    eynollah_grp,
    image_folder,
    crops_folder,
    scale,
    show_boundaries,
    jobs,
):
    """Trace the failed regions in FAILED_CSV (from crop_and_analyze.py) back
    through the segmentation steps of WORKSPACE, one PNG per page in
    OUTPUT_FOLDER."""
    mets = read_mets(workspace / "mets.xml")
    if ocr_grp not in mets.files:
        raise click.UsageError(f"File group {ocr_grp} not found in mets.xml")
    chain = trace_chain(mets, ocr_grp, ecpo_grp, eynollah_grp)
    click.echo("Pipeline found in mets.xml:")
    for agent in chain.steps:
        click.echo(
            f"  {agent.output_groups[0]} <- {agent.processor} "
            f"{agent.parameter}  (input {agent.input_groups[0]})"
        )
    click.echo(
        f"Using OCR {chain.ocr}, ECPO {chain.ecpo}, Eynollah {chain.eynollah}, "
        f"images {chain.image}"
    )
    for group in (chain.ecpo, chain.eynollah):
        if group not in mets.files:
            raise click.UsageError(f"File group {group} not found in mets.xml")

    with open(failed_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        in_fields = list(reader.fieldnames)
        rows = list(reader)
    rows_by_file = defaultdict(list)
    for row in rows:
        rows_by_file[row["xml_file"]].append(row)

    output_folder.mkdir(parents=True, exist_ok=True)
    jobs_list = []
    for xml_file, file_rows in rows_by_file.items():
        page_id = mets.page_of_file.get((chain.ocr, xml_file))
        if page_id is None:
            click.echo(f"{xml_file}: not in file group {chain.ocr}, skipped", err=True)
            continue
        missing = [
            g for g in (chain.ecpo, chain.eynollah) if page_id not in mets.files[g]
        ]
        if missing:
            click.echo(f"{page_id}: no file in {', '.join(missing)}, skipped", err=True)
            continue
        jobs_list.append(
            PageJob(
                workspace=workspace,
                page_id=page_id,
                ocr_xml=workspace / mets.files[chain.ocr][page_id],
                ecpo_xml=workspace / mets.files[chain.ecpo][page_id],
                eynollah_xml=workspace / mets.files[chain.eynollah][page_id],
                image_href=mets.files.get(chain.image, {}).get(page_id, ""),
                rows=file_rows,
                output_folder=output_folder,
                crops_folder=crops_folder or failed_csv.parent,
                image_folder=image_folder,
                scale=scale,
                show_boundaries=show_boundaries,
            )
        )

    traced_by_page = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(process_page, job): job for job in jobs_list}
        for future in as_completed(futures):
            job = futures[future]
            try:
                traced_by_page[job.page_id], log = future.result()
                click.echo(log)
            except Exception as e:
                click.echo(f"{job.page_id}: failed ({e})", err=True)

    traced = [row for job in jobs_list for row in traced_by_page.get(job.page_id, [])]
    csv_path = output_folder / "traced_regions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=in_fields + TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(traced)

    unclear = sum(float(r["parent_overlap"]) < MIN_PARENT_OVERLAP for r in traced)
    click.echo(f"Wrote {len(traced)} traced regions to {csv_path}")
    for ftype, count in sorted(Counter(r["failure_type"] for r in traced).items()):
        click.echo(f"  {ftype}: {count}")
    click.echo(
        f"  without a clear Eynollah parent (overlap < {MIN_PARENT_OVERLAP}): "
        f"{unclear}"
    )


if __name__ == "__main__":
    cli()
