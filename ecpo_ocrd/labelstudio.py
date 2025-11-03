from ocrd import Processor, Workspace
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_file import OcrdFileType
from ocrd_utils import config, bbox_from_points
from typing import Optional

import click
import json
import pathlib
import tqdm


_color_list = [
    "#3498DB",
    "#E74C3C",
    "#E67E22",
    "#9B59B6",
]


class LabelStudioExportProcessor(Processor):
    def process_workspace(self, workspace: Workspace) -> None:
        # Do I really need to do this manually?
        pathlib.Path(self.output_file_grp).mkdir(exist_ok=True)

        # Create the output data structure
        self.tasks = []
        self.labels = set()

        # Jump to process_page_file
        super().process_workspace(workspace)

        color_map = {l: _color_list[i] for i, l in enumerate(self.labels)}

        labels = "\n".join(
            f'<Label value="{l}" background="{color_map[l]}"/>' for l in self.labels
        )

        # Create a LabelStudio XML file
        xml_content = f"""
<View>

  <Header value="OCR-D created annotations"/>
  <Image name="image" value="$image" zoom="true"/>

  <Labels name="label" toName="image">
    {labels}
  </Labels>

  <Polygon name="plabel" toName="image" />
  <Ellipse name="elabel" toName="image" />
  <Rectangle name="rlabel" toName="image" />
</View>
"""

        workspace.add_file(
            file_id="labelstudio_json",
            file_grp=self.output_file_grp,
            content=json.dumps(self.tasks),
            mimetype="application/json",
            page_id=None,
            local_filename=pathlib.Path(self.output_file_grp).resolve()
            / "labelstudio.json",
            force=config.OCRD_EXISTING_OUTPUT == "OVERWRITE",
        )

        workspace.add_file(
            file_id="labelstudio_xml",
            file_grp=self.output_file_grp,
            content=xml_content,
            mimetype="text/xml",
            page_id=None,
            local_filename=pathlib.Path(self.output_file_grp).resolve()
            / "labelstudio.xml",
            force=config.OCRD_EXISTING_OUTPUT == "OVERWRITE",
        )

    def process_page_file(self, *input_files: Optional[OcrdFileType]) -> None:
        pcgts = page_from_file(input_files[0])
        self.page = pcgts.get_Page()

        annotations = []

        # Iterate over text regions
        for region in self.page.get_TextRegion():
            # Determine label
            label = "text"
            if region.type_:
                label = region.type_

            annotations.append(self._handle_region(region, label))

        # Iterate over image regions
        for region in self.page.get_ImageRegion():
            annotations.append(self._handle_region(region, "image"))

        # Create the final task object
        key = "annotations" if self.parameter["predictions"] else "predictions"
        self.tasks.append(
            {
                "id": self.page.id,
                "data": {
                    "image": pathlib.Path(self.page.imageFilename).name,
                    "name": pathlib.Path(self.page.imageFilename).stem,
                },
                key: [{"result": annotations}],
            }
        )

    def _handle_region(self, region, label):
        # Register this label
        self.labels.add(label)

        coords = bbox_from_points(region.get_Coords().points)

        return {
            "original_width": int(self.page.imageWidth),
            "original_height": int(self.page.imageHeight),
            "image_rotation": 0,
            "id": region.id,
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": coords[0] / self.page.imageWidth * 100,
                "y": coords[1] / self.page.imageHeight * 100,
                "width": (coords[2] - coords[0]) / self.page.imageWidth * 100,
                "height": (coords[3] - coords[1]) / self.page.imageHeight * 100,
                "labels": [label],
            },
        }


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(LabelStudioExportProcessor, *args, **kwargs)


def _filename_to_url(filename):
    if "jingbao" in filename:
        pass


def correct_urls(json_filename, sds, data_source):
    with open(json_filename, "r") as f:
        data = json.load(f)

    for task in tqdm.tqdm(data):
        # Find the one occurence on SDS
        sds_paths = list(
            (sds / "cats-ecpo" / "ecpo" / data_source).rglob(task["data"]["image"])
        )

        assert sds_paths

        def pick_one(paths):
            for path in paths:
                if "Old files" not in str(path):
                    return path

        sds_path = pick_one(sds_paths)

        # Replace with IIIF URL
        task["data"][
            "image"
        ] = f"https://ecpo.cats.uni-heidelberg.de/fcgi-bin/iipsrv.fcgi?IIIF=/ecpo/images/{str(sds_path.relative_to(sds / 'cats-ecpo' / 'ecpo'))}/full/full/0/default.jpg"

    with open(json_filename, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    cli()
