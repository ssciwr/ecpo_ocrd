from ocrd import Processor, Workspace, OcrdPage, OcrdPageResult, OcrdPageResultImage
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from install import create_env

from typing import Optional
import click
import subprocess
import os


# wrap Eynollah inference into an ocrd processor class
class EynollahInferenceProcessor(Processor):
    """OCR-D Processor for Eynollah inference"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_page_pcgts(
        self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None
    ) -> OcrdPageResult:
        """Override process_page_pcgts of the Processor class to perform
        Eynollah inference on the input page(s) and return the result as an OcrdPageResult.
        """
        assert input_pcgts
        assert input_pcgts[0]
        assert self.parameter  # default values or from CLI with -p or -P

        pcgts = input_pcgts[0]
        page = pcgts.get_Page()
        img_filename = page.imageFilename
        img_filepath = os.path.join(self.workspace.directory, img_filename)

        page_image, page_coords, page_info = self.workspace.image_from_page(
            page,
            page_id,
        )

        model = self.parameter["model"]
        env_identifier = self.parameter["env_identifier"]

        # Ensure existence of the environment
        env = create_env(env_identifier=env_identifier)

        self.logger.info(
            f"Running Eynollah inference on page {page_id} with model {model}"
        )

        # run eynollah inference command
        subprocess.run(
            [
                str(env / "bin" / "eynollah-training"),
                "inference",
                "--model",
                model,
                "--image",
                img_filepath,
                "--patches",  # process in patches
                "--save_layout",  # only save layout prediction
                self.output_file_grp,
            ],
            check=True,
        )

        # TODO: update xml file with the new layout information from Eynollah inference


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(EynollahInferenceProcessor, *args, **kwargs)
