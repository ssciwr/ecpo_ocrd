import contextlib
import os
import pathlib
import random
import shutil
import subprocess


@contextlib.contextmanager
def chdir(path):
    """Backport of contextlib.chdir from Python 3.11+."""
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)


def create(workspace=None, sds=None, data=None, sample=None):
    assert workspace is not None
    assert sds is not None
    assert data is not None

    # Create the workspace directory
    workspace.mkdir(parents=True, exist_ok=True)

    # Gather images from SDS
    images = list((sds / "cats-ecpo" / "ecpo" / data).glob("**/*.tif"))

    # Maybe trim this down to a sample
    if sample > 0:
        random.shuffle(images)
        images = images[:sample]

    # Create a workspace id
    workspace_id = data
    if sample > 0:
        workspace_id += f"-sample{sample}"

    with chdir(workspace):
        # Create a folder in the workspace for the original images
        (pathlib.Path(".") / "OCR-D-IMG").mkdir(parents=True, exist_ok=True)

        # Initialize the workspace
        subprocess.run(["ocrd", "workspace", "init", str(workspace)], check=True)
        subprocess.run(["ocrd", "workspace", "set-id", workspace_id], check=True)

        # Add the images to the workspace
        for i, image in enumerate(images):
            copied = shutil.copy(image, pathlib.Path(".") / "OCR-D-IMG" / image.name)
            subprocess.run(
                [
                    "ocrd",
                    "workspace",
                    "add",
                    "-g",
                    f"P000{i+1}",
                    "-G",
                    "OCR-D-IMG",
                    "-i",
                    f"OCR-D-IMG-{i+1:04d}",
                    "-m",
                    "image/tiff",
                    str(copied),
                ],
                check=True,
            )
