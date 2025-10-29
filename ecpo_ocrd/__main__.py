from ecpo_ocrd.install import install as install_main
from ecpo_ocrd.install import uninstall as uninstall_main
from ecpo_ocrd.workspace import create as create_main

import click
import pathlib


@click.group()
def main():
    """Orchestration commands for our efforts to process ECPO data with OCR-D."""
    pass


@main.command("install")
@click.option(
    "--prefix",
    "-p",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        path_type=pathlib.Path,
        exists=True,
        resolve_path=True,
    ),
    default=".",
    help="Installation prefix",
)
def install(prefix):
    """Install the required ECPO OCR-D tools into the given prefix."""

    install_main(prefix=prefix)


@main.group()
def workspace():
    """Workspace management commands for our efforts to process ECPO data with OCR-D."""
    pass


@workspace.command("create")
@click.option(
    "--workspace",
    "-w",
    type=click.Path(
        path_type=pathlib.Path,
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    default=".",
    help="The workspace directory to create.",
    show_default=True,
)
@click.option(
    "--sds",
    "-s",
    type=click.Path(
        path_type=pathlib.Path,
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    help="The SDS@HD mountpoint. Must point to the directory of the SV sd21c016.",
    default=pathlib.Path.home() / "sds" / "sd21c016",
    show_default=True,
)
@click.option(
    "--data",
    "-d",
    type=str,
    help="Magazine identifier in the ECPO data",
    default="jingbao",
    show_default=True,
)
@click.option(
    "--sample",
    type=int,
    help="The size of the sample dataset (Use 0 for full data set).",
    default=0,
    show_default=True,
)
def create(workspace, sds, data, sample):
    create_main(workspace=workspace, sds=sds, data=data, sample=sample)


@main.command("uninstall")
@click.option(
    "--prefix",
    "-p",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        path_type=pathlib.Path,
        exists=True,
        resolve_path=True,
    ),
    default=".",
    help="Installation prefix",
)
def uninstall(prefix):
    """Uninstall the ECPO OCR-D tools from the given prefix."""
    uninstall_main(prefix=prefix)


if __name__ == "__main__":
    main()
