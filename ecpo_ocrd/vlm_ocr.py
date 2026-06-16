from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

import click


class VLMOCRProcessor(Processor):
    pass


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(VLMOCRProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
