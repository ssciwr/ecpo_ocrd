"""Send one region image to the vLLM server, as ocrd-vlm-ocr does.

Builds the same request bodies as ocrd-vlm-ocr (see vlm_ocr.py): the image is
sent as a PNG data URL, with greedy sampling. Like the processor, it first asks
for the reading order, then sends the OCR prompt with that reading order. With
--reading-order the reading order step is skipped. Each request is saved as
JSON next to the image and sent with curl, its response is saved next to
it as <image>.<kind>.response.json. With --upscale the image is resized by
that factor before it is sent, the resized image is saved as <image>.x<factor>.png
and the file names of requests and responses get .x<factor> too, so they do not
overwrite the results without upscaling.

Usage:
    python ecpo_ocrd/make_vllm_request.py <image_path>
        [--server http://localhost:9080/v1]
        [--model Qwen/Qwen3-VL-32B-Instruct]
        [--reading-order "top to bottom, right-to-left"]
        [--repetition-penalty 1.0]
        [--upscale 1.0]
"""

import base64
import json
import subprocess
from io import BytesIO
from pathlib import Path

import click
from PIL import Image


def load_image(image_path: str, upscale: float = 1.0) -> Image.Image:
    """Load the image as RGB, resized by the upscale factor (Lanczos)."""
    image = Image.open(image_path).convert("RGB")
    if upscale != 1.0:
        size = (
            max(1, round(image.width * upscale)),
            max(1, round(image.height * upscale)),
        )
        image = image.resize(size, Image.LANCZOS)
    return image


def image_data_url(image: Image.Image) -> str:
    """Encode the image as a PNG data URL, as vlm_ocr.py does."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def make_request(
    prompt: str,
    image: Image.Image,
    model: str = "MODEL_NAME",
    repetition_penalty: float = 1.0,
    max_tokens: int = 1024,
) -> dict:
    """Return the chat completion request body for the prompt and image."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image)},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "repetition_penalty": repetition_penalty,
        "top_k": 0,
        "top_p": 1.0,
    }


OCR_PROMPT_TEMPLATE = """
OCR this image. This image is a crop of a text region from a historical Chinese newspaper, 
so it normally contains text. The text is mostly traditional Chinese; its reading order is {reading_order}.

Your result needs to fulfill **all** of these constraints:
* Transcribe exactly what is printed. Keep traditional and variant characters; 
do not convert to simplified or modern forms, and do not correct errors.
* Keep numbers as printed: Chinese numerals stay Chinese numerals, Arabic digits stay digits.
* Keep punctuation as printed; do not add any.
* Output one line per printed column (vertical text) or row (horizontal text), in the given reading order. 
For vertical right-to-left text, begin with the right-most column, even if it does not start at the top.
* Transcribe text in other languages (e.g. English, Japanese kana) as printed, without translating.
* Output only the transcription, with no comments or explanations.
* Only if no character is legible (e.g. the crop is blank or too blurry), output exactly one line: [OCR failed: <short reason in English>]

Double-check your response so that it fulfills all constraints.
"""

READING_ORDER_PROMPT = """
You are an expert in historic Chinese newspaper layouts.
Determine the reading order of the text in this image.

Answer with exactly one of these lines:
vertical, right-to-left
horizontal, right-to-left
horizontal, left-to-right

* vertical: the characters run top to bottom in columns, the columns are read from right to left.
* horizontal: the characters run in one or more rows, e.g. headlines. 
Use right-to-left unless the row clearly reads left to right (e.g. Latin letters or Arabic numbers).
* If the image shows only one character, answer: horizontal, right-to-left

Output only the answer line, without any explanation.
"""


def curl_json(url: str, data_file: str | None = None) -> dict:
    """GET url, or POST the JSON in data_file to it, with curl."""
    command = ["curl", "-sS", "--fail-with-body", url]
    if data_file is not None:
        command += ["-H", "Content-Type: application/json", "-d", f"@{data_file}"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(
            f"curl {url} failed ({result.returncode}): "
            f"{result.stdout.strip() or result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def send_request(
    server: str,
    kind: str,
    prompt: str,
    image: Image.Image,
    output_stem: str,
    model: str,
    repetition_penalty: float,
) -> dict:
    """Save the request as <output_stem>.<kind>.request.json, send it to the
    server, save the response as <output_stem>.<kind>.response.json and
    return it."""
    request = make_request(prompt, image, model, repetition_penalty)
    request_path = f"{output_stem}.{kind}.request.json"
    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(request, f, ensure_ascii=False)
    click.echo(f"{kind} request saved to {request_path}", err=True)
    response = curl_json(f"{server}/chat/completions", request_path)
    response_path = f"{output_stem}.{kind}.response.json"
    with open(response_path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
    click.echo(f"{kind} response saved to {response_path}", err=True)
    return response


@click.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--server",
    default="http://localhost:9080/v1",
    show_default=True,
    help="API URL of the vLLM server",
)
@click.option(
    "--model",
    default="Qwen/Qwen3-VL-32B-Instruct",
    show_default=True,
    help="Model name",
)
@click.option(
    "--reading-order",
    default=None,
    help="Reading order filled into the OCR prompt, e.g. "
    "'top to bottom, right-to-left'. Skips the reading order request. "
    "Default: ask the model for the reading order first",
)
@click.option("--repetition-penalty", default=1.0, show_default=True, type=float)
@click.option(
    "--upscale",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Resize the image by this factor (Lanczos) before sending it, "
    "e.g. 2 or 3 to test whether small crops fail because of their resolution",
)
def main(
    image_path: str,
    server: str,
    model: str,
    reading_order: str | None,
    repetition_penalty: float,
    upscale: float,
) -> None:
    """Send IMAGE_PATH with the reading order prompt, then with the OCR
    prompt to the vLLM server. Print the reading order and the JSON
    response of the OCR request."""
    server = server.rstrip("/")
    image = load_image(image_path, upscale)
    output_stem = str(Path(image_path).with_suffix(""))
    if upscale != 1.0:
        output_stem += f".x{upscale:g}"
        image.save(f"{output_stem}.png")
        click.echo(
            f"upscaled image ({image.width}x{image.height}) saved to {output_stem}.png",
            err=True,
        )
    if reading_order is None:
        response = send_request(
            server,
            "reading_order",
            READING_ORDER_PROMPT,
            image,
            output_stem,
            model,
            repetition_penalty,
        )
        reading_order = response["choices"][0]["message"]["content"]
        click.echo(f"reading order (from the model): {reading_order}")
    else:
        click.echo(f"reading order (given): {reading_order}")

    response = send_request(
        server,
        "ocr",
        OCR_PROMPT_TEMPLATE.format(reading_order=reading_order),
        image,
        output_stem,
        model,
        repetition_penalty,
    )
    click.echo(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
