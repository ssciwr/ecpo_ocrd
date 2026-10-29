# OCR-D Pipeline for ECPO

**Work in Progress**

## Prerequisites

* Python `==3.10` (others might work, but this is what we tested. Very recent versions do not work.)

## Pipeline overview

Our pipeline consists of the following steps:

* `eynollah-inference`: Run Eynollah inference to get layout segmentation results from a fine-tuned Eynollah model. See [this training/fine-tuning instruction](https://github.com/ssciwr/ecpo-eynollah/blob/main/eynollah_training.md) for the details on our fine-tuned models.
* `ecpo-segment`: Refine the layout segmentation results from Eynollah using PaddleOCR's layout analysis model.
* `ocrd-vlm-ocr`: Run OCR on the text regions obtained from the previous step using VLLM.

## Installation
1. Clone the repository

```bash
git clone https://github.com/ssciwr/ecpo_ocrd.git
cd ecpo_ocrd
```

2. To install the repository's dependencies, there are two options, please choose just one of them:

* `eynollah-gpu`: for running Eynollah inference with GPU support (requires CUDA and cuDNN)
* `non-eynollah-gpu`: for running non-GPU-Eynollah processes (e.g. Eynollah on CPU or PaddleOCR)

For example, to install the dependencies for running the `eynollah-inference` and `ecpo-segment` with CPU , use:

```bash
python -m pip install .[non-eynollah-gpu]
```

3. Then install relevant OCR-D tools with:

```bash
ecpo_ocrd install
```


## Usage

**Before** doing anything, add the `bin` folder to `PATH`:

```bash
export PATH=$PWD/bin:$PATH
```

Then, create an OCR-D workspace for your data via:

```bash
mkdir myworkspace
cd myworkspace
ecpo_ocrd workspace create
```

For details, please have a look at `ecpo_ocrd workspace create --help`.
For testing purposed, you should *always* add `--sample n` with a reasonably
small `n` (otherwise you will process the entire subcorpus for the selected
data source).

Then, you can run any of the provided workflow shell scripts e.g.

```bash
../workflows/jingbao_demo.sh
```

## Running with GPUs

### Segmentation steps

#### Prepare conda environments

Dependencies for Eynollah inference and PaddleOCR are mutually exclusive due to `numpy` version.

To utilize GPU for Eynollah process, the following dependencies should be installed:

* `Python` 3.10
* `CUDA` 11.8
* `cuDNN` 8.6
* `TensorFlow` 2.12.0

So far, we can only create a `conda` environment with the above settings. Therefore, even though we already created `envs` and `bin` folders with `ecpo_ocrd install`, we still need to use `conda` environments to run Eynollah-related processes.

A quick-and-dirty solution to run the whole pipeline with GPU is to create two `conda` environments, one for Eynollah and one for the rest of the pipeline.

##### Non-Eynollah conda environment

Create the second conda environment and install dependencies for the rest of the pipeline with:

```bash
conda create -n ecpo_non_eynollah python=3.10
conda activate ecpo_non_eynollah
pip install .[non-eynollah-gpu] # from the root folder of ecpo_ocrd repository
```

#### Create workspace, if needed

If the workspace is not created yet, create it with `ecpo_non_eynollah` environment activated.

```bash
conda activate ecpo_non_eynollah
ecpo_ocrd install
export PATH=$PWD/bin:$PATH
```

For the specific Jingbao dataset, we use:

```bash
ecpo_ocrd workspace create --workspace <path_to_workspace> --sds <path_to_sds> --data jingbao --sample <n>
```

Omit `--sample <n>` to process the entire dataset. Otherwise, set a small number for testing purposes.

##### Eynollah conda environment

1. Use [this instruction](https://github.com/ssciwr/ecpo-eynollah/blob/main/eynollah_training.md#step-by-step-installation) to setup a `conda` environment, named e.g. `ecpo_eynollah`, with the specified Python, CUDA, cuDNN, and TensorFlow versions.

2. Install the Eynollah-related dependencies in this environment with:

```bash
pip install .[eynollah-gpu] # from the root folder of ecpo_ocrd repository
```

#### Run the parallel script

Move to the root folder of the repository (`ecpo_ocrd`) and run the parallel script up to the `ecpo-segment` step with:

```bash
# assuming you are in the repository root
./workflows/jingbao_parallel_eynollah_paddle.sh [gpu_id_optional] <path_to_workspace> <eynollah_conda_env> <non_eynollah_conda_env>
```

The gpu_id is optional with default GPU `0`. To specify a different GPU, use e.g. `1` for GPU `1`. Eynollah only uses one GPU for now.

### OCR step

#### Prepare vLLM server

To run the last step `ocrd-vlm-ocr` with VLLM, create a new conda environment to prevent clash on torch/CUDA stack:

```bash
conda create -n ecpo_vllm python=3.11
conda activate ecpo_vllm
pip install vllm
```

If the home directory is too small to install `vllm`, use data disk instead:

```bash
export PIP_CACHE_DIR=/mnt/data/<your_name>/pip-cache
export TMPDIR=/mnt/data/<your_name>/tmp          # pip unpacks wheels in TMPDIR
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

pip download vllm -d /mnt/data/<your_name>/vllm-wheels --timeout 120 --retries 10
pip install --no-index --find-links /mnt/data/<your_name>/vllm-wheels vllm
```

Then, point vLLM cache to a folder with enough space (e.g. 100GB) to store the model weights:

```bash
export HF_HOME=/mnt/data/<your_name>/vllm-cache
```

Start the vLLM server in a new screen to not block the current termind:

```bash
screen -S vllm_server
conda activate ecpo_vllm # if not already activated

export VLLM_USE_FLASHINFER_SAMPLER=0

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --port 9080 \
    --max-model-len 5000 \
    --gpu-memory-utilization 0.95 \
    --limit-mm-per-prompt.video 0 \
    --async-scheduling \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-gb 0
```

Here,

* `Qwen/Qwen3-VL-32B-Instruct` is the model name
* `--port 9080`: the port to run the server on
* `--max-model-len 5000`: caps the prompt (text plus image tokens) and the answer at 5000 tokens. That's enough for one region crop plus the processor's 1024-token answer limit, and a small limit leaves more GPU memory for running many requests at once.
* `--gpu-memory-utilization 0.95`: lets vLLM use 95% of the GPU. Nothing else should run on that GPU at the same time, so finish segmentation steps first or use another GPU.
* `--limit-mm-per-prompt.video 0`: turns off video input. We only send images, so vLLM doesn't reserve memory for video.
* `--async-scheduling`: overlaps scheduling with model execution, which gives more throughput.
* `--mm-encoder-tp-mode data`: runs the vision encoder data-parallel. It only changes anything if you use several GPUs (`--tensor-parallel-size N`).
* `--mm-processor-cache-gb 0`: turns off the image preprocessing cache. Every region crop is different, so the cache would never be hit.

`Ctrl + A then D` to detach the screen and wait until the server is ready:

```bash
until curl -sf http://localhost:9080/v1/models > /dev/null; do
    echo "waiting for vLLM..."; sleep 10
done
curl http://localhost:9080/v1/models
```

#### Run the OCR step

```bash
conda run --no-capture-output -n ecpo_non_eynollah \
    bash workflows/individual/jingbao_parallel_individual.sh <path_to_workspace> \
    ocrd-vlm-ocr OCR-D-ECPO OCR-D-OCR \
    -P model Qwen/Qwen3-VL-32B-Instruct \
    -P server http://localhost:9080/v1
```