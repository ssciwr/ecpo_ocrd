# OCR-D Pipeline for ECPO

**Work in Progress**

## Prerequisites

* Python `==3.10` (others might work, but this is what we tested. Very recent versions do not work.)

## Pipeline overview

Our pipeline consists of the following steps:

* `eynollah-inference`: Run Eynollah inference to get layout segmentation results from a fine-tuned Eynollah model. See [this training/fine-tuning instruction](https://github.com/ssciwr/ecpo-eynollah/blob/main/eynollah_training.md) for details.
* `ecpo-segment`: Refine the layout segmentation results from Eynollah using PaddleOCR's layout analysis model.
* `ecpo-ocr`: Run OCR on the text regions obtained from the previous step using VLLM.

## Installation
1. Clone the repository

```bash
git clone https://github.com/ssciwr/ecpo_ocrd.git
cd ecpo_ocrd
```

2. To install the repository's dependencies, there are two options, please choose just one of them:

* `eynollah`: for running Eynollah inference
* `non-eynollah`: for running non-Eynollah processes (e.g. PaddleOCR)

For example, to install the dependencies for running the Eynollah inference part, use:

```bash
python -m pip install .[eynollah]
```

3. Then install relevant OCR-D tools with:

```bash
ecpo_ocrd install
```


## Usage

Before doing anything, add the `bin` folder to `PATH`:

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
../workflows/jingbao.sh
```

## Running the pipeline with GPUs

Dependencies for Eynollah inference and PaddleOCR are mutually exclusive due to `numpy` version.

To utilize GPU for Eynollah process, the following dependencies should be installed:

* `Python` 3.10
* `CUDA` 11.8
* `cuDNN` 8.6
* `TensorFlow` 2.12.0

So far, we can only create a `conda` environment with the above settings. Therefore, even though we already created `envs` and `bin` folders with `ecpo_ocrd install`, we still need to use `conda` environments to run Eynollah-related processes.

A quick-and-dirty solution to run the whole pipeline with GPU is to create two `conda` environments, one for Eynollah and one for the rest of the pipeline.

### Eynollah conda environment

1. Use [this instruction](https://github.com/ssciwr/ecpo-eynollah/blob/main/eynollah_training.md#step-by-step-installation) to setup a `conda` environment, named e.g. `ecpo_eynollah`, with the specified Python, CUDA, cuDNN, and TensorFlow versions.

2. Install the Eynollah-related dependencies in this environment with:

```bash
pip install .[eynollah]
```

### Non-Eynollah conda environment

Create the second conda environment and install dependencies for the rest of the pipeline with:

```bash
conda create -n ecpo_non_eynollah python=3.10
conda activate ecpo_non_eynollah
pip install .[non-eynollah]
```

### Run the parallel script

After setting up the two conda environments, move to the root folder of the repository (`ecpo_ocrd`) and run the parallel script up to the `ecpo-segment` step with:

```bash
# assuming you are in the repository root
./workflows/jingbao_parallel_eynollah_paddle.sh <path_to_workspace>
```

To run the last step `ecpo-ocr` with VLLM: TBU.