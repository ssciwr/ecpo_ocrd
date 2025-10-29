# OCR-D Pipeline for ECPO

**Work in Progress**

## Prerequisites

* Python `==3.10` (others might work, but this is what we tested. Very recent versions do not work.)

## Installation

```bash
git clone https://github.com/ssciwr/ecpo_ocrd.git
cd ecpo_ocrd
python -m pip install .
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
