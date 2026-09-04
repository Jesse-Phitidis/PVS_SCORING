# Multi-task learning for the automatic grading of enlarged perivascular space burden using MRI

This repository contains the code for the paper:

> _To be added_

## Installation

Create the conda environment from the provided environment definition:

```bash
conda env create -f env.yaml
conda activate PVS_SCORING
```

Then install the project in editable mode:

```bash
pip install -e .
```

## Project Structure

- `src/pvs_scoring/` - Main package source code
  - `run.py` - Training entry point using PyTorch Lightning CLI
  - `data_modules/` - Data loading and preprocessing
  - `lightning_modules/` - Model implementations (classifier, segmentor, multitask)
- `configs/` - Configuration files for different model variants
  - `classifier/` - Classification model configs
  - `segmentor/` - Segmentation model configs
  - `multitask/` - Multitask learning model configs
- `notebooks/` - Jupyter notebooks for data preparation and analysis
- `data_new/` - Dataset storage (images, labels, metadata, splits)

## Run Training

To perform a PyTorch Lightning training run, create or select a config file and run:

```bash
pvs_scoring_cli fit --config configs/path_to_config.yaml
```

For example, to train a classifier:

```bash
pvs_scoring_cli fit --config configs/classifier/images_s1.yaml
```

To train a segmentor:

```bash
pvs_scoring_cli fit --config configs/segmentor/dynunet.yaml
```

To train the multitask model:

```bash
pvs_scoring_cli fit --config configs/multitask/multitask_const_050_s1.yaml
```

For additional training options and customization, refer to the PyTorch Lightning CLI documentation.
