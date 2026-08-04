# Individual Identification

This is a project for the Pattern Recognition course at the Jagiellonian University in Kraków.
The goal of the project is to implement an individual identification system for the Czech Lynx dataset.

The dataset is available at Kaggle: [Czech Lynx Dataset](https://www.kaggle.com/datasets/picekl/czechlynx).
In total, it contains ~40000 images from ~19000 observations of 319 different individual lynx.

## Report

The final report for this project can be found [here](report/main.pdf).

## Setup

Using pip and virtual environment, run:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or

Using NixOS flakes, run:
```bash
nix develop
```

## Set up wandb

Get your API key from https://wandb.ai/authorize and set it as an environment variable:
```bash
export WANDB_API_KEY='your-api-key-here'
```

or add it to your .env file:
```bash
WANDB_API_KEY='your-api-key-here'
```

## Run the tests

To run the tests, run:
```bash
pytest -v
```