# crop-segmentation

A multiclass semantic segmentation problem to identify 12 patterns (e.g., weeds, nutrient deficiency) from aerial images. Data from the Agriculture-Vision 2022 Challenge: https://www.agriculture-vision.com/agriculture-vision-2022/prize-challenge-2022

## Installation

### Prerequisites

Poetry: https://python-poetry.org/docs/#installation

### Clone the repository: 

    git clone https://github.com/dthuff/crop-segmentation

### and install dependencies:

    cd crop-segmentation/
    poetry install

# Usage
Set desired training parameters in a `.yml` config. An example is provided in `configs/train_config.yml`

Run training on GPU `cuda:0` with:

    poetry run python run_training.py --config /path/to/configs/train_config.yml --device 0
