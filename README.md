# Seabot
SeaBot is an integrated machine learning pipeline designed for underwater video classification and annotation. It employs a two-pronged approach, training both an image classification model and a text generation model simultaneously. The image model classifies underwater entities, while the text model generates human-readable annotations.

## Current Development Objectives
- [ ] Fully integrate SeaTube data as pretraining data.
- [ ] Add object detection to image classification.

## Image and Video Classification with Text Generation

## Overview

SeaBot is a machine learning pipeline that trains both an image classification model and a text generation model simultaneously. The project aims to classify underwater videos and generate human-readable annotations.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Structure](#structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Requires: sudo apt install ffmpeg

Clone the repository and navigate to the project directory. Run the following commands to install the required dependencies:

```bash
pip install transformers torch pytorchvideo ffmpeg-python torchvision tqdm fathomnet openai wandb pafy youtube-dl
```

## Structure

The project is structured as follows:

1. **Self-supervised Pre-training**: Train a generator to correct modified images as a form of pretraining.
2. **Supervised Pre-training**: Train a generator to correctly classify images that are from fathomnet.
3. **Video Classification**: Train the video classification model on the distribution of the annotation imagery.
4. **Pipeline Integration**: Combine both methods into a singular pipeline and then use the actual annotations to derive results.

## Usage

### Fine-tuning Image Vision Model

Run the following command to start the fine-tuning process:

```bash
install.sh
python pre_training.py
python fine_tuning.py
```

### Fathomnet Fine-Tuning

```python
# Define the custom dataset class for handling FathomNet data
class FathomNetDataset(Dataset):
    # ... (rest of the code)
```

Run the following command to start the Fathomnet fine-tuning process:

```bash
python fathomnet_fine_tuning.py
```

### Classify and Humanize Outputs

```python
# Make sure you have the right API key
openai.api_key = ''
# ... (rest of the code)
```

Run the following command to start the classification and humanization process:

```bash
python classify_and_humanize.py
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

