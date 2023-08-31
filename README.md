# Seabot
SeaBot is an integrated machine learning pipeline designed for underwater video classification and annotation. It employs a two-pronged approach, training both an image classification model and a text generation model simultaneously. The image model classifies underwater entities, while the text model generates human-readable annotations.

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

Clone the repository and navigate to the project directory. Run the following commands to install the required dependencies:

```bash
pip install transformers torch pytorchvideo ffmpeg-python torchvision tqdm fathomnet openai wandb pafy youtube-dl
```

## Structure

The project is structured as follows:

1. **Unsupervised Training**: Train a generator to create instances of the distribution of annotation text.
2. **Video Classification**: Train the video classification model on the distribution of the annotation imagery.
3. **Pipeline Integration**: Combine both methods into a singular pipeline and then use the actual annotations to derive results.

## Usage

### Fine-tuning Image Vision Model

```python
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
# ... (rest of the imports)
```

Run the following command to start the fine-tuning process:

```bash
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

For more details, refer to the [Colab Notebook](https://colab.research.google.com/drive/1Gcth1dGuMimPLkRt3jvYn7MTNUUg4rf0).
