# Revised Version for Training on Kaggle with My Document Dataset: Lossless Image Compression through Super-Resolution

## [[Paper]](https://arxiv.org/abs/2004.02872) ##

<div align="center">
  <img src="figs/concept_fig.png" />
</div>

## Citation
```bibtex
@article{cao2020lossless,
  title={Lossless Image Compression through Super-Resolution},
  author={Cao, Sheng and Wu, Chao-Yuan and Kr{"a}henb{"u}hl, Philipp},
  year={2020},
  journal={arXiv preprint arXiv:2004.02872},
}
```

If you use this codebase, please also consider citing [L3C](https://github.com/fab-jul/L3C-PyTorch#citation).

## Overview
This repository provides an adapted implementation of **SReC (Super-Resolution-based Lossless Image Compression)** for training on **Kaggle** with a **document dataset**. SReC formulates lossless compression as a super-resolution problem, leveraging deep learning models to achieve state-of-the-art compression rates.

This version is tailored to handle **document images**, optimizing for structured text-based content while maintaining high compression efficiency.

Training, compression, and decompression are fully supported and open-sourced.

## Getting Started
Follow these steps to set up and run the modified version:

1. [Install the necessary dependencies](INSTALL.md)
2. Prepare your **document dataset** for training and evaluation.
3. Run the training process on **Kaggle** using the modified pipeline.
4. Perform compression and decompression using the trained model.


## Training
Ensure you are in the top-level directory before running the training script.
```
python3 -um src.train \
  --train-path "path to directory of training document images" \
  --train-file "list of filenames of training images, one filename per line" \
  --eval-path "path to directory of evaluation document images" \
  --eval-file "list of filenames of evaluation images, one filename per line" \
  --plot "directory to store model output" \
  --batch "batch size"
```

Training images must follow the format `train-path/filename` as specified in the train-file.

For document-based datasets, hyperparameters might need adjustment to optimize text preservation.

To train using specific settings, such as lower-resolution crops, use:
```
python3 -um src.train \
  --train-path "path to training document images" \
  --train-file "list of training images" \
  --eval-path "path to evaluation document images" \
  --eval-file "list of evaluation images" \
  --plot "output directory" \
  --batch "batch size" \
  --epochs 10 \
  --lr-epochs 1 \
  --crop 64
```
Run `python3 -um src.train --help` for a list of tunable hyperparameters.

## Evaluation
To evaluate compression efficiency, use:
```
python3 -um src.eval \
  --path "path to evaluation document images" \
  --file "list of evaluation images" \
  --load "path to model weights"
```

## Compression/Decompression

To compress a directory of **document images**, use:
```
python3 -um src.encode \
  --path "path to document images" \
  --file "list of filenames of images" \
  --save-path "directory to save .srec files" \
  --load "path to model weights"
```
To decompress and restore original document images:
```
python3 -um src.decode \
  --path "path to .srec document images" \
  --file "list of filenames of .srec images" \
  --save-path "directory to save restored PNG files" \
  --load "path to model weights"
```

## Dataset Preparation
### Document Dataset
If you are using a custom document dataset, ensure it follows a structured format, with separate training and validation sets.

For **validation images**, provide a properly formatted list of file paths.
For **training images**, structure the dataset accordingly and prepare annotation files if necessary.

## Acknowledgments
This modified version builds upon the original [L3C](https://github.com/fab-jul/L3C-PyTorch) and incorporates techniques from prior research.

Special thanks to:
- [Fabian Mentzer](https://github.com/fab-jul) for foundational work on L3C and contributions to lossless image compression.
- The **original authors of SReC** for their pioneering research in using super-resolution for lossless image compression.

---
This repository is an extension of **SReC**, adapted for **document image compression** with Kaggle compatibility.

