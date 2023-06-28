# Collapsing
Framework for unsupervised semantic segmentation using a novel clustering algorithm named Collapsing.

## Installation
To compile the Cython code, run the following command from src/cython:
```
python3 setup.py build_ext --inplace
```

## Usage
To run evaluation using STEGO as a feature extractor, run `eval_stego.py`. This script will extract features from a user-defined number of images, then creates meta-points for them, and finally run Collapsing to assign pseudo-labels for each pixel. The script will then evaluate the performance of Collapsing using the pseudo-labels and the ground truth labels, logging the results into W&B.

To run evaluation using DINO, DINOv2 or SAM as a feature extractor, run `eval.py`. As before, this script extracts features from a user-defined number of images using the selected model, then creates meta-points for them, and finally run Collapsing to assign pseudo-labels for each pixel. The script will then evaluate the performance of Collapsing using the pseudo-labels and the ground truth labels, logging the results into W&B.

Each script has a number of options to select, including the number of images to use, the distance metric to use (between euclidean and 1-cosine_similarity), the radius of the meta-points and the `k` for the `k`-nearest neighbors algorithm used in Collapsing. Additionally, in `eval.py` the user can select the model to use as a feature extractor, and the option to apply PCA to those features.

These scripts expect the following directory structure:

```
.
├── PREFIX/
│   ├── ADEChallengeData2016/
│   │   ├── annotations/
│   │   │   ├── training/
│   │   │   │   ├── *.png
│   │   │   ├── validation/
│   │   │   │   ├── *.png
│   │   ├── images/
│   │   │   ├── training/
│   │   │   │   ├── *.jpg
│   │   │   ├── validation/
│   │   │   │   ├── *.jpg
│   ├── meta_points/
│   ├── saved_models/
```

The `PREFIX` directory is the root directory of the dataset. The `ADEChallengeData2016` directory contains the ADE20K dataset. The `annotations` directory contains the ground truth labels for the training and validation sets. The `images` directory contains the images for the training and validation sets. The `meta_points` directory is an empty directory that will contain .png files where each pixel value represent the meta_point id representing that pixel. The `saved_models` directory contains the saved models for the feature extractors.