# FOCL
Framework for unsupervised semantic segmentation using a novel clustering algorithm named FOCL.

![Segmentation results](results/figures/segmentation_result.png)


## Installation

### Clone the repository
```shell script
git clone https://github.com/bercowsky/Collapsing.git
```

### Compile Cython code
```shell script
cd src/cython
python3 setup.py build_ext --inplace
```

## Usage

### Run experiments
To run experiments using STEGO features:
```shell script
cd src
python3 eval_stego.py
```

To run experiments using DINO, DINOv2 or SAM as a feature extractor:
```shell script
cd src
python3 eval.py
```

These scripts will extract features from a user-defined number of images, then creates meta-points for them, and finally run FOCL to assign pseudo-labels for each pixel. The script will then evaluate the performance of FOCL using the pseudo-labels and the ground truth labels, logging the results into W&B.

Each script has a number of options to select, including the number of images to use, the distance metric to use (between euclidean and 1-cosine_similarity), the radius of the meta-points and the `k` for the `k`-nearest neighbors algorithm used in FOCL. Additionally, in `eval.py` the user can select the model to use as a feature extractor, and the option to apply PCA to those features.

### Folder structure
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