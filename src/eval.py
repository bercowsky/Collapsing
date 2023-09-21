"""Evaluation script for the DINO, DINOv2 and SAM models.

The script runs inference on the images of the ADE20K training set and
computes the mIoU, homogeneity, completeness and V-measure scores, uploading
the results to W&B. The results include images with the segmentation masks
overlaid on top of the original images.

The input to the script are the following:
    * The model to be used for inference
    * The distance metric
    * The meta-points radius
    * The k for K-Nearest Neighbors
    * The number of images to process
    * Whether to apply PCA or not (getting the first 90 principal components)

The radius is computed using the distance metric provided. For the Cosine
distance, the radius is 1 - cosine_similarity, so smaller values mean more
meta-points created.

This script assumes you follow the directory structure described bellow:

- PREFIX/
    - ADEChallengeData2016/
        - annotations/
            - training/
                - *.png
            - validation/
                - *.png
        - images/
            - training/
                - *.jpg
            - validation/
                - *.jpg
    - meta_points/
    - saved_models/
"""


import os
import time
from tqdm import tqdm

import numpy as np
from PIL import Image
from segment_anything import sam_model_registry
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import wandb

import cython.clustering as clustering
from utils_collapsing.eval_utils import (
    CustomDataset,
    get_class_labels,
    map_pseudo_to_label,
    measure_from_confusion_matrix,
)
from cython.feature_extractor import get_dino_embeddings

# Constants
# TODO: Pass as an argument to the script
BATCH_SIZE = 1
NUM_GT_CLASSES = 150

# TODO: Pass as an argument to the script
PREFIX = '/Users/andresbercowskyrama/Desktop/UNI/TFG/Datasets/'

DATASET_DIR     = os.path.join(PREFIX, 'ADEChallengeData2016/')
METAPOINTS_PATH = os.path.join(PREFIX, 'meta_points/')
LABEL_DIR       = os.path.join(DATASET_DIR, 'annotations/training')
CHECKPOINT_PATH = os.path.join(PREFIX, 'saved_models', 'sam_vit_h_4b8939.pth')

# Model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_principal_components(X, n):
    """Apply PCA to the given data and return the first n principal components."""
    pca = PCA(n_components=n)
    pca.fit(X)
    # print("Explained variance:", np.sum(pca.explained_variance_ratio_))
    return torch.from_numpy(pca.transform(X).astype(np.float32))
    

def process_batch(img, model, meta_points, config):
    """Runs inference on a batch of images and return the features.

    We run inference on the image and its horizontal flip and then average the results
    to get a more robust representation. Then, we resize the features to the original
    image size and apply PCA to the features if specified. Finally, we create the
    meta-points and return them along with their ids.
    """
    shape = img.shape[-2:]
    if config.model == 'SAM':
        with torch.no_grad():
            img = T.Resize((1024, 1024), antialias=True)(img)
            code1 = model.image_encoder(img)
            code2 = model.image_encoder(img.flip(dims=[3]))
    elif config.model.startswith('DINO'):
        with torch.no_grad():
            img = T.Resize((896, 896), antialias=True)(img)
            code1 = torch.tensor(get_dino_embeddings(DEVICE, model, img))
            code2 = torch.tensor(get_dino_embeddings(DEVICE, model, img.flip(dims=[3])))

    code = (code1 + code2.flip(dims=[3])) / 2
    # Output shape: (batch_size, dimensions, img_widht, img_height)
    code = F.interpolate(code, shape, mode='bilinear', align_corners=False)

    # Apply PCA to the features
    if config.use_pca:
        code = get_principal_components(code[0].permute(1,2,0).reshape((-1, code.shape[1])), 90).reshape((1, shape[0], shape[1], 90)).permute(0, 3, 1, 2)
    features = np.array(code[0].permute(1,2,0).reshape(-1, code.shape[1]))

    # Create meta-points
    meta_points, reps = clustering.create_meta_points(features, config.radius, config.metric)

    return reps, meta_points


def png_path_name(model_name, image_name):
    """Returns the path to the png file where the labels will be saved."""
    return os.path.join(METAPOINTS_PATH, model_name + image_name.split('.')[0] + '.png')
    

def inference(model, dataloader, config=None):
    config = wandb.config

    meta_points = np.empty(0, dtype=object)
    current_meta_point = np.empty(0, dtype=object)
    for batch_idx, samples in tqdm(enumerate(dataloader), total=config.num_images):
        if batch_idx == config.num_images:
            break

        reps, current_meta_point = process_batch(samples[0], model, current_meta_point, config)
        # Shift the representative ids by the number of meta points already created
        reps += len(meta_points)

        img = Image.fromarray(reps.astype(np.uint32))
        img.save(png_path_name(config.model, samples[2][0]), bitdepth=32, format='PNG')

        meta_points = np.append(meta_points, current_meta_point)
        current_meta_point = np.empty(0, dtype=object)

    centers = np.array([mp.get_center() for mp in meta_points])

    start_time = time.time()
    kmeans = KMeans(n_clusters=min(len(centers), NUM_GT_CLASSES), n_init='auto', random_state=0).fit(centers)
    labels_kmeans = kmeans.labels_
    print("Kmeans took {} seconds".format(time.time() - start_time))

    start_time = time.time()
    mean_shift = MeanShift(bandwidth=config.bandwith).fit(centers)
    labels_mean_shift = mean_shift.labels_
    print("Mean shift took {} seconds".format(time.time() - start_time))
    print("Number of clusters of Mean Shift: {}".format(len(np.unique(labels_mean_shift))))

    # Collapse and label meta points
    start_time = time.time()
    collapsed_meta_points = clustering.collapse_meta_points(meta_points, config.k_neighbors, config.metric)
    print("Collapsing took {} seconds".format(time.time() - start_time))
    start_time = time.time()
    labels_meta_points = clustering.get_labels_meta_points2(collapsed_meta_points)
    print("Labeling took {} seconds".format(time.time() - start_time))

    n_clusters = len(np.unique(labels_meta_points))
    confusion_matrix_collapsing = np.zeros((n_clusters, NUM_GT_CLASSES+1))
    confusion_matrix_kmeans = np.zeros((NUM_GT_CLASSES+1, NUM_GT_CLASSES+1))
    confusion_matrix_meanshift = np.zeros((len(np.unique(labels_mean_shift)), NUM_GT_CLASSES+1))
    for batch_idx, samples in tqdm(enumerate(dataloader), total=config.num_images):
        if batch_idx == config.num_images:
            break
        reps = np.asarray(Image.open(png_path_name(config.model, samples[2][0])), dtype=np.uint32)
        for i in range(samples[1][0][0].shape[0]):
            for j in range(samples[1][0][0].shape[1]):
                confusion_matrix_collapsing[labels_meta_points[reps[i*448+j]].flatten(), samples[1][0][0][i,j]] += 1
                confusion_matrix_kmeans[labels_kmeans[reps[i*448+j]].flatten(), samples[1][0][0][i,j]] += 1
                confusion_matrix_meanshift[labels_mean_shift[reps[i*448+j]].flatten(), samples[1][0][0][i,j]] += 1

    start_time = time.time()
    measurements_collapsing = measure_from_confusion_matrix(confusion_matrix_collapsing)
    print("Measuring collapsing took {} seconds".format(time.time() - start_time))
    start_time = time.time()
    measurements_kmeans = measure_from_confusion_matrix(confusion_matrix_kmeans)
    print("Measuring kmeans took {} seconds".format(time.time() - start_time))
    measurements_meanshift = measure_from_confusion_matrix(confusion_matrix_meanshift)

    wandb.log({
        'assigned_iou_collapsing': measurements_collapsing['assigned_iou'],
        'assigned_miou_collapsing': measurements_collapsing['assigned_miou'],
        'homogeneity_collapsing': measurements_collapsing['homogeneity'],
        'completeness_collapsing': measurements_collapsing['completeness'],
        'v_score_collapsing': measurements_collapsing['v_score'],
        'assigned_iou_kmeans': measurements_kmeans['assigned_iou'],
        'assigned_miou_kmeans': measurements_kmeans['assigned_miou'],
        'homogeneity_kmeans': measurements_kmeans['homogeneity'],
        'completeness_kmeans': measurements_kmeans['completeness'],
        'v_score_kmeans': measurements_kmeans['v_score'],
        'assigned_iou_meanshift': measurements_meanshift['assigned_iou'],
        'assigned_miou_meanshift': measurements_meanshift['assigned_miou'],
        'homogeneity_meanshift': measurements_meanshift['homogeneity'],
        'completeness_meanshift': measurements_meanshift['completeness'],
        'v_score_meanshift': measurements_meanshift['v_score'],
        'num_clusters': n_clusters,
        'num_clusters_meanshift': len(mean_shift.cluster_centers_),
        'num_meta_points': len(meta_points),
    })


    # Map label id to class name
    class_labels = get_class_labels(os.path.join(DATASET_DIR, "objectInfo150.txt"))
    map_to_label_collapsing = map_pseudo_to_label(measurements_collapsing['assignment'])
    map_to_label_kmeans = map_pseudo_to_label(measurements_kmeans['assignment'])
    map_to_label_meanshift = map_pseudo_to_label(measurements_meanshift['assignment'])
    mask_list = []
    # Send images to WandB
    for batch_idx, samples in enumerate(dataloader):
        if batch_idx == config.num_images:
            break
        # Load image and segmentation
        image = samples[0][0].squeeze().numpy().transpose((1, 2, 0))
        reps_collapsing = np.asarray(Image.open(png_path_name(config.model, samples[2][0])), dtype=np.uint32)

        # Get segmentation from collapsing
        segmentation_collapsing = np.array([labels_meta_points[reps_collapsing[i]] for i in range(len(reps_collapsing))]).flatten()
        for i, pseudo in enumerate(segmentation_collapsing.flatten()):
            segmentation_collapsing[i] = map_to_label_collapsing[pseudo] if pseudo in map_to_label_collapsing else 0
        segmentation_collapsing = segmentation_collapsing.reshape((448, 448))

        # Get segmentation from kmeans
        segmentation_kmeans = np.array([labels_kmeans[reps_collapsing[i]] for i in range(len(reps_collapsing))]).flatten()
        for i, pseudo in enumerate(segmentation_kmeans.flatten()):
            segmentation_kmeans[i] = map_to_label_kmeans[pseudo] if pseudo in map_to_label_kmeans else 0
        segmentation_kmeans = segmentation_kmeans.reshape((448, 448))

        segmentation_meanshift = np.array([labels_mean_shift[reps_collapsing[i]] for i in range(len(reps_collapsing))]).flatten()
        for i, pseudo in enumerate(segmentation_meanshift.flatten()):
            segmentation_meanshift[i] = map_to_label_meanshift[pseudo] if pseudo in map_to_label_meanshift else 0
        segmentation_meanshift = segmentation_meanshift.reshape((448, 448))

        # Get true segmentation
        true_segmentation = samples[1][0].squeeze().numpy()

        mask_list.append(wandb.Image(image, masks={
            "predictions": {
                "mask_data": segmentation_collapsing,
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": true_segmentation,
                "class_labels": class_labels
            },
            "kmeans": {
                "mask_data": segmentation_kmeans,
                "class_labels": class_labels,
            },
            "mean_shift": {
                "mask_data": segmentation_meanshift,
                "class_labels": class_labels,
            }
        }))
    wandb.log({"predictions": mask_list})

if __name__ == "__main__":
    print("Welcome to the evaluation script!")
    print("The following models are available:")
    print("1. Segment Anything (SAM).")
    print("2. DINO.")
    print("3. DINOv2.")
    print()

    # Record user input
    model_num = input("Enter the number of the model you want to use: ")
    while model_num not in ["1", "2", "3"]:
        model_num = input("Please enter a valid number: ")

    # Load model
    if model_num == "1":
        # SAM
        model_type = "vit_h"
        model = sam_model_registry[model_type](checkpoint=CHECKPOINT_PATH).to(DEVICE)
        model_name = "SAM"
    elif model_num == "2":
        # DINO
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(DEVICE)
        model_name = "DINO"
    elif model_num == "3":
        # DINOv2
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(DEVICE)
        model_name = "DINOv2"

    # Set hyperparameters
    metric = input("Set the distance metric to be used (cosine or euclidean): ")
    while metric not in ["cosine", "euclidean"]:
        metric = input("Please enter a valid distance metric: ")
    
    radius = float(input("Set the radius for the collapsing algorithm: "))
    k_neighbors = int(input("Set the number of neighbors for the collapsing algorithm: "))

    # Create the dataset
    dataset = CustomDataset(DATASET_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_iters = int(input(f"Select the number of images to process (max {len(dataset)}): "))
    use_pca = int(input("Do you want to apply PCA? (0 or 1): "))
    while use_pca not in [0, 1]:
        use_pca = int(input("Please enter a valid value (0 or 1): "))

    use_pca = bool(use_pca)

    # WandB config
    wandb.init(project="collapsing")
    wandb.config.update({
        'radius': radius,
        'k_neighbors': k_neighbors,
        'metric': metric,
        'model': model_name,
        'num_images': num_iters,
        'use_pca': use_pca,
        'bandwith': 0.75,
    })

    # Run inference
    inference(model, dataloader)
