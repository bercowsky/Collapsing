"""Evaluation script for the STEGO model.

The script runs inference on the images of the ADE20K training set and
computes the mIoU, homogeneity, completeness and V-measure scores, uploading
the results to W&B. The results include images with the segmentation masks
overlaid on top of the original images.

The input to the script are the following:
    * The distance metric
    * The meta-points radius
    * The k for K-Nearest Neighbors
    * The number of images to process

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
from os.path import join
import wget
from PIL import Image
from tqdm import tqdm
import time
import sys

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import wandb

import cython.clustering as clustering
from utils.eval_utils import (
    CustomDataset,
    get_class_labels,
    map_pseudo_to_label,
    measure_from_confusion_matrix
)

sys.path.append("STEGO/src")

from STEGO.src.train_segmentation import LitUnsupervisedSegmenter


# TODO: Pass as an argument to the script
PREFIX = '/Users/andresbercowskyrama/Desktop/UNI/TFG/Datasets/'

# Constants
BATCH_SIZE = 1
NUM_GT_CLASSES = 150

DATASET_DIR     = os.path.join(PREFIX, 'ADEChallengeData2016/')
REPS_DIR        = os.path.join(PREFIX, 'representatives/')
METAPOINTS_PATH = os.path.join(PREFIX, 'meta_points/')
LABEL_DIR       = os.path.join(DATASET_DIR, 'annotations/training')


# Download pretrained Model
os.chdir("STEGO/src")
saved_models_dir = join("..", "saved_models")
os.makedirs(saved_models_dir, exist_ok=True)

saved_model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"
saved_model_name = "cocostuff27_vit_base_5.ckpt"
if not os.path.exists(join(saved_models_dir, saved_model_name)):
    wget.download(saved_model_url_root + saved_model_name, join(saved_models_dir, saved_model_name))

# Load pretrained STEGO
model = LitUnsupervisedSegmenter.load_from_checkpoint(join(saved_models_dir, saved_model_name), map_location=torch.device('mps')).cpu()


def process_batch(model, img, meta_points, config):
    # Query model and pass results through CRF
    with torch.no_grad():
        code1 = model(img)
        code2 = model(img.flip(dims=[3]))
        code  = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

    features = np.array(code[0].permute(1,2,0).reshape(-1, 90))
    if len(meta_points) == 0:
        meta_points, reps = clustering.create_meta_points(features, config.radius, config.metric)
    else:
        meta_points, reps = clustering.update_meta_points(meta_points, features, config.radius)
    return reps, meta_points


def inference(config=None):
    config = wandb.config

    # Create the dataset
    dataset = CustomDataset(DATASET_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    meta_points = np.empty(0, dtype=object)
    current_meta_point = np.empty(0, dtype=object)
    num_meta_points = []
    times_meta_points = []
    for batch_idx, samples in tqdm(enumerate(dataloader), total=config.num_images):
        if batch_idx == config.num_images:
            break
        start_time = time.time()
        reps, current_meta_point = process_batch(model, samples[0], current_meta_point, config)
        reps += len(meta_points)
        total_time = time.time() - start_time
        times_meta_points.append(total_time)
        img = Image.fromarray(reps.astype(np.uint16))
        img.save(os.path.join(METAPOINTS_PATH, samples[2][0].split('.')[0] + '.png'), bitdepth=16, format='PNG')
        num_meta_points.append(len(current_meta_point) + len(meta_points))

        if total_time >= 1 or batch_idx == config.num_images - 1:
            meta_points = np.append(meta_points, current_meta_point)
            current_meta_point = np.empty(0, dtype=object)

    centers = np.array([mp.get_center() for mp in meta_points])

    kmeans = KMeans(n_clusters=min(len(centers), 150), random_state=0).fit(centers)
    labels_kmeans = kmeans.labels_

    # Collapse and label meta points
    start_time = time.time()
    collapsed_meta_points = clustering.collapse_meta_points(meta_points, config.k_neighbors)
    print("Collapsing took {} seconds".format(time.time() - start_time))
    start_time = time.time()
    labels_meta_points = clustering.get_labels_meta_points(collapsed_meta_points, 0)
    print("Labeling took {} seconds".format(time.time() - start_time))

    n_clusters = len(np.unique(labels_meta_points))
    confusion_matrix_collapsing = np.zeros((n_clusters, 151))
    confusion_matrix_kmeans = np.zeros((151, 151))

    for batch_idx, samples in tqdm(enumerate(dataloader), total=config.num_images):
        if batch_idx == config.num_images:
            break
        reps = np.asarray(Image.open(os.path.join(METAPOINTS_PATH, samples[2][0].split('.')[0] + '.png')), dtype=np.uint16)
        for i in range(samples[1][0][0].shape[0]):
            for j in range(samples[1][0][0].shape[1]):
                confusion_matrix_collapsing[labels_meta_points[reps[i*448+j]].flatten(), samples[1][0][0][i,j]] += 1
                confusion_matrix_kmeans[labels_kmeans[reps[i*448+j]].flatten(), samples[1][0][0][i,j]] += 1

    # Compute evaluation metrics
    measurements_collapsing = measure_from_confusion_matrix(confusion_matrix_collapsing)
    measurements_kmeans = measure_from_confusion_matrix(confusion_matrix_kmeans)

    # Log results to wandb
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
        'num_clusters': n_clusters,
        'num_meta_points': len(meta_points),
    })


    # Map label id to class name
    class_labels = get_class_labels(os.path.join(DATASET_DIR, "objectInfo150.txt"))
    map_to_label_collapsing = map_pseudo_to_label(measurements_collapsing['assignment'])
    map_to_label_kmeans = map_pseudo_to_label(measurements_kmeans['assignment'])
    mask_list = []

    # Send images to WandB
    for batch_idx, samples in enumerate(dataloader):
        if batch_idx == config.num_images:
            break
        # Load image and segmentation
        image = samples[0][0].squeeze().numpy().transpose((1, 2, 0))
        reps_collapsing = np.asarray(Image.open(os.path.join(METAPOINTS_PATH, samples[2][0].split('.')[0] + '.png')), dtype=np.uint16)
        segmentation_collapsing = np.array([labels_meta_points[reps_collapsing[i]] for i in range(len(reps_collapsing))]).flatten()
        for i, pseudo in enumerate(segmentation_collapsing.flatten()):
            segmentation_collapsing[i] = map_to_label_collapsing[pseudo] if pseudo in map_to_label_collapsing else 0
        segmentation_collapsing = segmentation_collapsing.reshape((448, 448))

        segmentation_kmeans = np.array([labels_kmeans[reps_collapsing[i]] for i in range(len(reps_collapsing))]).flatten()
        for i, pseudo in enumerate(segmentation_kmeans.flatten()):
            segmentation_kmeans[i] = map_to_label_kmeans[pseudo] if pseudo in map_to_label_kmeans else 0
        segmentation_kmeans = segmentation_kmeans.reshape((448, 448))

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
            # "crf": {
            #     "mask_data": crf_segmentation,
            #     "class_labels": class_labels,
            # },
            "kmeans": {
                "mask_data": segmentation_kmeans,
                "class_labels": class_labels,
            }
        }))
    wandb.log({"predictions": mask_list})

# wandb.agent(sweep_id, function=inference, count=25)

if __name__ == "__main__":

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
    while num_iters > len(dataset):
        num_iters = int(input(f"Please enter a number less than {len(dataset)}: "))

    # WandB config
    wandb.init(project="collapsing-ade20k")
    wandb.config.update({
        'radius': radius,
        'k_neighbors': k_neighbors,
        'metric': metric,
        'num_images': num_iters,
    })

    
    inference()
