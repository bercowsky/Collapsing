import os
from os.path import join
import wget
from PIL import Image
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
import time

# from torch import inf
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import wandb

import clustering
from eval_utils import (
    CustomDataset,
    get_class_labels,
    map_pseudo_to_label,
    measure_from_confusion_matrix
)


sys.path.append("STEGO/src")

from STEGO.src.train_segmentation import LitUnsupervisedSegmenter


LOCAL = True
if LOCAL:
    PREFIX = '/Users/andresbercowskyrama/Desktop/UNI/TFG/Datasets/'
else:
    PREFIX = '/cluster/home/abercowsky/data/'

# Constants
BATCH_SIZE = 1

DATASET_DIR     = os.path.join(PREFIX, 'ADEChallengeData2016/')
REPS_DIR        = os.path.join(PREFIX, 'representatives/')
COLLAPSING_PATH = os.path.join(PREFIX, 'labels/collapsing')
KMEANS_PATH     = os.path.join(PREFIX, 'labels/kmeans')
LABEL_DIR       = os.path.join(DATASET_DIR, 'annotations/training')


# WandB sweep config
wandb.init(project="collapsing-ade20k")
wandb.config.update({
    'radius': 0.07,
    'k_neighbors': 6,
})


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


def plot_confusion_matrix(confusion_matrix, name, add_counts=False):
    # Normalize the confusion matrix
    normalized_matrix = confusion_matrix / (np.sum(confusion_matrix, axis=0, keepdims=True) + 1e-8)
    n_clusters = normalized_matrix.shape[0]
    n_true_classes = normalized_matrix.shape[1]
    # plot the confusion matrix
    plt.imshow(normalized_matrix, cmap='Blues', aspect='auto')
    plt.title('Confusion Matrix')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.colorbar()

    # add tick marks for the classes
    tick_marks_x = np.arange(n_true_classes)
    tick_marks_y = np.arange(n_clusters)
    # plt.xticks(tick_marks_x, range(n_true_classes))
    # plt.yticks(tick_marks_y, range(n_clusters))

    # add the counts to the plot
    if add_counts:
        thresh = normalized_matrix.max() / 2.
        for i in range(n_clusters):
            for j in range(n_true_classes):
                plt.text(j, i, int(normalized_matrix[i, j]),
                        ha="center", va="center",
                        color="white" if normalized_matrix[i, j] > thresh else "black")
                
    plt.imshow()

    # # Convert the plot to an image
    # image = Image.frombytes("RGBA", plt.gcf().canvas.get_width_height(), plt.gcf().canvas.tostring_rgb())

    # # Log the confusion matrix plot as an image to wandb
    # wandb.log({name: wandb.Image(image)})



def process_batch(model, img, meta_points, config):
    # Query model and pass results through CRF
    with torch.no_grad():
        code1 = model(img)
        code2 = model(img.flip(dims=[3]))
        code  = (code1 + code2.flip(dims=[3])) / 2
        print("Code shape:", code.shape)
        code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)
        linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
        cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

        single_img = img[0].cpu()
        # linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
        # cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)

    features = np.array(code[0].permute(1,2,0).reshape(-1, 90))
    if len(meta_points) == 0:
        meta_points, reps = clustering.create_meta_points(features, config.radius)
    else:
        meta_points, reps = clustering.update_meta_points(meta_points, features, config.radius)
    return reps, meta_points


# def apply_crf(labels_meta_points, img):
#     # Find the unique labels present in the input
#     unique_labels = np.unique(labels_meta_points)

#     # Map the unique labels to a 0-n label range
#     label_mapping = {label: i for i, label in enumerate(unique_labels)}

#     mapped_labels = np.array([label_mapping[label] for label in labels_meta_points.flatten()]).reshape((448, 448))

#     # Find the number of unique labels after mapping
#     num_labels = len(unique_labels)

#     # Create an identity matrix of size num_labels
#     identity = np.eye(num_labels, dtype=int)

#     # Convert the mapped labels to one-hot encoded format
#     one_hot = identity[mapped_labels]

#     single_img = img.cpu()
#     collapsed_pred = dense_crf(
#         single_img,
#         torch.from_numpy(one_hot.astype("float32").transpose().reshape(one_hot.shape[-1], 448, 448))
#     ).argmax(0)

#     # Vectorize the label mapping and apply it to each element of collapsed_pred
#     vectorized_mapping = np.vectorize(lambda label: list(label_mapping.keys())[list(label_mapping.values()).index(label)])
#     original_labels = vectorized_mapping(collapsed_pred.flatten()).reshape((448, 448)).transpose()

#     return original_labels


def inference(config=None):
   # with wandb.init(config=config):
    config = wandb.config
    # config = {
    #     'k_neighbors': 3,
    #     'radius': 0.9,
    # }
    num_iters = 100

    # Create the dataset
    dataset = CustomDataset(DATASET_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    meta_points = np.empty(0, dtype=object)
    current_meta_point = np.empty(0, dtype=object)
    num_meta_points = []
    times_meta_points = []
    for batch_idx, samples in tqdm(enumerate(dataloader), total=num_iters):
        if batch_idx == num_iters:
            break
        start_time = time.time()
        reps, current_meta_point = process_batch(model, samples[0], current_meta_point, config)
        reps += len(meta_points)
        total_time = time.time() - start_time
        times_meta_points.append(total_time)
        img = Image.fromarray(reps.astype(np.uint16))
        img.save(os.path.join(COLLAPSING_PATH, samples[2][0].split('.')[0] + '.png'), bitdepth=16, format='PNG')
        num_meta_points.append(len(current_meta_point) + len(meta_points))

        if total_time >= 1 or batch_idx == num_iters - 1:
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

    print("Confusion matrix collapsing shape:", confusion_matrix_collapsing.shape)
    print("Confusion matrix collapsing K-Means:", confusion_matrix_kmeans.shape)
    print("labels_meta_points:", labels_meta_points.shape)
    print("labels_kmeans:", labels_kmeans.shape)
    print("reps shape:", reps.shape)
    for batch_idx, samples in tqdm(enumerate(dataloader), total=num_iters):
        if batch_idx == num_iters:
            break
        reps = np.asarray(Image.open(os.path.join(COLLAPSING_PATH, samples[2][0].split('.')[0] + '.png')), dtype=np.uint16)
        for i in range(samples[1][0][0].shape[0]):
            for j in range(samples[1][0][0].shape[1]):
                confusion_matrix_collapsing[labels_meta_points[reps[i*448+j]].flatten(), samples[1][0][0][i,j]] += 1
                confusion_matrix_kmeans[labels_kmeans[reps[i*448+j]].flatten(), samples[1][0][0][i,j]] += 1

    start_time = time.time()
    measurements_collapsing = measure_from_confusion_matrix(confusion_matrix_collapsing)
    print("Measuring collapsing took {} seconds".format(time.time() - start_time))
    start_time = time.time()
    measurements_kmeans = measure_from_confusion_matrix(confusion_matrix_kmeans)
    print("Measuring kmeans took {} seconds".format(time.time() - start_time))

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
    # for image_name in images_names:
    for batch_idx, samples in enumerate(dataloader):
        if batch_idx == num_iters:
            break
        # Load image and segmentation
        image = samples[0][0].squeeze().numpy().transpose((1, 2, 0))
        reps_collapsing = np.asarray(Image.open(os.path.join(COLLAPSING_PATH, samples[2][0].split('.')[0] + '.png')), dtype=np.uint16)
        segmentation_collapsing = np.array([labels_meta_points[reps_collapsing[i]] for i in range(len(reps_collapsing))]).flatten()
        for i, pseudo in enumerate(segmentation_collapsing.flatten()):
            segmentation_collapsing[i] = map_to_label_collapsing[pseudo] if pseudo in map_to_label_collapsing else 0
        segmentation_collapsing = segmentation_collapsing.reshape((448, 448))

        segmentation_kmeans = np.array([labels_kmeans[reps_collapsing[i]] for i in range(len(reps_collapsing))]).flatten()
        for i, pseudo in enumerate(segmentation_kmeans.flatten()):
            segmentation_kmeans[i] = map_to_label_kmeans[pseudo] if pseudo in map_to_label_kmeans else 0
        segmentation_kmeans = segmentation_kmeans.reshape((448, 448))

        true_segmentation = samples[1][0].squeeze().numpy()
        # crf_segmentation = apply_crf(segmentation_collapsing.flatten(), samples[0][0].squeeze())

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
    # Can not plot confusion matrix in WandB
    # plot_confusion_matrix(confusion_matrix_collapsing, 'collapsing_confusion_matrix')
    # plot_confusion_matrix(confusion_matrix_kmeans, 'kmeans_confusion_matrix')

    # np.save(os.path.join(REPS_DIR, 'confusion_matrix.npy'), confusion_matrix)
    # np.save(os.path.join(REPS_DIR, 'labels_meta_points.npy'), labels_meta_points)

# wandb.agent(sweep_id, function=inference, count=25)
inference()
