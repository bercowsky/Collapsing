import os
from PIL import Image
from typing import Dict

from munkres import Munkres
import numpy as np
from sklearn.metrics import mutual_info_score
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# Create the dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, 'images', 'training')
        self.ann_dir = os.path.join(root_dir, 'annotations', 'training')
        self.img_names = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        transform = get_transform(448, False, "center")
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name.replace('.jpg', '.png'))
        img = Image.open(img_path)
        img = transform(img)
        ann = Image.open(ann_path)
        ann = get_transform(448, True, "center")(ann)
        return img, ann, img_name
    

class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensor()])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          T.ToTensor(),
                          normalize])


def get_class_labels(path) -> Dict[int, str]:
    class_labels = {}

    with open(path, "r") as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split("\t")
            idx = int(parts[0])
            name = parts[4]
            class_labels[idx] = name
        class_labels[0] = "unmatched"

    return class_labels


def map_pseudo_to_label(assigments):
    pseudo_to_label = {}
    for label, pseudo in assigments:
        pseudo_to_label[pseudo] = label
    return pseudo_to_label


def measure_from_confusion_matrix(cm, beta=1.0):
    cm = np.transpose(cm)
    n_classes, n_clusters = cm.shape
    m = Munkres()
    cost = (cm.max() + 1) - cm
    cost = cost.tolist()
    assigned_idx = m.compute(cost)
    iou = np.zeros(n_classes, dtype=np.float32)
    for label, cluster in assigned_idx:
        if label >= n_classes:
            continue
        iou[label] = cm[label, cluster] / (cm[label].sum() + cm[:, cluster].sum() -
                                        cm[label, cluster] + 1e-8)
    measurements = {
        'assigned_iou': iou,
        'assigned_miou': np.nanmean(iou),
        'assignment': assigned_idx,
        'confusion_matrix': cm,
    }
    # contingency matrix based sklearn metrics
    # taken from https://github.com/scikit-learn/scikit-learn/blob/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/metrics/cluster/_supervised.py#L402
    cmf = cm.astype(np.float64)
    n_total = cmf.sum()
    n_labels = cmf.sum(1)
    n_labels = n_labels[n_labels > 0]
    entropy_labels = -np.sum(
        (n_labels / n_total) * (np.log(n_labels) - np.log(n_total)))
    n_pred = cmf.sum(0)
    n_pred = n_pred[n_pred > 0]
    entropy_pred = -np.sum(
        (n_pred / n_total) * (np.log(n_pred) - np.log(n_total)))
    mutual_info = mutual_info_score(None, None, contingency=cm)
    homogeneity = mutual_info / (entropy_labels) if entropy_labels else 1.0
    completeness = mutual_info / (entropy_pred) if entropy_pred else 1.0
    if homogeneity + completeness == 0.0:
        v_measure_score = 0.0
    else:
        v_measure_score = ((1 + beta) * homogeneity * completeness /
                        (beta * homogeneity + completeness))
    measurements.update({
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_score': v_measure_score,
    })
    return measurements