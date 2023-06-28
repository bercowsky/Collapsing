import torch
import numpy as np


def dino_color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    """
    Normalize the image by subtracting mean and dividing by std
    """
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

def get_dino_embeddings(device, pretrained_model, images):
    """
    Extract embeddings for a batch of images using DINO model
    """
    all_features = []
    batch_size = len(images)
    patch_size = pretrained_model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]

    # Preprocess the images
    image_tensors = images.to(device)

    # Run inference
    out = pretrained_model.get_intermediate_layers(image_tensors, n=1)[0]
    h = int(image_tensors.shape[2] / patch_size)
    w = int(image_tensors.shape[3] / patch_size)
    if patch_size == 16:
        out = out[:, 1:, :]  # discard the [CLS] token
    features = out.reshape(batch_size, h*w, out.shape[-1])

    # Resize the features
    features = features.permute(0, 2, 1) # shape (batch_size, feature_dim, num_patches)
    features = features.reshape(batch_size, out.shape[-1], h, w) # shape (batch_size, feature_dim, h, w)
    features = features.cpu().detach().numpy()

    # Check the shape
    assert features.shape[0] == batch_size

    return features
