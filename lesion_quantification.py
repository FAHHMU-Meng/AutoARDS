import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
from lungmask import LMInferer
from scipy.ndimage import gaussian_filter


def predict_lungmask_scan(ct_array):
    inferer = LMInferer(tqdm_disable=True)

    input_image = ct_array * 1600 - 1000
    input_image = np.transpose(input_image, (2, 0, 1))
    segmentation = inferer.apply(input_image)
    return np.transpose(segmentation, (1, 2, 0))


def calculate_volume_and_distance(ct_raw, infection, thresholds, center, total_length, voxel_volume=0.8):
    results = []
    for lower, upper in thresholds:
        mask = np.where((ct_raw > lower) & (ct_raw <= upper), 1, 0) * infection
        volume = float(np.sum(mask)) * voxel_volume
        points = np.argwhere(mask > 0)
        if points.size > 0:
            distances = np.linalg.norm(points - np.array(center), axis=1)
            mean_distance = float(np.mean(distances)) / total_length if total_length > 0 else 0.0
        else:
            mean_distance = 0.0
        results.append((volume, mean_distance))
    return results


def quantify_eight_metric(ct, lesion, VOXEL_VOL=0.8):
    lung = (predict_lungmask_scan(ct) > 0.52).astype("float32")
    lesion = lesion * lung

    loc = np.array(np.where(lung > 0))
    x_max, x_min = np.max(loc[0]), np.min(loc[0])
    y_max, y_min = np.max(loc[1]), np.min(loc[1])
    z_max, z_min = np.max(loc[2]), np.min(loc[2])
    total_length = float(np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2))
    points = np.argwhere(lung > 0)
    center = points.mean(axis=0) if points.size > 0 else np.array([0.0, 0.0, 0.0])

    ct_raw = gaussian_filter(ct, sigma=3) * 1600.0 - 1000.0

    thresholds = [(-np.inf, -600), (-600, -400), (-400, -200), (-200, np.inf)]
    lung_volume = float(np.sum(lung)) * VOXEL_VOL

    stats = calculate_volume_and_distance(ct_raw, lesion, thresholds, center, total_length)

    v0, v1, v2, v3 = [t[0] / lung_volume * 100 for t in stats]
    d0, d1, d2, d3 = [t[1] for t in stats]
    metrics = [v0, v1, v2, v3, d0, d1, d2, d3]
    return metrics


def calculate_PC1(metrics, sex, age):
    volume_com = 0.1578 * metrics[0] + 0.6377 * metrics[1] + 0.6982 * metrics[2] + 0.2844 * metrics[3]
    distance_com = 0.0035 * metrics[4] + 0.0037 * metrics[5] + 0.0047 * metrics[6] + 0.0075 * metrics[7]
    meta_com = 0.0005 * sex + 0.0069 * age

    return volume_com + distance_com + meta_com
