from skimage import data, io, util
from matplotlib import pyplot as plt

import scipy.io as sio
import numpy as np

import glob
import cv2
import os
import shutil
import config


def create_centroid_mask(centroids, height, width):
    cent_img = np.zeros((height, width))
    cent_img[tuple(centroids.T)] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    centroids_mask = cv2.dilate(cent_img, kernel, iterations=2)

    return centroids_mask


def read_image(imagePath):
    image = io.imread(imagePath)[:, :, :3]

    return image


def fix_type(type_map):
    type_map[(type_map == 3) | (type_map == 4)] = 3
    type_map[(type_map == 5) | (type_map == 6) | (type_map == 7)] = 4
    return type_map


def read_annotation(annotPath):
    mask = sio.loadmat(annotPath)

    type_map = fix_type(mask["type_map"])
    centroid_map = create_centroid_mask(
        mask["inst_centroid"].astype(int)[:, ::-1],
        mask["inst_map"].shape[0],
        mask["inst_map"].shape[1],
    )
    inst_map = mask["inst_map"]
    nuclear_map = inst_map + 0
    nuclear_map[nuclear_map != 0] = 1

    return centroid_map, inst_map, nuclear_map, type_map


test_folder = glob.glob(os.path.join(config.dataset_raw, "Test/Images"))[0]
train_folder = glob.glob(os.path.join(config.dataset_raw, "Train/Images"))[0]

try:
    shutil.rmtree(config.train_dir)
except:
    pass
try:
    shutil.rmtree(config.test_dir)
except:
    pass

if not os.path.exists(config.train_dir):
    os.makedirs(config.train_dir)
if not os.path.exists(config.test_dir):
    os.makedirs(config.test_dir)

for img_path in glob.glob("{}/*".format(test_folder)):
    print(img_path)
    image = read_image(img_path)
    centroid_map, inst_map, nuclear_map, type_map = read_annotation(
        img_path.replace("Images", "Labels").replace(".png", ".mat"))

    output_mask = np.zeros((image.shape[0], image.shape[1], 7))
    output_mask[:, :, :3] = image / 255.0
    output_mask[:, :, 3] = inst_map  # inst_mask
    output_mask[:, :, 4] = nuclear_map  # binary mask
    output_mask[:, :, 5] = centroid_map  # centroids_mask
    output_mask[:, :, 6] = type_map  # type mask

    output = util.view_as_windows(output_mask, (256, 256, 7),
                                  step=(245, 245, 7)).reshape(-1, 256, 256, 7)

    for idx in range(output.shape[0]):
        np.save(
            "{}/{}_{}.npy".format(
                config.test_dir,
                os.path.basename(img_path).replace(".png", ""), idx),
            output[idx],
        )

for img_path in glob.glob("{}/*".format(train_folder)):
    print(img_path)

    image = read_image(img_path)
    centroid_map, inst_map, nuclear_map, type_map = read_annotation(
        img_path.replace("Images", "Labels").replace(".png", ".mat"))

    output_mask = np.zeros((image.shape[0], image.shape[1], 7))
    output_mask[:, :, :3] = image / 255.0
    output_mask[:, :, 3] = inst_map  # inst_mask
    output_mask[:, :, 4] = nuclear_map  # binary mask
    output_mask[:, :, 5] = centroid_map  # centroids_mask
    output_mask[:, :, 6] = type_map  # type mask

    output = util.view_as_windows(output_mask, (256, 256, 7),
                                  step=(128, 128, 7)).reshape(-1, 256, 256, 7)

    for idx in range(output.shape[0]):
        np.save(
            "{}/{}_{}.npy".format(
                config.train_dir,
                os.path.basename(img_path).replace(".png", ""), idx),
            output[idx],
        )
