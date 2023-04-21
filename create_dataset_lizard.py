import glob
from skimage import io
import scipy.io as sio
from skimage import data, io, util
from matplotlib import pyplot as plt
import numpy as np

import cv2
import os
import random
from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm
import shutil
import pandas as pd


def create_centroid_mask(centroids, height, width):
    cent_img = np.zeros((height, width))
    cent_img[tuple(centroids.T)] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    centroids_mask = cv2.dilate(cent_img, kernel, iterations=2)

    return centroids_mask


def rescale_sample(sample):
    sample = rescale(sample,
                     2,
                     anti_aliasing=False,
                     multichannel=True,
                     preserve_range=True,
                     order=0)
    return sample


def read_image(imagePath):
    image = io.imread(imagePath)[:, :, :3]
    return image


def read_annotation(annotPath):
    mask = sio.loadmat(annotPath)

    centroid_map = create_centroid_mask(mask['centroid'].astype(int)[:, ::-1],
                                        mask['inst_map'].shape[0],
                                        mask['inst_map'].shape[1])
    inst_map = mask['inst_map']
    nuclear_map = inst_map + 0
    nuclear_map[nuclear_map != 0] = 1

    return centroid_map, inst_map, nuclear_map


def init_dir(input_dir):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    else:
        shutil.rmtree(input_dir)
        os.makedirs(input_dir)


meta = pd.read_csv("../datasets/lizard/Lizard_Labels/info.csv")
# print(meta.columns)
for fold_idx in range(1, 4):
    print("Generating Fold", fold_idx)

    train_dir = "data/liz{}/train".format(fold_idx)
    test_dir = "data/liz{}/test".format(fold_idx)

    init_dir(train_dir)
    init_dir(test_dir)

    for idx in tqdm(range(meta.shape[0]), total=meta.shape[0]):

        image = read_image(
            glob.glob("../datasets/lizard/Lizard_Images*/{}.png".format(
                meta['Filename'][idx]))[0])

        centroid_map, inst_map, nuclear_map = read_annotation(
            "../datasets/lizard/Lizard_Labels/Labels/{}.mat".format(
                meta['Filename'][idx]))
        type_map = np.zeros((image.shape[0], image.shape[1]))

        output_mask = np.zeros((image.shape[0], image.shape[1], 7))
        output_mask[:, :, :3] = image / 255.0
        output_mask[:, :, 3] = inst_map  # inst_mask
        output_mask[:, :, 4] = nuclear_map  # binary mask
        output_mask[:, :, 5] = centroid_map  # centroids_mask
        output_mask[:, :, 6] = type_map  # type mask

        output_mask = rescale_sample(output_mask)
        output = util.view_as_windows(output_mask, (256, 256, 6),
                                      step=(245, 245,
                                            6)).reshape(-1, 256, 256, 6)

        for p_idx in range(output.shape[0]):
            if fold_idx == meta['Split'][idx]:
                np.save("{}/{}_{}.npy".format(test_dir, meta['Filename'][idx],
                                              p_idx),
                        output[p_idx])  #.astype('int32')
            else:
                np.save("{}/{}_{}.npy".format(train_dir, meta['Filename'][idx],
                                              p_idx),
                        output[p_idx])  #.astype('int32')
