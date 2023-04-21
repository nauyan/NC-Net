import glob
from skimage import io
import scipy.io as sio
from skimage import data, io, util
from matplotlib import pyplot as plt
import numpy as np

import cv2
import os
from tqdm import tqdm
import shutil


def create_centroid_mask(centroids, height, width):
    cent_img = np.zeros((height, width))
    cent_img[tuple(centroids.T)] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    centroids_mask = cv2.dilate(cent_img, kernel, iterations=2)

    return centroids_mask


def rescale_sample(sample):
    sample = rescale(sample, 2, anti_aliasing=False, multichannel=True)
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


def get_inst_centroid(inst_map):
    """Get instance centroids given an input instance map.
    Args:
        inst_map: input instance map
    
    Returns:
        array of centroids
    
    """
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            int(inst_moment["m10"] / inst_moment["m00"]),
            int(inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


def process_fold(images, masks, output_dir, fold):

    for idx in tqdm(range(images.shape[0]), total=images.shape[0]):
        image = images[idx]
        mask = masks[idx]

        try:
            centroids = get_inst_centroid(np.sum(mask[:, :, :5], axis=2))
            centroid_map = create_centroid_mask(centroids[:, ::-1],
                                                mask.shape[0], mask.shape[1])
            inst_map = np.sum(mask[:, :, :5], axis=2)
            nuclear_map = ((mask[:, :, 5] * -1) + 1).astype(int)
            type_map = np.argmax(mask[:, :, :5], axis=2) + 1
            type_map[type_map == 7] = 0


#             return
        except:
            continue

        output_mask = np.zeros((image.shape[0], image.shape[1], 7))
        output_mask[:, :, :3] = image / 255.0
        output_mask[:, :, 3] = inst_map  # inst_mask
        output_mask[:, :, 4] = nuclear_map  # binary mask
        output_mask[:, :, 5] = centroid_map  # centroids_mask
        output_mask[:, :, 6] = type_map  # type mask

        #         show(inst_map)
        #         show(nuclear_map)
        #         show(centroid_map)

        #         print(np.unique(inst_map))
        #         print(np.unique(nuclear_map))
        #         print(np.unique(centroid_map))

        #         break
        np.save("{}/{}_{}.npy".format(output_dir, fold, idx), output_mask)


def init_dir(input_dir):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    else:
        shutil.rmtree(input_dir)
        os.makedirs(input_dir)


img_paths = glob.glob("../datasets/panNuke/*/*/*/*")

imgs_fold1 = img_paths[2]
labels_fold1 = img_paths[0]

imgs_fold2 = img_paths[8]
labels_fold2 = img_paths[6]

imgs_fold3 = img_paths[5]
labels_fold3 = img_paths[3]

print(imgs_fold1, labels_fold1)
print(imgs_fold2, labels_fold2)
print(imgs_fold3, labels_fold3)

train_dir = "data/pan1/train"
test_dir = "data/pan1/test"

init_dir(train_dir)
init_dir(test_dir)

print("Processing Fold1")
images = np.load(imgs_fold1)
masks = np.load(labels_fold1)
process_fold(images, masks, train_dir, "fold1")

print("Processing Fold2")
images = np.load(imgs_fold2)
masks = np.load(labels_fold2)
process_fold(images, masks, train_dir, "fold2")

print("Processing Fold3")
images = np.load(imgs_fold3)
masks = np.load(labels_fold3)
process_fold(images, masks, test_dir, "fold3")

train_dir = "data/pan2/train"
test_dir = "data/pan2/test"

init_dir(train_dir)
init_dir(test_dir)

print("Processing Fold1")
images = np.load(imgs_fold1)
masks = np.load(labels_fold1)
process_fold(images, masks, train_dir, "fold1")

print("Processing Fold2")
images = np.load(imgs_fold2)
masks = np.load(labels_fold2)
process_fold(images, masks, test_dir, "fold2")

print("Processing Fold3")
images = np.load(imgs_fold3)
masks = np.load(labels_fold3)
process_fold(images, masks, train_dir, "fold3")

train_dir = "data/pan3/train"
test_dir = "data/pan3/test"

init_dir(train_dir)
init_dir(test_dir)

print("Processing Fold1")
images = np.load(imgs_fold1)
masks = np.load(labels_fold1)
process_fold(images, masks, test_dir, "fold1")

print("Processing Fold2")
images = np.load(imgs_fold2)
masks = np.load(labels_fold2)
process_fold(images, masks, train_dir, "fold2")

print("Processing Fold3")
images = np.load(imgs_fold3)
masks = np.load(labels_fold3)
process_fold(images, masks, train_dir, "fold3")
