from torch.utils.data import Dataset as BaseDataset
import glob
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import cv2


class Dataset(BaseDataset):
    def __init__(self, dataset_dir, augmentation=None, preprocessing=None, mode=None):
        ## Dataloader for Regression and Classification

        self.dataset_dir = dataset_dir
        self.samples_paths = glob.glob("{}*.npy".format(self.dataset_dir))

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode

    def __getitem__(self, i):

        sample = np.load(self.samples_paths[i])

        image = (sample[:, :, :3] * 255.0).astype(np.uint8)
        inst_map = sample[:, :, 3]
        nuclear_mask = sample[:, :, 4]
        type_mask = sample[:, :, 5]

        centroid_prob_mask = self.distance_transform(sample[:, :, 3])

        if np.max(centroid_prob_mask) == 0:
            pass
        else:
            centroid_prob_mask = (centroid_prob_mask / np.max(centroid_prob_mask)) * 1.0

        mask = nuclear_mask
        mask = centroid_prob_mask

        mask = np.zeros((nuclear_mask.shape[0], nuclear_mask.shape[0], 3))
        mask[:, :, 0] = nuclear_mask
        mask[:, :, 1] = centroid_prob_mask
        mask[:, :, 2] = inst_map

        if self.augmentation:
            try:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]
            except:
                pass

        # apply preprocessing
        if self.preprocessing:
            image = image / 255.0
            sample = self.preprocessing(image=image)
            image = sample["image"]

        return image, mask

    def __len__(self):
        return len(self.samples_paths)

    def distance_transform(self, inst_mask):
        heatmap = np.zeros_like(inst_mask).astype("uint8")
        for x in np.unique(inst_mask)[1:]:
            temp = inst_mask + 0
            temp = np.where(temp == x, 1, 0).astype("uint8")

            heatmap = heatmap + cv2.distanceTransform(temp, cv2.DIST_L2, 3)

        return heatmap
