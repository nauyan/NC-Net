from src.dataloader import Dataset
from src.augmentation import get_preprocessing

from src.stats_util import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
    pair_coordinates,
)
import config

import torch
import numpy as np

from scipy import ndimage
from scipy.ndimage import label
from skimage.segmentation import watershed

from tqdm import tqdm


def get_stats(inst_map, pred_inst_map, print_img_stats=True):
    true = inst_map.astype("int32")

    pred = pred_inst_map.astype("int32")

    # to ensure that the instance numbering is contiguous
    pred = remap_label(pred, by_size=False)
    true = remap_label(true, by_size=False)

    try:
        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        dice_score = get_dice_1(true, pred)
        aji = get_fast_aji(true, pred)
        aji_plus = get_fast_aji_plus(true, pred)

        metrics[0].append(dice_score)
        metrics[1].append(aji)
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq
        metrics[5].append(aji_plus)
    except:
        pass

    if print_img_stats:
        for scores in metrics:
            print("%f " % scores[-1], end="  ")
        print()


def apply_watershed(distance_transform=None,
                    prob_map=None,
                    foreground_mask=None):
    marker = label(prob_map)[0]
    pred_inst_map = watershed(distance_transform,
                              markers=marker,
                              mask=foreground_mask,
                              compactness=0.01)
    return pred_inst_map


dataset = Dataset(config.test_dir, preprocessing=get_preprocessing(None))

model = torch.load("./{}/{}".format(config.checkpoints_dir,
                                    config.inference_weights))
model.eval()

metrics = [[], [], [], [], [], []]
for idx in tqdm(range(len(dataset)), total=len(dataset)):
    train_image, mask = dataset[idx]
    if int(np.unique(mask[:, :, 2]).shape[0]) == 1:
        continue
    train_image = torch.from_numpy(train_image).to(config.device).unsqueeze(0)

    model.eval()
    pred_mask = model(train_image).squeeze().cpu().detach().numpy()

    nuclei_map = pred_mask[:2, :, :].argmax(axis=0)
    prob_map = pred_mask[2, :, :]

    temp_prob_map = prob_map + 0
    temp_nuclei_map = nuclei_map + 0

    temp_prob_marker = np.where(temp_prob_map > config.watershed_threshold, 1,
                                0)
    temp_inst_map = mask[:, :, 2]
    pred_inst_map = apply_watershed(
        distance_transform=temp_prob_map,
        prob_map=temp_prob_marker,
        foreground_mask=temp_nuclei_map,
    )
    try:
        get_stats(temp_inst_map, pred_inst_map, print_img_stats=False)
    except:
        pass

metrics = np.array(metrics)
metrics_avg = np.mean(metrics, axis=-1)
np.set_printoptions(formatter={"float": "{: 0.5f}".format})
print(metrics_avg)
