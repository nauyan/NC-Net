import torch
import albumentations as albu

# from tiatoolbox.tools.stainaugment import StainAugmentor#StainAugmentaiton
import numpy as np


def get_training_augmentation():
    stain_matrix = np.array(
        [[0.91633014, -0.20408072, -0.34451435], [0.17669817, 0.92528011, 0.33561059]]
    )

    train_transform = [
        #         albu.OneOf(
        #             [
        # #                 StainAugmentor(stain_matrix=None),
        #                 StainAugmentor(method="vahadane", stain_matrix=None),
        #                 StainAugmentor(method="macenko", stain_matrix=None),
        #             ],
        #             p=0.9,
        #         ),
        albu.CoarseDropout(p=1, always_apply=True, max_height=20, max_width=20),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.Flip(p=0.5),
                albu.RandomRotate90(p=0.5),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        # albu.RandomCrop(height=256, width=256, always_apply=True),
        # albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):

    _transform = [
        # albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
