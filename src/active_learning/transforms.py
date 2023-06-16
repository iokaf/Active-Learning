"""This module is responsible for defining the data transformations and
augmentations used in the active learning pipeline."""

from typing import Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(config: Dict) -> A.Compose:
    """ Returns the data transformations and augmentations used in the active
    
    Args:
        config (Dict): The configuration dictionary.

    Returns:
        A.Compose: The data transforms.
    """
    img_width = config["data"]["image-width"]
    img_height = config["data"]["image-height"]
    img_mean = config["data"]["mean"]
    img_std = config["data"]["std"]
    transform = A.Compose([
        A.Resize(img_height, img_width),
        A.Normalize(
            mean=img_mean,
            std=img_std,
            max_pixel_value=255,
            always_apply=True
        ),
        ToTensorV2()
    ])

    return transform

def get_augmentations(config: Dict) -> A.Compose:
    """ Returns the data transformations and augmentations used in the active
    
    Args:
        config (Dict): The configuration dictionary.

    Returns:
        A.Compose: The train data augmentations.
    """
    img_width = config["data"]["image-width"]
    img_height = config["data"]["image-height"]

    augmentations = A.Compose([
        A.Resize(img_height, img_width),
        A.RandomBrightnessContrast(brightness_limit=0.1, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomGamma(p=0.5),
        A.HueSaturationValue(hue_shift_limit=3, p=0.5),
        
        A.CoarseDropout(
            max_holes=8, 
            max_height=int(img_height/8), 
            max_width=int(img_width/8), 
            min_holes=4, 
            min_height=int(img_height/12),
            min_width=int(img_height/12), 
            p=0.5
        ),
        
        A.OneOf([
                A.CLAHE(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ], p=0.5),

        A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.ElasticTransform(p=0.3),
            ], p=0.5),


        A.OneOf([
                A.ISONoise(p=0.5, intensity=(0.1, 0.2), color_shift=(0.01, 0.05)),
                A.GaussNoise(p=0.5),
                A.MultiplicativeNoise(per_channel=True, elementwise=True, p=0.5)
            ], p=0.5),
        

        A.ColorJitter(hue=0.05, p=0.8),
        
        A.OneOf([
                A.MotionBlur(p=.2, blur_limit=3),
                A.MedianBlur(blur_limit=3, p=.1),
                A.Blur(blur_limit=2, p=.1),
                A.GaussNoise(p=0.5),
            ], p=0.5),
        
    ])

    return augmentations