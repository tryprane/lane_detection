from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation(image_size: Tuple[int, int], use_weather: bool = False) -> A.Compose:
    height, width = image_size
    augmentation_block = []
    if use_weather:
        augmentation_block = [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.RandomRain(
                        blur_value=3,
                        brightness_coefficient=0.9,
                        drop_width=1,
                        drop_length=18,
                        rain_type="heavy",
                        p=1.0,
                    ),
                    A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=1.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.25,
                        contrast_limit=0.25,
                        p=1.0,
                    ),
                ],
                p=0.7,
            )
        ]

    return A.Compose(
        [
            A.Resize(height=height, width=width),
            *augmentation_block,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_validation_augmentation(image_size: Tuple[int, int]) -> A.Compose:
    height, width = image_size
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
