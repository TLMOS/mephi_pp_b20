import cv2
import os
import random
import numpy as np
from torch.utils.data import Dataset
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, \
    DualTransform, ImageOnlyTransform
from albumentations.pytorch.functional import img_to_tensor
from albumentations.augmentations.functional import image_compression, rot90
import matplotlib.pyplot as plt

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized

class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")
    
def create_transforms(size):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    )

def blackout_random(image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    binary_mask = mask > 0.4 * 255
    h, w = binary_mask.shape[:2]

    tries = 50
    current_try = 1
    while current_try < tries:
        first = random.random() < 0.5
        if random.random() < 0.5:
            pivot = random.randint(h // 2 - h // 5, h // 2 + h // 5)
            bitmap_msk = np.ones_like(binary_mask)
            if first:
                bitmap_msk[:pivot, :] = 0
            else:
                bitmap_msk[pivot:, :] = 0
        else:
            pivot = random.randint(w // 2 - w // 5, w // 2 + w // 5)
            bitmap_msk = np.ones_like(binary_mask)
            if first:
                bitmap_msk[:, :pivot] = 0
            else:
                bitmap_msk[:, pivot:] = 0

        if np.count_nonzero(image * np.expand_dims(bitmap_msk, axis=-1)) / 3 > (h * w) / 5 \
                or np.count_nonzero(binary_mask * bitmap_msk) > 40:
            mask *= bitmap_msk
            image *= np.expand_dims(bitmap_msk, axis=-1)
            break
        current_try += 1
    return image

class DeepFakeClassifierDataset(Dataset):
    def __init__(self, metadata, processed_data_path, image_size, mode,
                normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                blackout_probability=0.42,
                rotate=True, transform=True):
        self.metadata = metadata
        self.processed_data_path = processed_data_path
        self.mode = mode
        self.normalize = normalize
        self.blackout_probability = blackout_probability
        self.rotate = rotate
        self.transform = transform
        if transform:
            self.transforms = create_transforms(image_size)
    
    def __getitem__(self, index):
        vid, cid, label = self.metadata.loc[index]
        img_path = os.path.join(self.processed_data_path, 'crops', vid, '{}.png'.format(cid))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            if self.transform:
                img = self.transforms(image=img)['image']
            if random.random() < self.blackout_probability:
                img = blackout_random(img)
            if self.rotate:
                rotation = random.randint(0, 3)
                img = rot90(img, rotation)
        img = img_to_tensor(img, self.normalize)
        return img, label, vid, cid
        
    def __len__(self):
        return len(self.metadata)