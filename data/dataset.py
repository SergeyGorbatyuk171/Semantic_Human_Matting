"""
    dataset create
Author: Zhengwei Li
Date  : 2018/12/24
"""
import cv2
import os
import random as r
import numpy as np

import torch
import torch.utils.data as data

from albumentations import Compose, OneOf, HorizontalFlip, Rotate, OpticalDistortion, HueSaturationValue, \
    RGBShift, RandomBrightness, RandomContrast, JpegCompression, Resize, RandomRain, RandomSunFlare, GaussNoise, \
    IAAAdditiveGaussianNoise, Normalize
from torch.utils.data import DataLoader

SIZE = (320, 320)


def read_files(data_dir, file_name={}):

    image_name = os.path.join(data_dir, 'image', file_name['image'])
    trimap_name = os.path.join(data_dir, 'trimap', file_name['trimap'])
    alpha_name = os.path.join(data_dir, 'alpha', file_name['alpha'])

    image = cv2.imread(image_name)
    trimap = cv2.imread(trimap_name)
    alpha = cv2.imread(alpha_name)

    return image, trimap, alpha


def random_scale_and_creat_patch(image, trimap, alpha, patch_size):
    # random scale
    if r.random() < 0.5:
        h, w, c = image.shape
        scale = 0.75 + 0.5*r.random()
        image = cv2.resize(image, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)
        trimap = cv2.resize(trimap, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)    
    # creat patch
    if r.random() < 0.5:
        h, w, c = image.shape
        if h > patch_size and w > patch_size:
            x = r.randrange(0, w - patch_size)
            y = r.randrange(0, h - patch_size)
            image = image[y:y + patch_size, x:x+patch_size, :]
            trimap = trimap[y:y + patch_size, x:x+patch_size, :]
            alpha = alpha[y:y+patch_size, x:x+patch_size, :]
        else:
            image = cv2.resize(image, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
            trimap = cv2.resize(trimap, (patch_size,patch_size), interpolation=cv2.INTER_NEAREST)
            alpha = cv2.resize(alpha, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
    else:
        image = cv2.resize(image, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
        trimap = cv2.resize(trimap, (patch_size,patch_size), interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)

    return image, trimap, alpha


def random_flip(image, trimap, alpha):

    if r.random() < 0.5:
        image = cv2.flip(image, 0)
        trimap = cv2.flip(trimap, 0)
        alpha = cv2.flip(alpha, 0)

    if r.random() < 0.5:
        image = cv2.flip(image, 1)
        trimap = cv2.flip(trimap, 1)
        alpha = cv2.flip(alpha, 1)
    return image, trimap, alpha
       
def np2Tensor(array):
    ts = (2, 0, 1)
    tensor = torch.FloatTensor(array.transpose(ts).astype(float))    
    return tensor

class human_matting_data(data.Dataset):
    """
    human_matting
    """

    def __init__(self, root_dir, imglist, patch_size):
        super().__init__()
        self.data_root = root_dir

        self.patch_size = patch_size
        with open(imglist) as f:
            self.imgID = f.readlines()
        self.num = len(self.imgID)
        print("Dataset : file number %d"% self.num)




    def __getitem__(self, index):
        # read files
        image, trimap, alpha = read_files(self.data_root, 
                                          file_name={'image': self.imgID[index].strip(),
                                                     'trimap': self.imgID[index].strip()[:-4] +'.png',
                                                     'alpha': self.imgID[index].strip()[:-4] +'.png'})
        # NOTE ! ! !
        # trimap should be 3 classes : fg, bg. unsure
        trimap[trimap==0] = 0
        trimap[trimap==128] = 1
        trimap[trimap==255] = 2 

        # augmentation
        image, trimap, alpha = random_scale_and_creat_patch(image, trimap, alpha, self.patch_size)
        image, trimap, alpha = random_flip(image, trimap, alpha)


        # normalize
        image = (image.astype(np.float32)  - (114., 121., 134.,)) / 255.0
        alpha = alpha.astype(np.float32) / 255.0
        # to tensor
        image = np2Tensor(image)
        trimap = np2Tensor(trimap)
        alpha = np2Tensor(alpha)

        trimap = trimap[0,:,:].unsqueeze_(0)
        alpha = alpha[0,:,:].unsqueeze_(0)

        sample = {'image': image, 'trimap': trimap, 'alpha': alpha}

        return sample

    def __len__(self):
        return self.num


def transforms_train(aug_proba=1.):
    return Compose(
                transforms=[
                    HorizontalFlip(p=0.5),
                    Rotate(limit=25, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0, interpolation=cv2.INTER_CUBIC),
                    OneOf([
                        IAAAdditiveGaussianNoise(p=1),
                        GaussNoise(p=1),
                    ], p=0.2),
                    OneOf([
                        HueSaturationValue(hue_shift_limit=10,
                                           sat_shift_limit=15,
                                           val_shift_limit=10, p=1),
                        RGBShift(r_shift_limit=10,
                                 g_shift_limit=10,
                                 b_shift_limit=10, p=1)
                    ]),
                    OneOf([
                        RandomContrast(p=1),
                        RandomBrightness(p=1)
                    ], p=0.3),
                    OpticalDistortion(p=0.1),
                    Resize(*SIZE),
                    Normalize()
                ], p=aug_proba, additional_targets={'trimap': 'mask'}
            )


def transforms_test(aug_proba=1.):
    return Compose(
                transforms=[
                    Resize(*SIZE),
                    Normalize()
                ], p=aug_proba, additional_targets={'trimap': 'mask'}
            )


class CocoDensepose(data.Dataset):
    def __init__(self, data_dir, transform=None):
        images_dir = os.path.join(data_dir, 'images')
        masks_dir = os.path.join(data_dir, 'masks')
        trimaps_dir = os.path.join(data_dir, 'trimaps')
        self.images = os.listdir(images_dir)
        self.masks = os.listdir(masks_dir)
        self.trimaps = os.listdir(trimaps_dir)

        self.transform = transform

    def __getitem__(self, item):
        image = cv2.imread(self.images[item])
        mask = cv2.imread(self.masks[item], 0)
        trimap = cv2.imread(self.trimaps[item], 0)

        trimap[trimap == 0] = 0
        trimap[trimap == 128] = 1
        trimap[trimap == 255] = 2

        data = {'image': image, 'mask': mask, 'trimap': trimap}
        transformed = self.transform(**data)
        image = torch.FloatTensor(transformed['image'].transpose((2, 0, 1).satype(float)))
        mask = torch.FloatTensor(transformed['mask'].astype(float))
        trimap = torch.FloatTensor(transformed['trimap'].astype(float))

        return {'image': image, 'trimap': trimap, 'alpha': mask}

    def __len__(self):
        return len(self.images)


def make_loader(data_dir, shuffle=False, transform=None, batch_size=1, workers=4):
    return DataLoader(
        dataset=CocoDensepose(data_dir, transform=transform),
        shuffle=shuffle,
        num_workers=workers,
        drop_last=True,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
