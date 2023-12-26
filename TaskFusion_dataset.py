# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os
import copy
import random
from imgaug import augmenters as iaa
import torchvision.transforms as transforms

import Util

sometimes = lambda aug: iaa.Sometimes(0.8, aug)
np.random.seed(2)


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'enLightIntrain',
                         'test'], 'split must be "train"|"val"|"test"|"enLightIntrain"'

        if split == 'train':
            data_dir_vis = './MSRS/Visible/train/MSRS/'
            data_dir_ir = './MSRS/Infrared/train/MSRS/'
            data_dir_bi = './MSRS/Bi/'

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_bi, self.filenames_bi = prepare_data_path(data_dir_bi)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir), len(self.filenames_bi))
            self._tensor = transforms.ToTensor()
        if split == 'enLightIntrain':
            data_dir_vis = './lianHeXuLian/EnlightOut/TrainOut'
            data_dir_ir = './MSRS/Infrared/train/MSRS/'
            data_dir_bi = './MSRS/Bi/'

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_bi, self.filenames_bi = prepare_data_path(data_dir_bi)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir), len(self.filenames_bi))
            self._tensor = transforms.ToTensor()

        elif split == 'test':
            data_dir_ir = './MSRS/Infrared/test/MSRS/'
            data_dir_vis = './lianHeXuLian/EnlightOut/TestOut/'

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'enLightIntrain':

            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            bi_path = self.filepath_bi[index]

            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            bi = np.array(Image.open(bi_path))

            image_vis = (
                    np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            bi = np.array(Image.fromarray(bi), dtype=np.int64) / 255.0

            name = self.filenames_vis[index]

            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name, torch.tensor(bi)
            )
        elif self.split == 'test':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            image_vis = (
                    np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )

    def __len__(self):
        return self.length


def bright_transform(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols, num_channels = x.shape
    num_block = 10

    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)

        for channel in range(num_channels):
            window = orig_image[noise_x:noise_x + block_noise_size_x,
                     noise_y:noise_y + block_noise_size_y, channel]

            window = brightness_aug(window, 0.5 + 0.5 * np.random.random_sample())

            image_temp[noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y, channel] = window

    bright_transform_x = image_temp
    return bright_transform_x


def brightness_aug(x, gamma):
    aug_brightness = iaa.Sequential(sometimes(iaa.GammaContrast(gamma=gamma)))
    aug_image = aug_brightness(images=x)
    return aug_image


def fourier_transform(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 10
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        window = orig_image[noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y]
        window = fourier_broken(window, block_noise_size_x, block_noise_size_y)
        image_temp[noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = window
    bright_transform_x = image_temp

    return bright_transform_x


def fourier_transform_color(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols, num_channels = x.shape
    num_block = 10

    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        for channel in range(num_channels):
            window = orig_image[noise_x:noise_x + block_noise_size_x,
                     noise_y:noise_y + block_noise_size_y, channel]
            window = fourier_broken(window, block_noise_size_x, block_noise_size_y)
            image_temp[noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y, channel] = window
    bright_transform_x = image_temp

    return bright_transform_x


def fourier_broken(x, nb_rows, nb_cols):
    aug_a = iaa.GaussianBlur(sigma=0.5)
    aug_p = iaa.Jigsaw(nb_rows=nb_rows, nb_cols=nb_cols, max_steps=(1, 5))
    fre = np.fft.fft2(x)
    fre_a = np.abs(fre)
    fre_p = np.angle(fre)
    fre_a_normalize = fre_a / (np.max(fre_a) + 0.0001)
    fre_p_normalize = fre_p
    fre_a_trans = aug_a(image=fre_a_normalize)
    fre_p_trans = aug_p(image=fre_p_normalize)
    fre_a_trans = fre_a_trans * (np.max(fre_a) + 0.0001)
    fre_p_trans = fre_p_trans
    fre_recon = fre_a_trans * np.e ** (1j * (fre_p_trans))
    img_recon = np.abs(np.fft.ifft2(fre_recon))
    return img_recon
