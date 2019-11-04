from io import BytesIO

import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torch

import glob
import imageio
import scipy.io
import cv2
import IPython
import os
import random
import torch
import tqdm

import paths


def To_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).float()

def To_numpy(tensor):
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)


def load_img(filename, dim):
    return cv2.resize(imageio.imread(filename), (dim, dim))

def save_img(filename_out, img, skip_if_exist=False):    
    if skip_if_exist and os.path.exists(filename_out):
        return
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    imageio.imwrite(filename_out, img)

    
def load_mat(filename, key):
    return scipy.io.loadmat(filename)[key]

def save_mat(filename_out, data, key, skip_if_exist=False):
    if skip_if_exist and os.path.exists(filename_out):
        return
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    scipy.io.savemat(filename_out, {key: data})

##########################################################################
## new
##########################################################################
class MultiResolutionDataset():
    def __init__(self, resolution=256, exclude_neutral=True):
        self.resolution = resolution
        self.exclude_neutral = exclude_neutral
        
        self.labels = sorted(glob.glob(paths.file_exp_weights))
        self.images = sorted(glob.glob(paths.file_exp_pointcloud_ms))
            
        self.labels_mean = paths.load_exp_mean()
        self.labels_std = paths.load_exp_std()
            
        self.length = len(self.labels)
        print (len(self.labels), len(self.images))
        
    def __len__(self):
        return 10_000_000
        
    
    def sample_label(self, k=1, randn=True):
        # return [k * label_size]
        mean = torch.from_numpy(self.labels_mean).unsqueeze(0).repeat(k, 1)
        std = torch.from_numpy(self.labels_std).unsqueeze(0).repeat(k, 1)

        mask = torch.zeros((k, 17*3))
        choose_id = torch.randint(0,17,(k,))
        choose_ids = torch.stack((choose_id*3, choose_id*3+1, choose_id*3+2),dim=1)
        mask = mask.scatter_(1, choose_ids, 1)

        return (mask*torch.normal(mean=mean, std=std)).float()
            
    
    def __getitem__(self, index):
        index = index % self.length
        img = To_tensor(load_img(self.images[index], self.resolution))
        label = torch.from_numpy(load_mat(self.labels[index], "theta").flatten())
        return img.float(), label.float()
    
