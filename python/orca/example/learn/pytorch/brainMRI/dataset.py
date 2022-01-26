import os
from os.path import exists
from os import makedirs
import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import cv2
from PIL import Image
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torchvision.transforms as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader

from mpl_toolkits.axes_grid1 import ImageGrid
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def set_seed(seed = 0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def diagnosis(mask_path):
    return 1 if np.max(cv2.imread(mask_path)) > 0 else 0

def dataset():
    ROOT_PATH = '/home/mingxuan/BigDL/python/orca/example/learn/pytorch/brainMRI/data/kaggle_3m/'
    mask_files = glob.glob(ROOT_PATH + '*/*_mask*')
    image_files = [file.replace('_mask', '') for file in mask_files]
    files_df = pd.DataFrame({"image_path": image_files,
                    "mask_path": mask_files,
                    "diagnosis": [diagnosis(x) for x in mask_files]})

    train_df, test_df = train_test_split(files_df, stratify=files_df['diagnosis'], test_size=0.15, random_state=0)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df

class BrainDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0])
        image = np.array(image)/255.
        mask = cv2.imread(self.df.iloc[idx, 1], 0)
        mask = np.array(mask)/255.
        
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image).type(torch.float32)
        image = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        mask = np.expand_dims(mask, axis=-1).transpose((2,0,1))
        mask = torch.from_numpy(mask).type(torch.float32)
    
        return image, mask



