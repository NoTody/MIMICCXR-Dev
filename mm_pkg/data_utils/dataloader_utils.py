# util libraries
import h5py
import math

# preprocessing libraries
import numpy as np
import pandas as pd
from skimage.transform import resize

# torch libraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# user defined files
from ..data_utils.augmentation_utils import *
from ..data_utils.augmentation_utils import TrainTransform


class MIMIC_CXR_Unsupervised(Dataset):
    def __init__(self, args, dict_image_mapping, data_df_path, full_report=True, two_transform=True, train=True):
        self.args = args
        self.full_report = full_report
        self.two_transform = two_transform
        self.data_df = pd.read_csv(data_df_path, sep='\t')
        self.train = train
        self.dict_image_mapping = dict_image_mapping
        #self.mimic_cxr_path = Path('/gpfs/data/denizlab/Datasets/Public/physionet.org/files/mimic-cxr/2.0.0')

    def __getitem__(self, index):
        args = self.args
        # images
        image_path = self.data_df.iloc[index]['dicom_path']
        image = self.dict_image_mapping[image_path]
        # to PIL images
        PIL_image = Image.fromarray(image, 'L')
        # texts
        impression = self.data_df.iloc[index]['impression']
        findings = self.data_df.iloc[index]['findings']

        # if not using full report, only impression is used
        if self.full_report:
            if isinstance(findings, str):
                text = impression + findings
            else:
                text = impression

        transform = TrainTransform(self.two_transform)
        images = transform(PIL_image)
        return images[0], text


    def __len__(self):
        return len(self.data_df)


