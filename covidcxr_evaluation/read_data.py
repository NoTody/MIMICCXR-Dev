# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class CovidDataset(Dataset):
    def __init__(self, data_dir, data_df_path, transform):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(data_df_path, sep=' ')
        self.transform = transform

    def __getitem__(self, index):
        # load images
        index = (int)(index)
        image_path = self.data_df.iloc[index]['filename']
        label = self.data_df.iloc[index]['labels']
        image_name = os.path.join(self.data_dir, image_path)

        try:
            PIL_image = Image.open(image_name).convert('RGB')
        except ValueError:
            pass

        # transform images
        transform = self.transform
        images = transform(PIL_image)
        
        return images, torch.tensor(label).float()
    
    def __len__(self):
        return len(self.data_df)

