# util libraries
import h5py
import math
import argparse
import random

# preprocessing libraries
import numpy as np
import pandas as pd
from skimage.transform import resize
import os
import csv

# torch libraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from pathlib import Path
import pickle

from tqdm import tqdm

from torchmetrics.image import fid
from torchmetrics.image.fid import FrechetInceptionDistance

class MIMIC_CXR_Unsupervised(Dataset):
    def __init__(self, dict_image_mapping, data_df_path, transform, train=True):
        self.data_df = pd.read_csv(data_df_path, sep='\t')
        self.transform = transform
        self.train = train
        self.dict_image_mapping = dict_image_mapping
        #self.mimic_cxr_path = Path('/gpfs/data/denizlab/Datasets/Public/physionet.org/files/mimic-cxr/2.0.0')

    def __getitem__(self, index):
        # load images
        image_path = self.data_df.iloc[index]['dicom_path']
        image = self.dict_image_mapping[image_path]
        # to PIL images
        PIL_image = Image.fromarray(image).convert("RGB")
        
        # transform images
        transform = self.transform
        
        images = transform(PIL_image)
        
        images = ((images * 255) / torch.max(images)).to(torch.uint8)
        return images
    
    def __len__(self):
        return len(self.data_df)

    
class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []

        with open(data_PATH, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in csvReader:
                image_name = line[0]
                label = line[5:]
                
                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                
                image_names.append('/gpfs/data/denizlab/Datasets/Public/' + image_name)
        
#                 image_names.append('./' + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
            
        image = ((image * 255) / torch.max(image)).to(torch.uint8)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
    
    
class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
            
        image = ((image * 255) / torch.max(image)).to(torch.uint8)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


num_samples = 100000
batch_size = 64
max_iter = num_samples // batch_size


def calc_fid(dataloader1, dataloader2, max_iter, use_mimic=True):
    count = 0
    fid = FrechetInceptionDistance(feature=768).cuda()
    for batch1, batch2 in tqdm(zip(dataloader1, dataloader2), total=max_iter):
        if count > max_iter:
            break
         
        if use_mimic:
            imgs1 = batch1
        else:
            imgs1 = batch1[0]
        
        imgs2 = batch2[0]
        fid.update(imgs1.cuda(), real=True)
        fid.update(imgs2.cuda(), real=False)

        count += 1
        
    fid_score = fid.compute()
    return fid_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str , choices=["mimic_chexpert", "mimic_nih", "chexpert_nih"] ,default="mimic_chexpert")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    return args


def main(args):
    num_samples = 50000
    batch_size = args.batch_size
    max_iter = num_samples // batch_size

    mimic_cxr_path = Path('/gpfs/data/denizlab/Datasets/Public/physionet.org/files/mimic-cxr/2.0.0')
    with open(mimic_cxr_path / 'mimic_cxr_imgs_v3.pkl', 'rb') as handle:
        dict_image_mapping = dict(pickle.load(handle))

    if "mimic" in args.datasets:
        train_df_path = '/gpfs/data/denizlab/Users/hh2740/mimic-cxr_full_train.csv'
        mimic_data = MIMIC_CXR_Unsupervised(dict_image_mapping=dict_image_mapping, data_df_path= train_df_path, transform=transform)
        mimic_loader = DataLoader(mimic_data, batch_size=batch_size, num_workers=10, pin_memory=False, \
                            shuffle=True, drop_last=True)

    if "nih" in args.datasets:
        data_dir = '/gpfs/data/denizlab/Datasets/Public/NIH_Chest_X-ray/images'
        image_list_file = './chestxray14_evaluation/train_list.txt'
        nih_data = ChestXrayDataSet(data_dir, image_list_file, transform=transform)
        nih_loader = DataLoader(nih_data, batch_size=batch_size, num_workers=10, pin_memory=False, \
                            shuffle=True, drop_last=True)

    if "chexpert" in args.datasets:
        pathFileTrain = '/gpfs/data/denizlab/Users/skr2369/Chexpert/CheXpert-v1/U1-V1/train_mod.csv'
        chexpert_data = CheXpertDataSet(pathFileTrain, transform, policy = "ones")
        chexpert_loader = DataLoader(chexpert_data, batch_size=batch_size, num_workers=10, pin_memory=False, \
                                shuffle=True, drop_last=True)

    if args.datasets == "mimic_chexpert":
        fid = calc_fid(mimic_loader, chexpert_loader, max_iter, use_mimic=True)
    elif args.datasets == "mimic_nih":
        fid = calc_fid(mimic_loader, nih_loader, max_iter, use_mimic=True)
    elif args.datasets == "chexpert_nih":
        fid = calc_fid(chexpert_loader, nih_loader, max_iter, use_mimic=False)
    else:
        raise NotImplementedError("Datasets combination doesn't exist!")

    print(fid)

if __name__ == '__main__':
    args = get_args()
    main(args)


