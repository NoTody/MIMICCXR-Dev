import numpy as np
import pandas as pd
import datetime
import os
from collections import OrderedDict
from pathlib import Path
import uuid
import pydicom

import cv2
import matplotlib.pyplot as plt

import gzip

import multiprocessing as mp
from multiprocessing import Manager, Pool
from tqdm import tqdm

# mimic-cxr should have two subfolders:
#   files/ - with MIMIC-CXR DICOMs
#   jpg/files/ - with MIMIC-CXR-JPG JPG files
mimic_cxr_path = Path('/gpfs/data/denizlab/Datasets/Public/physionet.org/files/mimic-cxr/2.0.0')

def process_df(row):
    subject_id, study_id, dicom_id, dicom_path = row[0], row[1], row[2], row[3]
    report_path = path[:19] + study_id + '.txt'

    #report_path = mimic_cxr_path + "/" + report_path
    with open(mimic_cxr_path / report_path) as f:
        lines = f.read().replace('\n', '')
    if "IMPRESSION:" in lines:
        impression = lines.split("IMPRESSION:")[1]

    new_row = [subject_id, study_id, dicom_id, dicom_path, report_path, impression]
    shared_res.append(new_row)


if __name__ == "__main__":
	lst_df = df.values.to_list()

	manager = Manager()
	shared_res = manager.list()

	print(f"Num CPU = {mp.cpu_count()}, Num CUIs = {len(lst_df)}")
	pool = Pool(processes=mp.cpu_count())

	for _ in tqdm(pool.imap_unordered(process_df, lst_df), total=len(lst_df)):
	    pass

	pool.close()

	lst = list(shared_res)
	new_df = pd.DataFrame(lst, columns = ['subject_id', 'study_id', 'dicom_id', 'dicom_path', 'report_path'])
	new_df.to_csv("./mimic_cxr_mapping.csv")

	print("Save Done")

