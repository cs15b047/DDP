import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch.utils.data as data
import cv2
import h5py
from numba import jit
import multiprocessing
from joblib import Parallel, delayed


crop1, crop2, channels = 224, 224, 3

def get_imgs(base_dir, pth_list, image_file, dataset_name):
	global images
	full_pth_list = [os.path.join(base_dir, pth) for pth in pth_list]
	data_len = len(full_pth_list)
	sz = (data_len, crop1, crop2, channels)
	# images = np.memmap(dataset_name, dtype = np.float64, mode='w+', shape = sz)
	images = image_file.create_dataset(dataset_name, sz, np.float64, compression = "gzip")
	# print(images.shape)
	full_pth_list_with_idx = [list(x) for x in zip(full_pth_list, list(range(data_len)))]
	loop(full_pth_list_with_idx)

	return images

def loop(full_pth_list_with_idx):
	global temp_img_array
	batch = 7500
	temp_img_array = np.zeros((batch, crop1, crop2, channels))
	length = len(full_pth_list_with_idx)
	curr_idx = 0
	for idx, data_pt in enumerate(tqdm(full_pth_list_with_idx)):
		[pth, idx] = data_pt
		img = cv2.imread(pth, cv2.IMREAD_COLOR)
		mxm1, mxm2 = img.shape[0] - crop1, img.shape[1] - crop2
		
		#TODO: How to deal with variable sized images??

		#REMOVE if after resolving TODO
		if(min(img.shape[0], img.shape[1]) <= 224):
			# print(img.shape)
			temp_img_array[idx % batch] = np.random.normal(size = (crop1, crop2, channels)) #Fill random numbers for now
		else:
			x, y = np.random.randint(0, mxm1), np.random.randint(0, mxm2)
			temp_img_array[idx % batch] = img[x:x+crop1, y:y+crop2]
		
		mod_idx = (idx % batch)
		if((mod_idx == batch - 1) or (idx == length - 1)):
			images[curr_idx: curr_idx + mod_idx + 1] = temp_img_array[:mod_idx + 1]
			curr_idx += mod_idx + 1
	