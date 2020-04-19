import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch.utils.data as data
import cv2
# import h5py
# from numba import jit
from clothing_utils import get_imgs
from PIL import Image

class Clothing(data.Dataset):
	clean_data_file_name = "clean_label_kv.txt"
	noisy_data_file_name = "noisy_label_kv.txt"
	noisy_data_keys = "noisy_train_key_list.txt"
	val_data_file_name = "clean_val_key_list.txt"
	test_data_file_name = "clean_test_key_list.txt"
	clean_train_data_file_name = "clean_train_key_list.txt"

	def __init__(self, data_dir = "clothing1M", train = True, val = False, test = False, transform = None):
		self.data_dir = data_dir
		self.transform = transform
		self.train, self.val, self.test = train, val, test

		self.clean_train_keys, self.clean_train_labels = None, None
		self.noisy_train_keys, self.noisy_train_labels = None, None

		self.image_keys, self.labels, self.label_dict = self.preprocess()

		if self.train:
			self.noisy_train_keys, self.noisy_train_labels = self.image_keys, self.labels

		#Shuffle the dataset
		# permutation = np.random.permutation(np.arange(len(self.labels)))
		# self.image_keys = np.array(self.image_keys)[permutation]
		# self.labels = np.array(self.labels)[permutation]
		
		# Balance the dataset to contain equal number of examples of each class (as given by the noisy labels)
		# if self.train:
		# 	self.reduced_idxs =  self.balance_dataset()
		# 	self.reduced_idxs = self.reduced_idxs.astype(int)
		# 	print(self.reduced_idxs)
		# 	self.labels = self.labels[self.reduced_idxs]
		# 	self.image_keys = self.image_keys[self.reduced_idxs]

	def balance_dataset(self):
		counts = np.bincount(self.labels)
		non_zero_cnt_ele = np.nonzero(counts)[0]
		label_cnts = counts[non_zero_cnt_ele]
		new_idxs = np.array([])
		min_cnt = int(np.amin(label_cnts))
		print(min_cnt)
		for label in non_zero_cnt_ele:
			class_idxs = (np.argwhere(self.labels == label)).reshape(-1)
			print(len(class_idxs) == label_cnts[label])
			reduced_class_idxs = np.random.choice(class_idxs, size = (min_cnt, ), replace = False)
			new_idxs = np.concatenate((new_idxs, reduced_class_idxs))
		
		return new_idxs


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		key = self.image_keys[index]
		img_pth = os.path.join(self.data_dir, key)
		img = cv2.imread(img_pth, cv2.IMREAD_COLOR)

		#Resize image to specific size
		dim = (256, 256) ##Randomly decided to crop 224 x 224 image for augmentn
		img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA) # ??

		########### PIL is RGB, cv2 is BGR#######
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		#########################################
		img = Image.fromarray(img)

		if(self.transform is not None):
			img = self.transform(img)
		
		label = self.labels[index]
		return img, label, index

	def get_label(self, index):
		return self.labels[index]


	def preprocess(self):
		clean_data_raw = pd.read_csv(os.path.join(self.data_dir, self.clean_data_file_name), delim_whitespace=True, header=None)
		clean_data_images_path, clean_data_labels = list(clean_data_raw[0]), list(clean_data_raw[1])
		clean_data_dict = dict(zip(clean_data_images_path, clean_data_labels))

		if self.train:
			noisy_data_raw = pd.read_csv(os.path.join(self.data_dir, self.noisy_data_file_name), delim_whitespace=True, header=None)
			noisy_data_images_path, noisy_data_labels = list(noisy_data_raw[0]), list(noisy_data_raw[1])
			noisy_data_dict = dict(zip(noisy_data_images_path, noisy_data_labels))

			noisy_data_relevant = pd.read_csv(os.path.join(self.data_dir, self.noisy_data_keys), delim_whitespace=True, header=None)
			noisy_data_relevant = list(noisy_data_relevant[0])
			noisy_labels_relevant = [noisy_data_dict[key] for key in noisy_data_relevant]
			noisy_dict_relevant = dict(zip(noisy_data_relevant, noisy_labels_relevant))

			clean_train_data_key = pd.read_csv(os.path.join(self.data_dir, self.clean_train_data_file_name), delim_whitespace=True, header=None)
			clean_train_data_key = list(clean_train_data_key[0])
			clean_train_labels = [clean_data_dict[key] for key in clean_train_data_key]
			clean_train_dict = dict(zip(clean_train_data_key, clean_train_labels))
			self.clean_train_keys = clean_train_data_key
			self.clean_train_labels = clean_train_labels


			return noisy_data_relevant, noisy_labels_relevant, noisy_dict_relevant
		elif self.val:
			val_data_key = pd.read_csv(os.path.join(self.data_dir, self.val_data_file_name), delim_whitespace=True, header=None)
			val_data_key = list(val_data_key[0])
			val_labels = [clean_data_dict[key] for key in val_data_key]
			val_dict = dict(zip(val_data_key, val_labels))
			return val_data_key, val_labels, val_dict

		elif self.test:
			test_data_key = pd.read_csv(os.path.join(self.data_dir, self.test_data_file_name), delim_whitespace=True, header=None)
			test_data_key = list(test_data_key[0])
			test_labels = [clean_data_dict[key] for key in test_data_key]
			test_dict = dict(zip(test_data_key, test_labels))
			return test_data_key, test_labels, test_dict