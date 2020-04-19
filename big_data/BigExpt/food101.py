import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

class Food101N(data.Dataset):
	class_names = "classes.txt"
	noisy_image_file = "imagelist.tsv"
	verified_train = "verified_train.tsv"
	verified_val = "verified_val.tsv"
	test_file = "test.txt"
	
	def __init__(self, data_dir = "../Food-101N_release", train = True, val = False, test = False, transform = None):
		self.data_dir = data_dir
		self.train, self.val, self.test = train, val, test
		self.transform = transform
		self.image_paths, self.labels, self.dict = self.preprocess()
		if self.val:
			self.img_pth_list, self.all_labels, self.label_clean_or_not = self.preprocess_annotated_data()
		assert((len(self.image_paths) == len(self.labels)) and (len(self.image_paths) == len(list(self.dict.keys()))))

	def __len__(self):
		return len(self.labels)

	def get_label(self, index):
		return self.labels[index]

	def __getitem__(self, index):
		pth = self.image_paths[index]
		img = cv2.imread(pth, cv2.IMREAD_COLOR)
		dim = (224, 224) ##Crop 224 x 224 from image for training/inference
		img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
		########### PIL is RGB, cv2 is BGR#######
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		#########################################
		img = Image.fromarray(img)
		if(self.transform is not None):
			img = self.transform(img)

		label = self.labels[index]

		return img, label, index

	def preprocess_annotated_data(self):
		if self.val:
			valid_indices = [ind for ind in self.annotated_list.index if os.path.exists(os.path.join(self.data_dir, "images", self.annotated_list["class_name/key"][ind]))]
			# print(len(self.annotated_list.index) - len(valid_indices))
			self.annotated_list = self.annotated_list.iloc[valid_indices]
			self.img_pth_list = list(self.annotated_list["class_name/key"])
			self.img_pth_list = [os.path.join(self.data_dir, "images", pth) for pth in self.img_pth_list]
			self.all_labels = [self.class_dict[string.split('/')[3]] for string in self.img_pth_list]
			self.label_clean_or_not = list(self.annotated_list["verification_label"])
			
			return self.img_pth_list, self.all_labels, self.label_clean_or_not
		else:
			return None, None, None

	def preprocess(self):
		class_name_file_pth = os.path.join(self.data_dir, "meta", self.class_names)
		classes = list((pd.read_csv(class_name_file_pth))['class_name'])
		self.class_dict = dict((classes[i], i) for i in range(len(classes)))
		self.reverse_dict = dict((self.class_dict[k], k) for k in self.class_dict.keys())

		if self.train:
			image_list_file_pth = os.path.join(self.data_dir, "meta", self.noisy_image_file)
			self.noisy_image_list = list(pd.read_csv(image_list_file_pth)["class_name/key"])
			self.noisy_image_path = [os.path.join(self.data_dir, "images", pth) for pth in self.noisy_image_list if os.path.exists(os.path.join(self.data_dir, "images", pth))]
			self.train_labels = [self.class_dict[string.split('/')[3]] for string in self.noisy_image_path]
			assert(len(self.noisy_image_path) == len(self.train_labels))
			self.train_dict = dict(zip(self.noisy_image_path, self.train_labels))

			###############Remove samples common in val and train#################
			val_file_pth = os.path.join(self.data_dir, "meta", self.verified_train)
			val_list = pd.read_csv(val_file_pth, delim_whitespace = True) #sep = "\t"
			all_val_imgs = list(val_list["class_name/key"])
			correct_val_rows = val_list.loc[val_list["verification_label"] == 1]
			correct_val_imgs = list(correct_val_rows["class_name/key"])
			val_images_pth = [os.path.join(self.data_dir, "images", pth) for pth in all_val_imgs if os.path.exists(os.path.join(self.data_dir, "images", pth))]

			for pth in val_images_pth:
				self.train_dict.pop(pth, None)
			self.noisy_image_path = list(self.train_dict.keys())
			self.train_labels = [self.train_dict[img_pth] for img_pth in self.noisy_image_path]
			#####################################################################################

			return self.noisy_image_path, self.train_labels, self.train_dict
		elif self.val:
			val_file_pth = os.path.join(self.data_dir, "meta", self.verified_train)
			val_list = pd.read_csv(val_file_pth, delim_whitespace = True) #sep = "\t"
			all_val_imgs = list(val_list["class_name/key"])
			self.annotated_list = val_list
			correct_val_rows = val_list.loc[val_list["verification_label"] == 1]
			correct_val_imgs = list(correct_val_rows["class_name/key"])
			
			############ Filter Non existent paths##########################
			self.correct_val_images_pth = [os.path.join(self.data_dir, "images", pth) for pth in correct_val_imgs if os.path.exists(os.path.join(self.data_dir, "images", pth))]
			self.val_images_pth = [os.path.join(self.data_dir, "images", pth) for pth in all_val_imgs if os.path.exists(os.path.join(self.data_dir, "images", pth))]
			################################################################

			#######################Calc labels and make mapping#######################################
			self.correct_val_labels = [self.class_dict[string.split('/')[3]] for string in self.correct_val_images_pth]
			
			assert(len(self.correct_val_labels) == len(self.correct_val_images_pth))
			for i in range(len(self.correct_val_labels)):
				substr = self.reverse_dict[self.correct_val_labels[i]]
				assert(self.correct_val_images_pth[i].find(substr) != -1)

			self.val_dict = dict(zip(self.correct_val_images_pth, self.correct_val_labels))
			return self.correct_val_images_pth, self.correct_val_labels, self.val_dict
			##########################################################################################

		elif self.test:
			image_list_file_pth = os.path.join(self.data_dir, "meta", self.test_file)
			with open(image_list_file_pth, 'r') as file:
				raw = file.readlines()
				#Rstrip removes trailing newline
				pths = [line.rstrip() + ".jpg" for line in raw]
				self.test_image_pth = [os.path.join(self.data_dir, "test_images", pth) for pth in pths if os.path.exists(os.path.join(self.data_dir, "test_images", pth))]
				self.test_labels = [self.class_dict[pth.split('/')[3]] for pth in self.test_image_pth]
				self.test_dict = dict(zip(self.test_image_pth, self.test_labels))
				
				return self.test_image_pth, self.test_labels, self.test_dict

f = Food101N(val = True, train=False)

