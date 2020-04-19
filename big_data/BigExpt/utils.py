import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse, sys
import datetime
import shutil
import pickle
from tqdm import tqdm
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt

from data.mnist import MNIST
from data.cifar import CIFAR10, CIFAR100
from clothing import Clothing
from food101 import Food101N


def get_input_info(dataset_name):
	if (dataset_name == "MNIST"):
		input_channel=1
		num_classes=10
		size = 28
	elif (dataset_name == "CIFAR10"):
		input_channel=3
		num_classes=10
		size = 32
	elif (dataset_name == "CIFAR100"):
		input_channel=3
		num_classes=100
		size = 32
	elif (dataset_name == "Clothes1M"):
		input_channel = 3
		num_classes = 14
		size = 224
	elif(dataset_name == "Food101N"):
		input_channel = 3
		num_classes = 101
		size = 224
	return input_channel, num_classes, size

def get_transform():
	choice = transforms.RandomChoice([
		transforms.RandomHorizontalFlip(p=1),
		transforms.RandomRotation(degrees=30),
		])
	return transforms.Compose([choice, transforms.ToTensor()])

def clothing_transform(img_sz):
	normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
	choice = transforms.RandomChoice([
			transforms.RandomHorizontalFlip(p = 1),
			transforms.RandomRotation(degrees=30),
		])

	return transforms.Compose([choice, transforms.RandomCrop(img_sz), transforms.ToTensor(), normalize])

def food_transform(img_sz):
	normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

	return transforms.Compose([transforms.RandomCrop(img_sz), transforms.ToTensor(), normalize])

# def cmp(dict1, dict2):
# 	keys1 = list(dict1.keys())
# 	keys2 = list(dict2.keys())
# 	assert(keys1 == keys2)

# 	for k in keys1:
# 		assert(dict1[k] == dict2[k])
# 	return True

def get_datasets(dataset_name, noise_type = None, noise_rate = None):
	val_dataset = None
	if(dataset_name == "MNIST"):
		train_dataset = MNIST(root='./data/',download=True,  train=True, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
		test_dataset = MNIST(root='./data/',download=True,  train=False, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
	elif (dataset_name == "CIFAR10"):
		train_dataset = CIFAR10(root='./data/',download=True,  train=True, transform=get_transform(),noise_type=noise_type,noise_rate=noise_rate)
		test_dataset = CIFAR10(root='./data/',download=True,  train=False, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
	elif (dataset_name == "CIFAR100"):
		train_dataset = CIFAR100(root='./data/',download=True,  train=True, transform=get_transform(),noise_type=noise_type,noise_rate=noise_rate)
		test_dataset = CIFAR100(root='./data/',download=True,  train=False, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
	elif (dataset_name == "Clothes1M"):
		img_sz = 224
		train_dataset = Clothing(data_dir='../clothing1M/', train = True, val = False, test=False, transform = clothing_transform(img_sz))
		val_dataset = Clothing(data_dir='../clothing1M/', train = False, val = True, test=False, transform = clothing_transform(img_sz))
		test_dataset = Clothing(data_dir='../clothing1M/', train = False, val = False, test=True, transform = clothing_transform(img_sz))
	elif (dataset_name == "Food101N"):
		img_sz = 224
		train_dataset = Food101N(data_dir='../Food-101N_release/', train = True, val = False, test=False, transform = food_transform(img_sz))
		val_dataset = Food101N(data_dir='../Food-101N_release/', train = False, val = True, test=False, transform = food_transform(img_sz))
		test_dataset = Food101N(data_dir='../Food-101N_release/', train = False, val = False, test=True, transform = food_transform(img_sz))

	return train_dataset, val_dataset, test_dataset

def save_data(data, ID, epoch):
	[all_loss, all_acc, all_val_acc, all_test_acc] = data
	with open("loss_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	        pickle.dump(all_loss, f)
	
	with open("acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	        pickle.dump(all_acc, f)
	with open("clean_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	        pickle.dump(all_val_acc, f)
	with open("test_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_test_acc, f)


def analyzable(dataset_name):
	return ((dataset_name == "MNIST") or (dataset_name == "CIFAR10") or (dataset_name == "CIFAR100"))

def tsne(features, labels, num_classes):
	X_embedded = TSNE(n_components = 2, verbose = 1, n_jobs = 6, n_iter = 3000).fit_transform(features)
	for i in range(num_classes):
		ind_i = np.where(labels == i)
		ind_i = ind_i[0].tolist()
		X_i = X_embedded[ind_i, :]
		plt.scatter(X_i[:, 0], X_i[:, 1], label = str("class " + str(i)))
	plt.legend()
	plt.show()

def transform_feature_map(wtd_feature_map, img_sz):
	wtd_feature_map = wtd_feature_map.mean(1)
	wtd_feature_map = wtd_feature_map.unsqueeze(1)
	upsampled_feature_map = F.interpolate(wtd_feature_map, size = (img_sz, img_sz))
	heatmap = upsampled_feature_map
	heatmap = F.relu(heatmap)
	heatmap = heatmap / torch.max(heatmap)

	return heatmap

def visualize_CAM(img, wtd_feature_map_given_label, given_label, epoch, dataset_name):
	img_sz = img.size(-1)
	print(img_sz)
	heatmap_given_label = transform_feature_map(wtd_feature_map_given_label, img_sz)
	# print(img.size(), upsampled_feature_map.size())
	PIL_tr = transforms.ToPILImage()
	heatmap_given_label = heatmap_given_label.cpu().data.numpy()
	for i in range(5):
		print("Label: ", given_label[i])
		PILImage = PIL_tr((img.cpu())[i, :, :, :]*255)
		if dataset_name == "MNIST":
			cmap = 'gray'
		else:
			cmap = 'gnuplot'
		plt.imshow(PILImage, origin = 'upper', cmap = cmap)
		plt.imshow(heatmap_given_label[i, 0], interpolation='nearest',cmap = 'jet',origin='upper', alpha = 0.5)
		plt.colorbar(cmap='jet')
		plt.show()
		# plt.savefig(dataset_name + "/heatmap_idx_" + str(i)+"_" + "epoch_" + str(epoch)+".jpg")
		# plt.clf()