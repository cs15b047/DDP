import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.mnist import MNIST
from data.cifar import CIFAR10, CIFAR100
import argparse, sys
import datetime
import shutil
import pickle
from model import model
from tqdm import tqdm
from PIL import Image

def get_datasets(dataset_name, noise_type, noise_rate):
	if(dataset_name == "MNIST"):
		train_dataset = MNIST(root='./data/',download=True,  train=True, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
		test_dataset = MNIST(root='./data/',download=True,  train=False, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
	elif (dataset_name == "CIFAR10"):
		train_dataset = CIFAR10(root='./data/',download=True,  train=True, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
		test_dataset = CIFAR10(root='./data/',download=True,  train=False, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
	elif (dataset_name == "CIFAR100"):
		train_dataset = CIFAR100(root='./data/',download=True,  train=True, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
		test_dataset = CIFAR100(root='./data/',download=True,  train=False, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
	# elif(dataset_name == "Clothes1M"):
	# 	train_dataset, test_dataset = 
	return train_dataset, test_dataset

def get_gen_info(dataset_name):
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
	return input_channel, num_classes, size

def save_data(data, ID, epoch):
	[all_loss, all_acc, all_clean_acc, all_test_acc, classified_numbers] = data
	[total_classified, clean_classfied, dirty_classified] = classified_numbers
	with open("loss_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	        pickle.dump(all_loss, f)
	
	with open("acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	        pickle.dump(all_acc, f)
	with open("clean_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	        pickle.dump(all_clean_acc, f)
	with open("test_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_test_acc, f)

	with open("classified_breakup.pkl", "wb") as f:
		pickle.dump([total_classified[:epoch + 1], clean_classfied[:epoch + 1], dirty_classified[:epoch + 1]], f)


def initializations():
	noise_type = input("Enter noise type: (symmetric or pairflip)")
	noise_rate = float(input("Enter noise rate: "))
	lr = float(input("Enter learning rate: "))
	weight_method = input("Enter example weighing method")
	seed = 1
	num_workers = 4
	batch_size = 128
	num_models = 4
	dataset_name = input("Enter Dataset name: ")
	noise_string = noise_type + "_" +str(int(noise_rate * 100))
	ID = dataset_name + "_" + noise_string
	return [noise_type, noise_rate, lr, weight_method, seed, num_workers, batch_size, num_models, dataset_name, noise_string, ID]

def set_wt(classified, noise_rate, dataset_sz):
	portion = classified / ((1 - noise_rate)*dataset_sz)
	if(portion > 1):
		return 0.1
	else:
		return 1 - portion

def reweight(method, probabilities, mxm, mnm, epoch):
	if(method == "linear"):
		wts = (probabilities - mnm) / (mxm - mnm)
	elif(method == "quadratic"):
		wts = ((probabilities - mnm) / (mxm - mnm))**(2) # + int(epoch/10)
	elif(method == "exponential"):
		wts = np.exp(((probabilities - mnm) / (mxm - mnm))) - 1
	else:
		wts = np.array([0.5]*len(probabilities))
	return wts

def separation_analysis(noise_rate, noise_or_not, example_weights):
	percent_noise = (noise_rate*100)
	cutoff = np.percentile(example_weights, percent_noise)
	pred_clean = (example_weights >= cutoff)

	# Calculate separation accuracy, precision, recall
	separation_accuracy = np.average(noise_or_not == pred_clean)
	separation_accuracy_clean = np.sum((noise_or_not == True) * (noise_or_not == pred_clean)) / np.sum((noise_or_not == True))
	separation_accuracy_dirty = np.sum((noise_or_not == False) * (noise_or_not == pred_clean)) / np.sum((noise_or_not == False))
	separation_precision_clean = np.sum((noise_or_not == True) * (noise_or_not == pred_clean)) / np.sum((pred_clean == True))
	separation_precision_dirty = np.sum((noise_or_not == False) * (noise_or_not == pred_clean)) / np.sum((pred_clean == False))

	return [cutoff, separation_accuracy, separation_accuracy_clean, separation_accuracy_dirty, separation_precision_clean, separation_precision_dirty]
