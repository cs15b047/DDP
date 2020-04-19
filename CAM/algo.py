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
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import SubsetRandomSampler

# from model import model
from model_restruct import model
from pytorch_grad_cam import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2

# For Now
noise_type = input("Enter noise type: (symmetric or pairflip)")
noise_rate = float(input("Enter noise rate: "))
lr = float(input("Enter learning rate: "))
weight_method = input("Enter example weighing method")
dataset_name = input("Enter Dataset name: ")
strt_ep = int(input("Enter reweight start epoch : "))
noise_string = noise_type + "_" +str(int(noise_rate * 100))
top_bn = False
epoch_decay_start = 80
n_epoch = 200
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
num_workers = 4
batch_size = 64

num_models = 4

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

strt_acc = (1 - noise_rate)/2

total_classified, clean_classfied, dirty_classified = np.zeros((n_epoch,)), np.zeros((n_epoch,)), np.zeros((n_epoch,))

ID = dataset_name + "_" + noise_string

def transform_feature_map(wtd_feature_map, img_sz):
	wtd_feature_map = wtd_feature_map.mean(1)
	wtd_feature_map = wtd_feature_map.unsqueeze(1)
	upsampled_feature_map = F.interpolate(wtd_feature_map, size = (img_sz, img_sz))
	heatmap = upsampled_feature_map
	heatmap = F.relu(heatmap)
	heatmap = heatmap / torch.max(heatmap)

	return heatmap

def visualize_CAM(img, wtd_feature_map_given_label, wtd_feature_map_actual_label, given_label, cleanness, epoch):
	img_sz = img.size(-1)
	print(img_sz)
	heatmap_given_label, heatmap_actual_label = transform_feature_map(wtd_feature_map_given_label, img_sz), transform_feature_map(wtd_feature_map_actual_label, img_sz)
	# print(img.size(), upsampled_feature_map.size())
	PIL_tr = transforms.ToPILImage()
	heatmap_given_label, heatmap_actual_label = heatmap_given_label.cpu().data.numpy(), heatmap_actual_label.cpu().data.numpy()
	for i in range(10):
		print("Label: ", given_label[i], cleanness[i])
		PILImage = PIL_tr((img.cpu())[i, :, :, :]*255)
		if dataset_name == "MNIST":
			cmap = 'gray'
		elif dataset_name == "CIFAR10":
			cmap = 'gnuplot'
		plt.imshow(PILImage, origin = 'upper', cmap = cmap)
		plt.imshow(heatmap_given_label[i, 0], interpolation='nearest',cmap = 'jet',origin='upper', alpha = 0.15)
		plt.colorbar(cmap='jet')
		plt.savefig(dataset_name + "/heatmap_idx_" + str(i)+"_" + "epoch_" + str(epoch)+".jpg")
		plt.clf()
		if not cleanness[i]:
			plt.imshow(PILImage, origin = 'upper', cmap = cmap)
			plt.imshow(heatmap_actual_label[i, 0], interpolation='nearest',cmap = 'jet',origin='upper', alpha = 0.15)
			plt.colorbar(cmap='jet')
			plt.savefig(dataset_name + "/heatmap_actual_label_idx_" + str(i)+"_" + "epoch_" + str(epoch)+".jpg")
			plt.clf()

def get_transform():
	choice = transforms.RandomChoice([
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(degrees=30),
		])
	return transforms.Compose([transforms.Resize(size = size), choice, transforms.ToTensor()])

def MNIST_transform():
	choice = transforms.RandomChoice([
		transforms.RandomRotation(degrees=30),
		])
	return transforms.Compose([transforms.Resize(size = 2 * size), choice, transforms.ToTensor()])



def MNIST_test_transform():
	return transforms.Compose([transforms.Resize(size = 2 * size), transforms.ToTensor()])

def get_datasets(dataset_name):
	if(dataset_name == "MNIST"):
		train_dataset = MNIST(root='./data/',download=True,  train=True, transform=MNIST_transform(),noise_type=noise_type,noise_rate=noise_rate)
		test_dataset = MNIST(root='./data/',download=True,  train=False, transform=MNIST_test_transform(),noise_type=noise_type,noise_rate=noise_rate)
	elif (dataset_name == "CIFAR10"):
		train_dataset = CIFAR10(root='./data/',download=True,  train=True, transform=get_transform(),noise_type=noise_type,noise_rate=noise_rate)
		test_dataset = CIFAR10(root='./data/',download=True,  train=False, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
	elif (dataset_name == "CIFAR100"):
		train_dataset = CIFAR100(root='./data/',download=True,  train=True, transform=get_transform(),noise_type=noise_type,noise_rate=noise_rate)
		test_dataset = CIFAR100(root='./data/',download=True,  train=False, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
	
	return train_dataset, test_dataset

def set_wts(method, probabilities, mxm, mnm, epoch, accuracy):
	acc_step = (1 - noise_rate)* 100
	print(int((accuracy*100)/acc_step))
	if(method == "linear"):
		wts = (probabilities - mnm) / (mxm - mnm)
	elif(method == "quadratic"):
		wts = ((probabilities - mnm) / (mxm - mnm))**(2 + int((accuracy*100)/acc_step) )
	elif(method == "exponential"):
		wts = np.exp(((probabilities - mnm) / (mxm - mnm))) - 1
	else:
		wts = np.array([0.5]*len(probabilities))
	return wts


train_dataset, test_dataset = get_datasets(dataset_name)
clean_labels = train_dataset.train_labels
clean_labels = clean_labels.reshape(-1)

dataset_sz = len(train_dataset.train_noisy_labels)
example_weights = np.array([0.5]*dataset_sz)

#validation split
split_fraction = 0.05
split = int(split_fraction * dataset_sz)
indices = np.arange(dataset_sz)
np.random.seed(seed)
np.random.shuffle(indices)

train_ind, val_ind = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_ind)
valid_sampler = SubsetRandomSampler(val_ind)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers,drop_last=False,sampler= train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers,drop_last=False,sampler= valid_sampler)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,shuffle=False)


noise_or_not = train_dataset.noise_or_not
models = [model(image_size = size, input_channel = input_channel, n_outputs = num_classes) for i in range(num_models)]
for i in range(num_models):
	models[i].cuda()
opt = [torch.optim.Adam(models[i].parameters(), lr = lr, weight_decay = 0.0001) for i in range(num_models)]


all_loss, all_acc, all_clean_acc, all_test_acc = [], [], [], []
for epoch in range(n_epoch):
	mod = deepcopy(models[0])
	grad_cam, gb_model = setup_GCAM(mod, ["0"], size)
	print("epoch : " + str(epoch))
	Loss = np.zeros((num_models,))
	Acc_noisy = np.zeros((num_models,))
	Acc_clean = np.zeros((num_models,))
	Acc_test = np.zeros((num_models,))
	Acc_valid = np.zeros((num_models,))
	steps = 0

	mean_selective = [] 
	all_mean = np.zeros((dataset_sz,))
	clean_mean = np.array([])
	dirty_mean = np.array([])
	batch_clean_mean, batch_dirty_mean = [],[]

	for idx, (images, labels, ind) in enumerate(tqdm(train_loader)):
		labels = Variable(labels).cuda()
		images = Variable(images).cuda()
		actual_labels = clean_labels[ind]
		batch_noise_or_not = noise_or_not[ind]
		noise = (actual_labels == labels.cpu().data.numpy())
		clean = (np.argwhere(batch_noise_or_not == True)).reshape(-1)
		dirty = (np.argwhere(batch_noise_or_not == False)).reshape(-1)

		# Whole probability data of batch
		prob_data = np.zeros((num_models, len(labels), num_classes))
		prob_data_clean = np.zeros((num_models, len(clean), num_classes))
		prob_data_dirty = np.zeros((num_models, len(dirty), num_classes))

		for i in range(num_models):
			pred, _ = models[i](images)
			loss = F.cross_entropy(pred, labels, reduce=False)

			#Weigh Examples and calculate weighted average loss
			batch_wts = example_weights[ind]
			wts = torch.Tensor(batch_wts).cuda()
			loss = loss * wts
			loss = torch.mean(loss)

			opt[i].zero_grad()
			loss.backward()
			opt[i].step()

			prob = F.softmax(pred, dim = 1)
			prob = prob.cpu().data.numpy()
			
			prob_data[i] = prob
			prob_data_clean[i] = prob[clean]
			prob_data_dirty[i] = prob[dirty]

			pred_data = pred.cpu().data.numpy()
			pred_label = np.argmax(pred_data, axis=1)

			Loss[i] += loss.cpu().data
			Acc_noisy[i] += np.average(pred_label == labels.cpu().data.numpy())
			Acc_clean[i] += np.average(pred_label == actual_labels)
			if(i == 0):
				base = pred_label == labels.cpu().data.numpy()
				tot, clean_done = np.sum(base), np.sum(base * batch_noise_or_not)
				total_classified[epoch] += tot
				clean_classfied[epoch] += clean_done
				dirty_classified[epoch] += tot - clean_done

		mean = np.mean(prob_data, axis = 0)
		mean_clean = np.mean(prob_data_clean, axis=0)
		mean_dirty = np.mean(prob_data_dirty, axis=0)

		selective_mean = mean[np.arange(len(labels)), (labels.cpu().data.numpy())]
		selective_clean = mean_clean[np.arange(len(clean)), (labels.cpu().data.numpy())[clean]]
		selective_dirty = mean_dirty[np.arange(len(dirty)), (labels.cpu().data.numpy())[dirty]]
		mean_selective += [np.average(selective_mean)]
		all_mean[ind] = selective_mean
		clean_mean = np.concatenate((clean_mean, selective_clean))
		dirty_mean = np.concatenate((dirty_mean, selective_dirty))
		batch_clean_mean += [np.average(selective_clean)]
		batch_dirty_mean += [np.average(selective_dirty)]

		steps += 1
		np.set_printoptions(precision = 3)

	torch.cuda.empty_cache()

	valid_steps = 0
	for idx, (images, labels, ind) in enumerate(valid_loader):
		labels = Variable(labels).cuda()
		images = Variable(images).cuda()
		for i in range(num_models):
			pred, features = models[i](images)
			pred_data = pred.cpu().data.numpy()
			pred_label = np.argmax(pred_data, axis=1)
			Acc_valid[i] += np.average(pred_label == labels.cpu().data.numpy())

			# Visualize attention over images using CAM
			if(i == 2 and idx == 0):
				linear_layer_wts = models[i].l_c1.weight
				appropriate_wts, actual_class_wts = linear_layer_wts[labels, :], linear_layer_wts[clean_labels[ind], :]
				appropriate_wts, actual_class_wts = (appropriate_wts.unsqueeze(-1)).unsqueeze(-1), (actual_class_wts.unsqueeze(-1)).unsqueeze(-1)
				wtd_feature_map_given_label, wtd_feature_map_actual_label = features * appropriate_wts, features * actual_class_wts
				# visualize_CAM(images, wtd_feature_map_given_label, wtd_feature_map_actual_label, labels, noise_or_not[ind], epoch)
				GCAM(mod, images.cpu(), labels, clean_labels[ind], grad_cam, gb_model)

		valid_steps += 1
	torch.cuda.empty_cache()


	test_steps = 0
	for idx, (images, labels, ind) in enumerate(test_loader):
		labels = Variable(labels).cuda()
		images = Variable(images).cuda()
		for i in range(num_models):
			pred, _ = models[i](images)
			pred_data = pred.cpu().data.numpy()
			pred_label = np.argmax(pred_data, axis=1)
			Acc_test[i] += np.average(pred_label == labels.cpu().data.numpy())
		test_steps += 1
	torch.cuda.empty_cache()

	Loss = Loss / steps
	Acc_clean = Acc_clean / steps
	Acc_noisy = Acc_noisy / steps
	Acc_test = Acc_test / test_steps
	Acc_valid = Acc_valid / valid_steps

	# Re-weighing examples
	mxm_mean, mnm_mean = np.amax(all_mean), np.amin(all_mean)
	print("Max and Min Probability mean values: ",mxm_mean, mnm_mean)

	acc = np.average(Acc_noisy)
	if(epoch >= strt_ep): # temporary expt
		# Method 1: Uniform change in weights
		example_weights = set_wts(weight_method, all_mean, mxm_mean, mnm_mean, epoch - strt_ep, acc)

		#Change wts according to rate
		# example_weights = (1 - curr_rate)* example_weights + new_example_weights * curr_rate

	percent_noise = (noise_rate*100)
	cutoff = np.percentile(example_weights, percent_noise)
	print(cutoff)
	pred_clean = (example_weights >= cutoff)

	# Calculate separation accuracy, precision, recall
	separation_accuracy = np.average(noise_or_not == pred_clean)
	separation_accuracy_clean = np.sum((noise_or_not == True) * (noise_or_not == pred_clean)) / np.sum((noise_or_not == True))
	separation_accuracy_dirty = np.sum((noise_or_not == False) * (noise_or_not == pred_clean)) / np.sum((noise_or_not == False))
	separation_precision_clean = np.sum((noise_or_not == True) * (noise_or_not == pred_clean)) / np.sum((pred_clean == True))
	separation_precision_dirty = np.sum((noise_or_not == False) * (noise_or_not == pred_clean)) / np.sum((pred_clean == False))
	print("Separation accuracy clean: "+ str(separation_accuracy_clean))
	print("Separation accuracy dirty: "+ str(separation_accuracy_dirty))
	print("Separation precision clean: "+ str(separation_precision_clean))
	print("Separation precision dirty: "+ str(separation_precision_dirty))
	print("Separation accuracy overall: "+ str(separation_accuracy))

	# with open("wts_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	# 	pickle.dump(example_weights, f)

	# with open("all_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	# 	pickle.dump(all_mean, f)

	# with open("clean_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	# 	pickle.dump(clean_mean, f)
	# with open("dirty_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	# 	pickle.dump(dirty_mean, f)

	# with open("classified_breakup_" + str(ID) +".pkl", "wb") as f:
	# 	pickle.dump([total_classified[:epoch + 1], clean_classfied[:epoch + 1], dirty_classified[:epoch + 1]], f)

	print(Loss)
	print(Acc_noisy)
	print(Acc_valid)
	print(Acc_clean)
	print(Acc_test)
	all_loss += [np.average(np.array(Loss))]
	all_acc += [np.average(np.array(Acc_noisy))]
	all_clean_acc += [np.average(np.array(Acc_clean))]
	all_test_acc += [np.average(np.array(Acc_test))]
	# with open("loss_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	# 	pickle.dump(all_loss, f)

	# with open("acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	# 	pickle.dump(all_acc, f)
	# with open("clean_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	# 	pickle.dump(all_clean_acc, f)
	# with open("test_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
	# 	pickle.dump(all_test_acc, f)
