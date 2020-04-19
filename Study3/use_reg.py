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
from utils import get_datasets, get_gen_info, save_data, initializations, separation_analysis, reweight
from tqdm import tqdm
from PIL import Image
# from small_model import model

[noise_type, noise_rate, lr, weight_method, seed, num_workers, batch_size, num_models, dataset_name, noise_string, ID] = initializations()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
input_channel, num_classes, size = get_gen_info(dataset_name)
n_epoch = 200
total_classified, clean_classfied, dirty_classified = np.zeros((n_epoch,)), np.zeros((n_epoch,)), np.zeros((n_epoch,))

train_dataset, test_dataset = get_datasets(dataset_name, noise_type, noise_rate)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers,drop_last=False,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,shuffle=False)

noise_or_not = train_dataset.noise_or_not
clean_labels = train_dataset.train_labels
clean_labels = clean_labels.reshape(-1)
dataset_sz = len(train_dataset.train_noisy_labels)

models = [model(image_size = size, input_channel = input_channel, n_outputs = num_classes) for i in range(num_models)]
for i in range(num_models):
	models[i].cuda()
opt = [torch.optim.Adam(models[i].parameters(), lr = lr) for i in range(num_models)]

# tot_prev_epochs = 149

# with open("wts_" + ID + "_"+str(tot_prev_epochs)+".pkl", "rb") as f:
# 	[example_weights, all_clean, all_dirty] = pickle.load(f)

example_weights = np.array([0.5]*dataset_sz)

log_reg = float(input("Enter log of regulzn coeff"))
reg = pow(10, log_reg)

strt_time = 20

all_loss, all_acc, all_clean_acc, all_test_acc = [], [], [], []

for epoch in range(n_epoch):
	print("epoch : " + str(epoch))
	Loss = np.zeros((num_models,))
	Acc_noisy = np.zeros((num_models,))
	Acc_clean = np.zeros((num_models,))
	Acc_test = np.zeros((num_models,))
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
			pred = models[i](images)
			loss = F.cross_entropy(pred, labels, reduce=False)

			#Weigh Examples and calculate weighted average loss
			batch_wts = example_weights[ind]
			wts = torch.Tensor(batch_wts).cuda()
			loss = loss * wts
			loss = torch.mean(loss)

			reg_loss = Variable(torch.tensor(0.0)).cuda()
			for p in models[i].parameters():
				reg_loss += torch.norm(p)
			regulzn = reg * reg_loss

			#Regularizn added
			loss = loss + regulzn

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
		np.set_printoptions(precision = 2)

	test_steps = 0
	for idx, (images, labels, ind) in enumerate(test_loader):
		labels = Variable(labels).cuda()
		images = Variable(images).cuda()
		for i in range(num_models):
			pred = models[i](images)
			pred_data = pred.cpu().data.numpy()
			pred_label = np.argmax(pred_data, axis=1)
			Acc_test[i] += np.average(pred_label == labels.cpu().data.numpy())
		test_steps += 1

	# Re-weighing examples
	mxm_mean, mnm_mean = np.amax(all_mean), np.amin(all_mean)

	if(epoch >= strt_time):
		# Method 1: Uniform change in weights
		example_weights = reweight(weight_method, all_mean, mxm_mean, mnm_mean, epoch)
	
	[cutoff, separation_accuracy, separation_accuracy_clean, separation_accuracy_dirty, separation_precision_clean, separation_precision_dirty] = separation_analysis(noise_rate, noise_or_not, example_weights)
	print("Separation accuracy clean: "+ str(separation_accuracy_clean))
	print("Separation accuracy dirty: "+ str(separation_accuracy_dirty))
	print("Separation precision clean: "+ str(separation_precision_clean))
	print("Separation precision dirty: "+ str(separation_precision_dirty))
	print("Separation accuracy overall: "+ str(separation_accuracy))

	with open("wts_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(example_weights, f)

	with open("all_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_mean, f)

	with open("clean_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(clean_mean, f)
	with open("dirty_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(dirty_mean, f)

	Loss = Loss / steps;Acc_clean = Acc_clean / steps;Acc_noisy = Acc_noisy / steps;Acc_test = Acc_test / test_steps
	print(Loss);print(Acc_noisy);print(Acc_clean);print(Acc_test)
	all_loss += [np.average(np.array(Loss))];all_acc += [np.average(np.array(Acc_noisy))];all_clean_acc += [np.average(np.array(Acc_clean))];all_test_acc += [np.average(np.array(Acc_test))]

	save_data([all_loss, all_acc, all_clean_acc, all_test_acc, [total_classified, clean_classfied, dirty_classified]], ID, epoch)