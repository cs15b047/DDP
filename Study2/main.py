import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.mnist import MNIST
import argparse, sys
import datetime
import shutil
import pickle
# from model import model
from small_model import model

# For Now
noise_type = "pairflip"
noise_rate = 0.4

input_channel=1
num_classes=10
top_bn = False
epoch_decay_start = 80
n_epoch = 200
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
num_workers = 4
batch_size = 128
num_models = 8


def get_datasets():
	train_dataset = MNIST(root='./data/',download=True,  train=True, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
	test_dataset = MNIST(root='./data/',download=True,  train=False, transform=transforms.ToTensor(),noise_type=noise_type,noise_rate=noise_rate)
	return train_dataset, test_dataset

train_dataset, test_dataset = get_datasets()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers,drop_last=True,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=True,shuffle=False)

noise_or_not = train_dataset.noise_or_not

models = [model(input_channel = 1, n_outputs = 10) for i in range(num_models)]
for i in range(num_models):
	models[i].cuda()
opt = [torch.optim.Adam(models[i].parameters()) for i in range(num_models)]

clean_labels = train_dataset.train_labels
clean_labels = clean_labels.reshape(-1)

for epoch in range(n_epoch):
	print("epoch : " + str(epoch))
	Loss = np.zeros((num_models,))
	Acc_noisy = np.zeros((num_models,))
	Acc_clean = np.zeros((num_models,))
	Acc_test = np.zeros((num_models,))
	steps = 0

	accuracy_clean_or_noise = 0

	batchwise_var_clean = []
	batchwise_var_dirty = []

	variance_inside_clean = np.array([])
	variance_inside_dirty = np.array([])
	variance_clean = np.array([])
	variance_dirty = np.array([])

	variance_selective_clean = []
	variance_selective_dirty = []
	
	avg_var_clean, avg_var_dirty = 0, 0
	avg_inside_var_c, avg_inside_var_d = 0, 0
	tot_selective_var_c, tot_selective_var_d = 0, 0

	for idx, (images, labels, ind) in enumerate(train_loader):
		labels = Variable(labels).cuda()
		images = Variable(images).cuda()

		actual_labels = clean_labels[ind]


		prob_all_clean = np.zeros((num_models, num_classes))
		prob_all_noise = np.zeros((num_models, num_classes))
		batch_noise_or_not = noise_or_not[ind]
		
		noise = (actual_labels == labels.cpu().data.numpy())
		
		clean = (np.argwhere(batch_noise_or_not == True)).reshape(-1)
		dirty = (np.argwhere(batch_noise_or_not == False)).reshape(-1)

		prob_data_clean = np.zeros((num_models, len(clean), num_classes))
		prob_data_dirty = np.zeros((num_models, len(dirty), num_classes))

		for i in range(num_models):
			pred = models[i](images)
			loss = F.cross_entropy(pred, labels)
			opt[i].zero_grad()
			loss.backward()
			opt[i].step()

			prob = F.softmax(pred, dim = 1)
			prob = prob.cpu().data.numpy()

			prob_clean = prob[clean]
			prob_dirty = prob[dirty]
			
			prob_data_clean[i] = prob_clean
			prob_data_dirty[i] = prob_dirty

			prob_all_clean[i] = prob[clean[0]]
			prob_all_noise[i] = prob[dirty[0]]

			Loss[i] += loss.cpu().data

			pred_data = pred.cpu().data.numpy()
			pred_label = np.argmax(pred_data, axis=1)
			Acc_noisy[i] += np.average(pred_label == labels.cpu().data.numpy())
			Acc_clean[i] += np.average(pred_label == actual_labels)
		
		avg_prob_clean = np.average(prob_data_clean, axis = 0)
		avg_prob_dirty = np.average(prob_data_dirty, axis = 0)

		var_c = np.var(prob_data_clean, axis = 0)
		
		selective_var_c = var_c[np.arange(len(clean)), (labels.cpu().data.numpy())[clean]]
		var_c = np.sum(var_c, axis = 1)

		var_d = np.var(prob_data_dirty, axis = 0)
		selective_var_d = var_d[np.arange(len(dirty)), (labels.cpu().data.numpy())[dirty]]
		var_d = np.sum(var_d, axis = 1)
		variance_clean = np.concatenate((variance_clean, var_c))
		variance_dirty = np.concatenate((variance_dirty, var_d))

		avg_var_clean += np.average(var_c)
		avg_var_dirty += np.average(var_d)

		batchwise_var_clean += [np.average(var_c)]
		batchwise_var_dirty += [np.average(var_d)]
		


		inside_var_c = np.sum(np.var(prob_data_clean, axis=2), axis = 0)
		inside_var_d = np.sum(np.var(prob_data_dirty, axis=2), axis = 0)
		variance_inside_clean = np.concatenate((variance_inside_clean, inside_var_c))
		variance_inside_dirty = np.concatenate((variance_inside_dirty, inside_var_d))

		avg_inside_var_c += np.average(inside_var_c)
		avg_inside_var_d += np.average(inside_var_d)

		tot_selective_var_c += np.average(selective_var_c)
		tot_selective_var_d += np.average(selective_var_d)

		variance_selective_clean += [np.average(selective_var_c)]
		variance_selective_dirty += [np.average(selective_var_d)]

		steps += 1
		np.set_printoptions(precision = 2)
		if(idx % 100 == 0):
			print("Iter : " + str(idx))
			print("Clean : class = " + str(actual_labels[clean[0]]) )
			prob_all_clean = np.around(prob_all_clean, decimals = 2)
			prob_all_noise = np.around(prob_all_noise, decimals = 2)
			print(prob_all_clean)
			print("Noisy with Actual label = " + str(actual_labels[dirty[0]]) + " and noise label = " + str(labels[dirty[0]]))
			print(prob_all_noise)
			print("Variance of examples : " + str(var_c[0]) + " " + str(var_d[0]))
			print("Batch variance : Clean : " + str(np.average(var_c)) + ", Dirty: " + str(np.average(var_d)))
			print("Example selective variance : Clean : " + str(selective_var_c[0]) + ", Dirty : " + str(selective_var_d[0]))
			print("Selective Batch variance : " + str(np.average(selective_var_c)) + " " + str(np.average(selective_var_d)))
			input()

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

	Loss = Loss / steps
	Acc_clean = Acc_clean / steps
	Acc_noisy = Acc_noisy / steps
	Acc_test = Acc_test / test_steps

	with open("selective_variance_clean_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(variance_selective_clean, f)
	with open("selective_variance_dirty_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(variance_selective_dirty, f)

	# avg_var_clean = avg_var_clean / steps
	# avg_var_dirty = avg_var_dirty / steps
	
	print("Avg Loss : "+ str(Loss))
	print("Train Noisy Accuracy (avg of epoch) : "+ str(Acc_noisy))
	print("Train Clean Accuracy (avg of epoch) : "+ str(Acc_clean))
	print("Test Accuracy (avg of epoch) : "+ str(Acc_test))

	print("Variance clean: " + str(avg_var_clean) + ", dirty: " + str(avg_var_dirty))
	print("Only label Variance clean: " + str(avg_inside_var_c) + ", dirty: " + str(avg_inside_var_d))
	print("Only label Variance clean: " + str(tot_selective_var_c) + ", dirty: " + str(tot_selective_var_d))