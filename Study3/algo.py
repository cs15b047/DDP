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
from utils import get_datasets, get_gen_info, save_data, initializations, set_wt
from tqdm import tqdm
from PIL import Image
# from small_model import model

[noise_type, noise_rate, lr, weight_method, seed, num_workers, batch_size, num_models, dataset_name, noise_string, ID] = initializations()
n_epoch = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
input_channel, num_classes, size = get_gen_info(dataset_name)

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

example_weights = -np.ones((dataset_sz,))

total_clean = (np.argwhere(noise_or_not == True)).reshape(-1)
total_dirty = (np.argwhere(noise_or_not == False)).reshape(-1)

all_loss, all_acc, all_clean_acc, all_test_acc = [], [], [], []
Regularization = np.linspace(-1.8, -5, 28)
print(Regularization)
Regularization = [pow(10, x) for x in Regularization]
tot_epoch = n_epoch * len(Regularization)
total_classified, clean_classfied, dirty_classified = np.zeros((tot_epoch,)), np.zeros((tot_epoch,)), np.zeros((tot_epoch,))
total_agreed = np.zeros((tot_epoch,))
idx_reg = 0
Num_examples = []
Clean_examples = []
for epoch in range(n_epoch*len(Regularization)):
	print("epoch : " + str(epoch))
	#INit variables
	Loss = np.zeros((num_models,))
	Acc_noisy = np.zeros((num_models,))
	Acc_clean = np.zeros((num_models,))
	Acc_test = np.zeros((num_models,))
	steps = 0

	if(epoch % n_epoch == 0 and epoch !=0):
		idx_reg += 1
	reg = Regularization[idx_reg]
	print(reg)

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
		clean = (np.argwhere(batch_noise_or_not == True)).reshape(-1)
		dirty = (np.argwhere(batch_noise_or_not == False)).reshape(-1)

		# Whole probability data of batch
		prob_data = np.zeros((num_models, len(labels), num_classes))
		prob_data_clean = np.zeros((num_models, len(clean), num_classes))
		prob_data_dirty = np.zeros((num_models, len(dirty), num_classes))

		agreement = np.zeros((num_models, len(ind)))
		agreed = np.array([])

		for i in range(num_models):
			pred = models[i](images)
			# loss = F.cross_entropy(pred, labels, reduce=False)

			reg_loss = Variable(torch.tensor(0.0)).cuda()
			for p in models[i].parameters():
				reg_loss += torch.norm(p)
			regulzn = reg * reg_loss

			#Don't Weigh loss here!!!
			# loss = torch.mean(loss)
			loss = F.cross_entropy(pred, labels)

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
			
			base = (pred_label == labels.cpu().data.numpy())
			agreement[i] = base
			tot, clean_done = np.sum(base), np.sum(base * batch_noise_or_not)
			total_classified[epoch] += tot
			clean_classfied[epoch] += clean_done
			dirty_classified[epoch] += tot - clean_done

		# Assigning weights logic --> All models agree --> example classified
		# Linear wt from 0 to 1 assign according to example was classified in which th percentage of total
		curr_wt = set_wt(total_agreed[max(epoch - 1,0)], noise_rate, dataset_sz)
		selected = (np.sum(agreement, axis=0) >= num_models)
		correct_acc_to_data_ind = np.array(ind)[selected]
		not_classified_till_now_filter = correct_acc_to_data_ind[example_weights[correct_acc_to_data_ind] == -1]
		example_weights[not_classified_till_now_filter] = curr_wt
		total_agreed[epoch] += np.sum(selected)

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

	with open("wts_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump([example_weights, total_clean, total_dirty], f)

	# save and print loss and different accuracies
	Loss = Loss / steps;Acc_clean = Acc_clean / steps;Acc_noisy = Acc_noisy / steps;Acc_test = Acc_test / test_steps
	print(Loss);print(Acc_noisy);print(Acc_clean);print(Acc_test)
	all_loss += [np.average(np.array(Loss))];all_acc += [np.average(np.array(Acc_noisy))];all_clean_acc += [np.average(np.array(Acc_clean))];all_test_acc += [np.average(np.array(Acc_test))]

	# calculate # classified examples acc to noisy data
	done_examples = Acc_noisy * dataset_sz
	if(epoch % n_epoch == 0):
		Num_examples += [done_examples]
	print(Num_examples, Clean_examples)
	with open("num_classified_examples_with_reg_" + str(ID) + ".pkl", "wb") as f:
		pickle.dump([Num_examples, Regularization], f)

	save_data([all_loss, all_acc, all_clean_acc, all_test_acc, [total_classified, clean_classfied, dirty_classified]], ID, epoch)