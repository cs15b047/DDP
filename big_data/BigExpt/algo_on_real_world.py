import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.mnist import MNIST
from data.cifar import CIFAR10, CIFAR100

from torchsampler import ImbalancedDatasetSampler

import argparse, sys
import datetime
import shutil
import pickle
from big_model import model
from utils import get_datasets, get_input_info, save_data
from tqdm import tqdm
from PIL import Image

global noise_rate
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type = int)
parser.add_argument("--batch_size", type = int)
parser.add_argument("--num_models", type = int)
parser.add_argument("--devices", type = int, nargs = '+')
parser.add_argument("--dataset_name", type = str)
parser.add_argument("--lr", type = float)
parser.add_argument("--percent", type = float, required = False)
parser.add_argument("--n_epoch", type = int)
args = parser.parse_args()

num_workers, batch_size, num_models, devices, lr, percent, n_epoch, dataset_name = args.num_workers, args.batch_size, args.num_models, args.devices, args.lr, args.percent, args.n_epoch, args.dataset_name

noise_rate, noise_type = None, None
ID = dataset_name
input_channel, num_classes, size = get_input_info(dataset_name)

#################################################################################
def _get_labels(dataset_obj, idx):
	return dataset_obj.get_label(idx)
##################################################################################

########################################Data and Loader#####################################
train_dataset, val_dataset, test_dataset = get_datasets(dataset_name, noise_rate, noise_type)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers,drop_last=False, pin_memory=True, sampler = ImbalancedDatasetSampler(train_dataset, callback_get_label = _get_labels))
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,shuffle=False, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,shuffle=False, pin_memory=True)

dataset_sz = len(train_dataset)

models = [model(image_size = size, input_channel = input_channel, n_outputs = num_classes) for i in range(num_models)]
for i in range(num_models):
	models[i].cuda()
opt = [torch.optim.Adam(models[i].parameters(), lr = lr) for i in range(num_models)]
models = [nn.DataParallel(mod, device_ids = devices) for mod in models]

all_loss, all_loss_wo_reg, all_val_loss, all_acc, all_val_acc, all_test_acc = [], [], [], [], [], []

###################################################
Regularization = np.linspace(-2, -5, 25)
###################################################

print(Regularization)
Regularization = [pow(10, x) for x in Regularization]
tot_epoch = n_epoch * len(Regularization)

idx_reg = 0
Num_examples = []
for epoch in range(n_epoch*len(Regularization)):
	print("epoch : " + str(epoch))
	#Init variables
	Loss = np.zeros((num_models,))
	Loss_wo_reg = np.zeros((num_models,))
	Acc_noisy = np.zeros((num_models,))
	Acc_val = np.zeros((num_models,))
	Acc_test = np.zeros((num_models,))
	steps = 0

	if(epoch % n_epoch == 0 and epoch !=0):
		idx_reg += 1
	reg = Regularization[idx_reg]
	print(reg)
	all_mean = np.zeros((dataset_sz,))

	for i in range(num_models):
		models[i].train()

	for idx, (images, labels, ind) in enumerate((train_loader)):
		
		if(idx >= int(percent * len(train_dataset)/batch_size)):
			break

		labels = Variable(labels).cuda()
		images = Variable(images).cuda()

		for i in range(num_models):
			pred = models[i](images)

			reg_loss = Variable(torch.tensor(0.0)).cuda()
			for p in models[i].parameters():
				reg_loss += (torch.norm(p))**2
			
			regulzn = reg * reg_loss
			loss = F.cross_entropy(pred, labels)

			Loss_wo_reg[i] += loss.cpu().data

			loss = loss + regulzn

			opt[i].zero_grad()
			loss.backward()
			opt[i].step()

			pred_data = pred.cpu().data.numpy()
			pred_label = np.argmax(pred_data, axis=1)

			Loss[i] += loss.cpu().data
			Acc_noisy[i] += np.average(pred_label == labels.cpu().data.numpy())
			
			base = (pred_label == labels.cpu().data.numpy())

		steps += 1
		np.set_printoptions(precision = 2)
	torch.cuda.empty_cache()

	for i in range(num_models):
		models[i].eval()

	val_steps = 0
	Val_Loss = np.zeros((num_models,))
	with torch.no_grad():
		for idx, (images, labels, ind) in enumerate(tqdm(val_loader)):
			labels = Variable(labels).cuda()
			images = Variable(images).cuda()
			for i in range(num_models):
				pred = models[i](images)
				loss = F.cross_entropy(pred, labels)
				Val_Loss[i] += loss.cpu().data
				pred_data = pred.cpu().data.numpy()
				pred_label = np.argmax(pred_data, axis=1)
				Acc_val[i] += np.average(pred_label == labels.cpu().data.numpy())
			val_steps += 1
	torch.cuda.empty_cache()

	test_steps = 0
	with torch.no_grad():
		for idx, (images, labels, ind) in enumerate(tqdm(test_loader)):
			labels = Variable(labels).cuda()
			images = Variable(images).cuda()
			for i in range(num_models):
				pred = models[i](images)
				pred_data = pred.cpu().data.numpy()
				pred_label = np.argmax(pred_data, axis=1)
				Acc_test[i] += np.average(pred_label == labels.cpu().data.numpy())
			test_steps += 1
	torch.cuda.empty_cache()

	# save and print loss and different accuracies
	Loss = Loss / steps; Loss_wo_reg = Loss_wo_reg / steps; Val_Loss = Val_Loss / val_steps
	Acc_noisy = Acc_noisy / steps;Acc_val = Acc_val / val_steps; Acc_test = Acc_test / test_steps
	
	print(Loss);print(Acc_noisy);print(Acc_val);print(Acc_test)
	all_loss += [np.average(np.array(Loss))]; all_val_loss +=[np.average(np.array(Val_Loss))]
	all_acc += [np.average(np.array(Acc_noisy))];all_val_acc += [np.average(np.array(Acc_val))];all_test_acc += [np.average(np.array(Acc_test))]

	# calculate # classified examples acc to noisy data
	done_examples = Acc_noisy * dataset_sz
	if(epoch % n_epoch == 0):
		Num_examples += [done_examples]
	print(Num_examples)
	with open("num_classified_examples_with_reg_" + str(ID) + ".pkl", "wb") as f:
		pickle.dump([Num_examples, Regularization], f)

	save_data([[all_loss, all_loss_wo_reg, all_val_loss], all_acc, all_val_acc, all_test_acc], ID, epoch)