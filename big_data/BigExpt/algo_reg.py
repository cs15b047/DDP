import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from data.mnist import MNIST
from data.cifar import CIFAR10, CIFAR100
from clothing import Clothing
from food101 import Food101N

from torchsampler import ImbalancedDatasetSampler

import argparse, sys
import datetime
import shutil
import pickle
from tqdm import tqdm
import datetime
import argparse

from utils import *

import matplotlib.pyplot as plt

# from small_model import model

#####################################Initial Configuration####################################
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
parser.add_argument("--weight_method", type = str)
parser.add_argument("--strt_ep", type = int)
parser.add_argument("--load", type = int, required = False)
parser.add_argument("--percent", type = float, required = False)
parser.add_argument("--log_reg", type = float, required = False)
parser.add_argument("--ep", type = int, required = False)
parser.add_argument("--dropout_rate", type = float, required = False)
parser.add_argument("--wt_decay", type = float)
args = parser.parse_args()

num_workers, batch_size, num_models, devices, lr, weight_method, strt_ep, load,  percent, dataset_name, log_reg, ep, dropout_rate, wt_decay = args.num_workers, args.batch_size, args.num_models, args.devices, args.lr, args.weight_method, args.strt_ep, args.load, args.percent, args.dataset_name, args.log_reg, args.ep, args.dropout_rate, args.wt_decay

if dropout_rate == None:
	dropout_rate = 0

noise_rate, noise_type = None, None
if(analyzable(dataset_name)):
	noise_type = input("Enter noise type: (symmetric or pairflip)")
	noise_rate = float(input("Enter noise rate: "))
	noise_string = noise_type + "_" +str(int(noise_rate * 100))
	top_bn = False
	ID = dataset_name + "_" + noise_string
	n_epoch = 200

elif(dataset_name == "Clothes1M" or dataset_name == "Food101N"):
	ID = dataset_name
	n_epoch = 30 * 20
	###########Put in the regulzn value from graph##############
	reg = pow(10, log_reg)
	print(reg)
	############################################################
	###########Use noise rates in weighting###########
	if dataset_name == "Clothes1M":
		noise_rate = 1- 0.615
	elif dataset_name == "Food101N":
		noise_rate = 1 - 0.8

#############################Hyperparams####################################
# 1. Regularization value 2. NUmber of epochs to train on that regulzn value/When to start reweighting
# 3. How many indices in 1 epoch 4. Weighting method , 5. Dropout 6. Number of models
############################################################################

input_channel, num_classes, size = get_input_info(dataset_name)

total_classified, clean_classfied, dirty_classified = np.zeros((n_epoch,)), np.zeros((n_epoch,)), np.zeros((n_epoch,))
#########################################################################

################Re-weighting Function###################
def set_wts(method, probabilities, mxm, mnm, epoch, accuracy, noise_rate):
	#TODO: Remove noise rate dependence for this step
	if(noise_rate == None):
		noise_rate = 0

	acc_step = (1 - noise_rate)* 100
	print(int((accuracy*100)/acc_step))
	if(method == "linear"):
		wts = (probabilities - mnm) / (mxm - mnm)
	elif(method == "quadratic"):
		wts = ((probabilities - mnm) / (mxm - mnm))**2 #(2 + int((accuracy*100)/acc_step) )
	elif(method == "exponential"):
		###CHANGE:Added e - 1 in denom to normalize to between 0 to 1##########
		wts = (np.exp(((probabilities - mnm) / (mxm - mnm))) - 1) / (np.exp(1) - 1)
	else:
		wts = np.array([0.5]*len(probabilities))
	return wts
########################################################

####################Callback fn for imbalanced dataset sampler####################
def _get_labels(dataset_obj, idx):
	return dataset_obj.get_label(idx)
##################################################################################

########################################Data and Loader#####################################
train_dataset, val_dataset, test_dataset = get_datasets(dataset_name, noise_rate, noise_type)
if dataset_name == "Clothes1M":
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers,drop_last=False, pin_memory=True, sampler = ImbalancedDatasetSampler(train_dataset, callback_get_label = _get_labels))
elif dataset_name == "Food101N":
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers,drop_last=False, shuffle = True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,shuffle=False, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,shuffle=False, pin_memory=True)

if(analyzable(dataset_name)):
	clean_labels = train_dataset.train_labels
	clean_labels = clean_labels.reshape(-1)
	noise_or_not = train_dataset.noise_or_not
############################################################################################


############################Models and Optimizer#########################################
if(dataset_name == "Clothes1M" or dataset_name == "Food101N"):
	from big_model import model
	models = [model(image_size = size, input_channel = input_channel, n_outputs = num_classes, dropout_rate = dropout_rate) for i in range(num_models)]
else:
	from model import model
	models = [model(image_size = size, input_channel = input_channel, n_outputs = num_classes) for i in range(num_models)]

opt = [torch.optim.Adam(models[i].parameters(), lr = lr, weight_decay = wt_decay) for i in range(num_models)] ############# Weight decay --> hyperparam #################

models = [nn.DataParallel(mod, device_ids = devices) for mod in models]

for i in range(num_models):
	models[i].cuda()

#Load model from file
if(load == 1):
	for i in range(num_models):
		checkpoint = torch.load("checkpoint_" + "epoch_" + str(ep) + "_" + str(i) + "_" + str(ID) + ".pt", map_location = 'cpu')
		models[i].load_state_dict(checkpoint['model_state_dict'])
		opt[i].load_state_dict(checkpoint['opt_state_dict'])

#################################################################################################

##############Initializn#################
dataset_sz = len(train_dataset)
example_weights = np.array([0.5]*dataset_sz)
all_loss, all_loss_wo_reg, all_acc, all_test_acc, all_val_acc = [], [], [], [], []
val_loss, test_loss = [], [] 
all_clean_acc = []
##########################################

#################################MAIN LOOP###################################
for epoch in range(n_epoch):
	#####################INIT####################################
	print("epoch : " + str(epoch))
	Loss_wo_reg, Loss = np.zeros((num_models,)), np.zeros((num_models,))

	Acc_noisy, Acc_test = np.zeros((num_models,)), np.zeros((num_models,))
	all_mean = np.zeros((dataset_sz,))
	visited = np.zeros((dataset_sz,))
	mean_selective = []
	steps = 0
	
	if(analyzable(dataset_name)):
		Acc_clean = np.zeros((num_models,))
		clean_mean = np.array([])
		dirty_mean = np.array([])
		batch_clean_mean, batch_dirty_mean = [],[]
	##################################################################
	for i in range(num_models):
		models[i].train()
	##################################Training Loop##############################################
	for idx, (images, labels, ind) in enumerate((train_loader)):
		# Shorten an epoch
		if(idx >= int(percent * len(train_dataset)/batch_size)):
			break

		if(idx % int(0.1*len(train_dataset)/batch_size) == 0):
			print(idx)

		visited[ind] = 1

		batch_strt = datetime.datetime.now()
		labels = Variable(labels).cuda()
		images = Variable(images).cuda()

		# Whole probability data of batch
		prob_data = np.zeros((num_models, len(labels), num_classes))

		if(analyzable(dataset_name)):
			actual_labels = clean_labels[ind]
			batch_noise_or_not = noise_or_not[ind]
			noise = (actual_labels == labels.cpu().data.numpy())
			clean = (np.argwhere(batch_noise_or_not == True)).reshape(-1)
			dirty = (np.argwhere(batch_noise_or_not == False)).reshape(-1)

			prob_data_clean = np.zeros((num_models, len(clean), num_classes))
			prob_data_dirty = np.zeros((num_models, len(dirty), num_classes))

		strt_time = datetime.datetime.now()

		for i in range(num_models):
			pred = models[i](images)
			loss = F.cross_entropy(pred, labels, reduction='none')

			#Weigh Examples and calculate weighted average loss
			batch_wts = example_weights[ind]
			wts = torch.Tensor(batch_wts).cuda()

			###########################Regulzn Loss#################################################
			reg_loss = Variable(torch.tensor(0.0)).cuda()
			for p in models[i].parameters():
				reg_loss += (torch.norm(p))**2
			regulzn = reg * reg_loss
			############################################################################

			loss = loss * wts
			loss = torch.mean(loss)

			Loss_wo_reg[i] += loss.cpu().data

			# if epoch < strt_ep:
			#####################Loss with Regulzn####################
			loss = loss + regulzn
			#########################################
			if idx == int(percent * len(train_dataset)/batch_size) - 1:
				print(regulzn.cpu().data.numpy())

			opt[i].zero_grad()
			loss.backward()
			opt[i].step()

			prob = F.softmax(pred, dim = 1)
			prob = prob.cpu().data.numpy()
			prob_data[i] = prob
			pred_data = pred.cpu().data.numpy()
			pred_label = np.argmax(pred_data, axis=1)

			Loss[i] += loss.cpu().data
			Acc_noisy[i] += np.average(pred_label == labels.cpu().data.numpy())
			
			if(analyzable(dataset_name)):
				prob_data_clean[i] = prob[clean]
				prob_data_dirty[i] = prob[dirty]
				Acc_clean[i] += np.average(pred_label == actual_labels)

			if((i == 0) and analyzable(dataset_name)):
				base = pred_label == labels.cpu().data.numpy()
				tot, clean_done = np.sum(base), np.sum(base * batch_noise_or_not)
				total_classified[epoch] += tot
				clean_classfied[epoch] += clean_done
				dirty_classified[epoch] += tot - clean_done

		mean = np.mean(prob_data, axis = 0)
		selective_mean = mean[np.arange(len(labels)), (labels.cpu().data.numpy())]
		mean_selective += [np.average(selective_mean)]
		all_mean[ind] = selective_mean
		
		end_time = datetime.datetime.now()
		# print("Batch compute: ",end_time - strt_time)


		if(analyzable(dataset_name)):
			mean_clean = np.mean(prob_data_clean, axis=0)
			mean_dirty = np.mean(prob_data_dirty, axis=0)
			selective_clean = mean_clean[np.arange(len(clean)), (labels.cpu().data.numpy())[clean]]
			selective_dirty = mean_dirty[np.arange(len(dirty)), (labels.cpu().data.numpy())[dirty]]
			clean_mean = np.concatenate((clean_mean, selective_clean))
			dirty_mean = np.concatenate((dirty_mean, selective_dirty))
			batch_clean_mean += [np.average(selective_clean)]
			batch_dirty_mean += [np.average(selective_dirty)]

		steps += 1
		np.set_printoptions(precision = 3)
		#After batch
		# torch.cuda.empty_cache()
		batch_end = datetime.datetime.now()
		# print("Batch total: ",batch_end - batch_strt)
	#After train
	torch.cuda.empty_cache()
	############################################################################################
	for i in range(num_models):
		models[i].eval()
	###############################VALIDATION##########################
	if(dataset_name == "Clothes1M" or dataset_name == "Food101N"):
		val_steps = 0
		Val_Loss = np.zeros((num_models,))
		Acc_val = np.zeros((num_models,))

		with torch.no_grad():
			for idx, (images, labels, ind) in enumerate(tqdm(val_loader)):
				labels = Variable(labels).cuda()
				images = Variable(images).cuda()
				for i in range(num_models):
					pred = models[i](images)
					loss = F.cross_entropy(pred, labels)
					pred_data = pred.cpu().data.numpy()
					pred_label = np.argmax(pred_data, axis=1)
					batch_acc = np.average(pred_label == labels.cpu().data.numpy())
					Acc_val[i] += batch_acc
					Val_Loss[i] += loss.cpu().data
				val_steps += 1
		Acc_val = Acc_val/val_steps
		Val_Loss = Val_Loss / val_steps
		print("Val accuracy:", Acc_val)
		print("Val loss: ", Val_Loss)
		all_val_acc += [np.average(np.array(Acc_val))]
		val_loss += [np.average(np.array(Val_Loss))]
		with open("val_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
			pickle.dump(all_val_acc, f)
		with open("val_loss_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
			pickle.dump(val_loss, f)
	#After Validation
	torch.cuda.empty_cache()
	##################################################################

	################################TEST###############################
	test_steps = 0
	Test_Loss = np.zeros((num_models,))
	with torch.no_grad():
		for idx, (images, labels, ind) in enumerate(tqdm(test_loader)):
			labels = Variable(labels).cuda()
			images = Variable(images).cuda()
			for i in range(num_models):
				pred = models[i](images)
				loss = F.cross_entropy(pred, labels)
				pred_data = pred.cpu().data.numpy()
				pred_label = np.argmax(pred_data, axis=1)
				Acc_test[i] += np.average(pred_label == labels.cpu().data.numpy())
				Test_Loss[i] += loss.cpu().data
			test_steps += 1
	#After test
	torch.cuda.empty_cache()
	###################################################################

	#################################Model Save############################################
	for i in range(num_models):
		torch.save({'model_state_dict': models[i].state_dict(),'opt_state_dict': opt[i].state_dict()}, "checkpoint_" + "epoch_" + str(epoch) + "_" + str(i) + "_" + str(ID) + ".pt")
	#############################################################################

	Loss = Loss / steps
	Test_Loss = Test_Loss / test_steps

	Acc_noisy = Acc_noisy / steps
	Acc_test = Acc_test / test_steps
	
	# Re-weighing examples
	###############################################Re-weighting########################################
	if (epoch >= strt_ep):
		###################################Run inference through whole dataset and take avg probabilities for weighting#######################################
		infer_acc = np.zeros((num_models,))
		infer_Loss = np.zeros((num_models,))
		infer_steps = 0
		with torch.no_grad():
			for idx, (images, labels, ind) in enumerate((train_loader)):
				labels = Variable(labels).cuda()
				images = Variable(images).cuda()
				visited[ind] = 1
				prob_data = np.zeros((num_models, len(labels), num_classes))
				for i in range(num_models):
					pred = models[i](images)
					loss = F.cross_entropy(pred, labels)
					pred_data = pred.cpu().data.numpy()
					pred_label = np.argmax(pred_data, axis=1)
					batch_acc = np.average(pred_label == labels.cpu().data.numpy())
					infer_acc[i] += batch_acc
					infer_Loss[i] += loss.cpu().data

					prob = F.softmax(pred, dim = 1)
					prob = prob.cpu().data.numpy()
					prob_data[i] = prob

				mean = np.mean(prob_data, axis = 0)
				selective_mean = mean[np.arange(len(labels)), (labels.cpu().data.numpy())]
				all_mean[ind] = selective_mean

				infer_steps += 1

		acc = np.average(infer_acc)

		infer_Loss = infer_Loss/ infer_steps
		infer_acc = infer_acc/ infer_steps
		print("Infer Loss: ", infer_Loss)
		print("Infer Acc: ", infer_acc)

		###########################################################################################
	# Only calc from those which are visited
	mxm_mean, mnm_mean = np.amax(all_mean[visited == 1]), np.amin(all_mean[visited == 1])
	print("Max and Min Probability mean values: ", mxm_mean, mnm_mean)
	if(epoch >= strt_ep):
		example_weights[visited == 1] = set_wts(weight_method, all_mean[visited == 1], mxm_mean, mnm_mean, epoch - strt_ep, acc, noise_rate)
		num_visited_examples = np.sum(visited == 1)
		######################Epochwise Normalization of weights --DDP Review comment##################
		example_weights[visited == 1] = (example_weights[visited == 1] / np.sum(example_weights[visited == 1])) * (num_visited_examples / 2)
		###############################################################################################
	####################################################################################################

	############################# Printing and Saving section start###################################
	if(analyzable(dataset_name)):
		Acc_clean = Acc_clean / steps
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

		with open("clean_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
			pickle.dump(clean_mean, f)
		with open("dirty_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
			pickle.dump(dirty_mean, f)

		with open("classified_breakup_" + str(ID) +".pkl", "wb") as f:
			pickle.dump([total_classified[:epoch + 1], clean_classfied[:epoch + 1], dirty_classified[:epoch + 1]], f)

	with open("wts_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(example_weights, f)

	with open("all_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_mean, f)


	print(Loss)
	print(Test_Loss)
	print(Acc_noisy)
	print(Acc_test)
	all_loss += [np.average(np.array(Loss))]
	all_loss_wo_reg += [np.average(np.array(Loss_wo_reg))]
	test_loss += [np.average(np.array(Test_Loss))]
	all_acc += [np.average(np.array(Acc_noisy))]
	all_test_acc += [np.average(np.array(Acc_test))]
	
	if(analyzable(dataset_name)):
		print(Acc_clean)
		all_clean_acc += [np.average(np.array(Acc_clean))]
		with open("clean_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
			pickle.dump(all_clean_acc, f)

	with open("loss_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_loss, f)
	with open("loss_wo_reg_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_loss_wo_reg, f)
	with open("test_loss_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(test_loss, f)
	with open("acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_acc, f)
	with open("test_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_test_acc, f)
	####################### Printing and Saving section ends###########################
