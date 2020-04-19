import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

# from data.mnist import MNIST
# from data.cifar import CIFAR10, CIFAR100
from clothing import Clothing

from torchsampler import ImbalancedDatasetSampler
from siamese_triplet import losses, utils

import argparse, sys
import datetime
import shutil
import pickle
from tqdm import tqdm
import datetime

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
parser.add_argument("--weight_method", type = str)
parser.add_argument("--lr", type = float)
parser.add_argument("--lamda", type = float)
parser.add_argument("--weight_decay", type = float)
parser.add_argument("--percent", type = float, required = False)
parser.add_argument("--n_epoch", type = int)
parser.add_argument("--strt_ep", type = int)
parser.add_argument("--load", type = int)
parser.add_argument("--ep", type = int, required = False)
parser.add_argument("--alpha", type = float)
args = parser.parse_args()

num_workers, batch_size, num_models, devices, lr, percent, n_epoch, dataset_name, weight_method, strt_ep, lamda, load, ep, weight_decay, alpha = args.num_workers, args.batch_size, args.num_models, args.devices, args.lr, args.percent, args.n_epoch, args.dataset_name, args.weight_method, args.strt_ep, args.lamda, args.load, args.ep, args.weight_decay, args.alpha
clean_epochs = strt_ep

noise_rate, noise_type = None, None
ID = dataset_name
if(dataset_name != "Clothes1M"):
	print("Not supported yet!!")
	exit(0)

input_channel, num_classes, size = get_input_info(dataset_name)

total_classified, clean_classfied, dirty_classified = np.zeros((n_epoch,)), np.zeros((n_epoch,)), np.zeros((n_epoch,))
#########################################################################

################Re-weighting Function###################
def set_wts(method, probabilities, mxm, mnm, epoch, accuracy, noise_rate):
	#TODO: Remove noise rate dependence for this step
	if(noise_rate == None):
		noise_rate = 0

	acc_step = (1 - noise_rate)* 100	
	
	if(method == "linear"):
		wts = (probabilities - mnm) / (mxm - mnm)
	elif(method == "quadratic"):
		wts = ((probabilities - mnm) / (mxm - mnm))**2  #(2 + int((accuracy*100)/acc_step) )
	elif(method == "exponential"):
		wts = (np.exp(((probabilities - mnm) / (mxm - mnm))) - 1)/ (np.exp(1) - 1)
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
if(dataset_name == "Clothes1M"):
	train_dataset.image_keys = train_dataset.clean_train_keys
	train_dataset.labels = train_dataset.clean_train_labels

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers,drop_last=False, pin_memory=True, sampler = ImbalancedDatasetSampler(train_dataset, callback_get_label = _get_labels))
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,shuffle=False, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,shuffle=False, pin_memory=True)
infer_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers,drop_last=False, pin_memory=True)
########################################Data and LOader#####################################


############################Models and Optimizer#########################################
if(dataset_name == "Clothes1M"):
	from big_model import model
	# from model import model
	models = [model(image_size = size, input_channel = input_channel, n_outputs = num_classes) for i in range(num_models)]
else:
	from model import model
	models = [model(image_size = size, input_channel = input_channel, n_outputs = num_classes) for i in range(num_models)]

opt = [torch.optim.Adam(models[i].parameters(), lr = lr, weight_decay = weight_decay) for i in range(num_models)] ############# Weight decay --> hyperparam #################

models = [nn.DataParallel(mod, device_ids = devices) for mod in models]

for i in range(num_models):
	models[i].cuda()

#Load model from file
if(load == 1):
	# ep denotes epoch at which model was saved
	for i in range(num_models):
		checkpoint = torch.load("checkpoint_" + "epoch_" + str(ep) + "_" + str(i) + "_" + str(ID) + ".pt", map_location = 'cpu')
		models[i].load_state_dict(checkpoint['model_state_dict'])
		opt[i].load_state_dict(checkpoint['opt_state_dict'])

		# # Switch to all trainable
		# for p in models[i].parameters():
		# 	p.requires_grad = True

#################################################################################################

##############Initializn#################
dataset_sz = len(train_dataset) #########CHANGED TO select from dataset
example_weights = np.array([0.5]*dataset_sz)
all_loss, all_acc, all_test_acc, all_val_acc = [], [], [], []
val_loss, test_loss = [], [] 
all_clean_acc = []
##########################################

pair_selector = utils.HardNegativePairSelector()
contrast_loss_obj = losses.OnlineContrastiveLoss(alpha, pair_selector)

#################################MAIN LOOP###################################
for epoch in range(n_epoch):
	#####################INIT####################################
	print("epoch : " + str(epoch))
	Loss = np.zeros((num_models,))
	Acc_noisy, Acc_test = np.zeros((num_models,)), np.zeros((num_models,))
	all_mean = np.zeros((dataset_sz,))
	visited = np.zeros((dataset_sz,))
	mean_selective = []
	steps = 0
	
	################## Switch over to the bigger and noisy dataset after some epochs#############
	if(dataset_name == "Clothes1M" and epoch == clean_epochs):
		train_dataset.image_keys, train_dataset.labels  = train_dataset.noisy_train_keys, train_dataset.noisy_train_labels
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=num_workers,drop_last=False, pin_memory=True, sampler = ImbalancedDatasetSampler(train_dataset, callback_get_label = _get_labels))
		dataset_sz = len(train_dataset)
		example_weights = np.array([0.5]*dataset_sz)
		all_mean = np.zeros((dataset_sz,))

	# Perform inference on whole noisy set before training
	if(epoch >= clean_epochs):
		for i in range(num_models):
			models[i].train = False
		with torch.no_grad():
			# Different data loader as sampler doesn't cover whole data
			for (images, labels, ind) in tqdm(infer_loader):
				labels, images = Variable(labels).cuda(), Variable(images).cuda()
				prob_data = np.zeros((num_models, len(labels), num_classes))
				for i in range(num_models):
					pred, feat, _ = models[i](images)
					wts = torch.Tensor(example_weights[ind]).cuda()
					prob_data[i] = (F.softmax(pred, dim=1)).cpu().data.numpy()
				mean = np.mean(prob_data, axis = 0)
				selective_mean = mean[np.arange(len(labels)), (labels.cpu().data.numpy())]
				all_mean[ind] = selective_mean
			# print(np.sum(all_mean == 0))
		for i in range(num_models):
			models[i].train = True

	visited = np.zeros((dataset_sz,))
	print(len(train_dataset))
	##############################################################################################

	##################################Training Loop##############################################
	for idx, (images, labels, ind) in enumerate(tqdm(train_loader)):
		for i in range(num_models):
			models[i].train = True

		# Shorten an epoch for noisy set
		if(idx >= int(percent * len(train_dataset)/batch_size) and epoch >= clean_epochs):
			break

		visited[ind] = 1
		batch_strt = datetime.datetime.now()
		labels = Variable(labels).cuda()
		images = Variable(images).cuda()

		# Whole probability data of batch
		prob_data = np.zeros((num_models, len(labels), num_classes))

		strt_time = datetime.datetime.now()

		for i in range(num_models):
			pred, features, _ = models[i](images)
			loss = F.cross_entropy(pred, labels, reduce=False)

			#Weigh Examples and calculate weighted average loss
			batch_wts = example_weights[ind]
			wts = torch.Tensor(batch_wts).cuda()
			loss = loss * wts
			loss = torch.mean(loss)
			# contrast_loss = contrast_loss_obj(features, labels)
			# loss = loss + lamda * contrast_loss

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

		mean = np.mean(prob_data, axis = 0)
		selective_mean = mean[np.arange(len(labels)), (labels.cpu().data.numpy())]
		mean_selective += [np.average(selective_mean)]
		all_mean[ind] = selective_mean
		
		end_time = datetime.datetime.now()

		steps += 1
		np.set_printoptions(precision = 3)
		#After batch
		# torch.cuda.empty_cache()
		batch_end = datetime.datetime.now()
		# print("Batch total: ",batch_end - batch_strt)
	#After train
	# print(np.sum(all_mean == 0))
	torch.cuda.empty_cache()
	############################################################################################

	for i in range(num_models):
		models[i].train = False

	###############################VALIDATION##########################
	if(dataset_name == "Clothes1M"):
		val_steps = 0
		Val_Loss = np.zeros((num_models,))
		Acc_val = np.zeros((num_models,))
		for idx, (images, labels, ind) in enumerate(tqdm(val_loader)):
			with torch.no_grad():
				labels = Variable(labels).cuda()
				images = Variable(images).cuda()
				for i in range(num_models):
					pred, features, features_vis = models[i](images)
					loss = F.cross_entropy(pred, labels, reduce=False)
					loss = torch.mean(loss)
					pred_data = pred.cpu().data.numpy()
					pred_label = np.argmax(pred_data, axis=1)
					Acc_val[i] += np.average(pred_label == labels.cpu().data.numpy())
					Val_Loss[i] += loss.cpu().data

					# Visualize CAM for dataset
					# if(i == 0 and idx == 0):
					# 	linear_layer_wts = models[i].module.backbone.fc.weight
					# 	appropriate_wts = linear_layer_wts[labels, :]
					# 	appropriate_wts = (appropriate_wts.unsqueeze(-1)).unsqueeze(-1)
					# 	wtd_feature_map_given_label = features_vis * appropriate_wts
					# 	visualize_CAM(images, wtd_feature_map_given_label, labels, epoch, dataset_name)


					# Feature visualization
					# if(idx == 0 and i == 0):
						# tsne(features.cpu().data.numpy(), labels.cpu().data.numpy(), num_classes)

				val_steps += 1
		Acc_val = Acc_val/val_steps
		print("Val accuracy:", Acc_val)
		all_val_acc += [np.average(np.array(Acc_val))]
		with open("val_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
			pickle.dump(all_val_acc, f)
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
				pred, _, _ = models[i](images)
				loss = F.cross_entropy(pred, labels, reduce=False)
				loss = torch.mean(loss)
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
	# Only calc from those which are visited
	# mxm_mean, mnm_mean, mean_mean = np.amax(all_mean[visited == 1]), np.amin(all_mean[visited == 1]), np.mean(all_mean[visited == 1])

	# ALl are visited
	mxm_mean, mnm_mean, mean_mean = np.amax(all_mean), np.amin(all_mean), np.mean(all_mean)
	print("Max, Min, and Avg Probability mean values: ", mxm_mean, mnm_mean, mean_mean)

	# Across models avg
	acc = np.average(Acc_noisy)
	
	if(epoch >= strt_ep):
		# Wt all examples
		example_weights = set_wts(weight_method, all_mean, mxm_mean, mnm_mean, epoch - strt_ep, acc, noise_rate)
		
		# example_weights[visited == 1] = set_wts(weight_method, all_mean[visited == 1], mxm_mean, mnm_mean, epoch - strt_ep, acc, noise_rate)
		# num_visited_examples = np.sum(visited == 1)
		# print(num_examples)
		## Batchwise Normalization of weights --DDP Review comment
		# example_weights[visited == 1] = (example_weights[visited == 1] / np.sum(example_weights[visited == 1])) * (num_visited_examples / 2)
		# example_weights = (example_weights / np.sum(example_weights)) * (dataset_sz / 2)
	####################################################################################################

	############################# Printing and Saving section start###################################

	with open("wts_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(example_weights, f)

	with open("all_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_mean, f)

	print(Loss)
	print(Test_Loss)
	print(Acc_noisy)
	print(Acc_test)
	all_loss += [np.average(np.array(Loss))]
	test_loss += [np.average(np.array(Test_Loss))]
	all_acc += [np.average(np.array(Acc_noisy))]
	all_test_acc += [np.average(np.array(Acc_test))]

	with open("loss_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_loss, f)
	with open("test_loss_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(test_loss, f)
	with open("acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_acc, f)
	with open("test_acc_mean_" + str(ID) + "_" + str(epoch) + ".pkl", "wb") as f:
		pickle.dump(all_test_acc, f)
	######################################### Printing and Saving section ends########################################
