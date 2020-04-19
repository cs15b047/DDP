import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = self.model.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = np.array(img).copy()
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(preprocessed_img)
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.transpose(np.float32(img), (1, 2, 0))
    cam = cam / np.max(cam)
    return cam


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda, size):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.size = size
        self.ep = 0
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (self.size, self.size))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output, _ = self.forward(input.cuda())
        else:
            output, _ = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

def setup_GCAM(mod, layers, size):
    grad_cam = GradCam(model=mod, target_layer_names=layers, use_cuda=True, size = size)
    gb_model = GuidedBackpropReLUModel(model=mod, use_cuda=True)
    return grad_cam, gb_model


def GCAM(mod, imgs, labels, clean_labels, grad_cam, gb_model):
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    for ind in range(5):
        idx = np.random.randint(0, len(labels))
        img, label, clean = imgs[idx, :, :, :], labels[idx].cpu().data.numpy(), clean_labels[idx]
        size = img.size(-1)
        input = preprocess_image(img)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        # print(clean, label)
        if clean != label:
            target_index = clean
            mask = grad_cam(input, target_index)

            cam_img = show_cam_on_image(img, mask)
            cam_img = np.uint8(cam_img * 255)
            cv2.imwrite("GB/cam_clean_" + str(ind) + "_epoch_" + str(grad_cam.ep) + ".jpg", cam_img)

            gb = gb_model(input, index=target_index)
            gb = gb.transpose((1, 2, 0))
            cam_mask = cv2.merge([mask, mask, mask])
            cam_gb = deprocess_image(cam_mask*gb)
            gb = deprocess_image(gb)
            cv2.imwrite("GB/gb_clean" + "_" + str(ind) + "_epoch_" + str(grad_cam.ep) + ".jpg", gb)
            cv2.imwrite("GB/cam_gb_clean" + "_" + str(ind) + "_epoch_" + str(grad_cam.ep) + ".jpg", cam_gb)

        target_index = label
        mask = grad_cam(input, target_index)

        cam_img = show_cam_on_image(img, mask)
        cam_img = np.uint8(cam_img * 255)
        cv2.imwrite("GB/cam_" + str(ind) + "_epoch_" + str(grad_cam.ep) + ".jpg", cam_img)

        gb = gb_model(input, index=target_index)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask*gb)
        gb = deprocess_image(gb)

        cv2.imwrite("GB/gb" + "_" + str(ind) + "_epoch_" + str(grad_cam.ep) + ".jpg", gb)
        cv2.imwrite("GB/cam_gb" + "_" + str(ind) + "_epoch_" + str(grad_cam.ep) + ".jpg", cam_gb)

    grad_cam.ep += 1

if __name__ == '__main__':
    mod = Model(32, 3, 10)
    layers = ["0"]
    size = 32
    grad_cam, gb_model = setup_GCAM(mod, layers, size)
    noise_type = "symmetric"
    noise_rate = 0.2
    train_dataset = CIFAR10(root='./data/',download=True,  train=True, transform=get_transform(),noise_type=noise_type,noise_rate=noise_rate)
    img, label, idx = train_dataset[0]
    GCAM(mod, img, label, grad_cam, gb_model)
