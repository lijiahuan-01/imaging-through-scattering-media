
import argparse
import os
import numpy as np
import math
import random
import itertools
import datetime
import time
import matplotlib.pyplot as plt
from PIL import Image
import sys

from torchvision.transforms import transforms, InterpolationMode
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable   
import torch.autograd as autograd

from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import LambdaLR

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=195, help="epoch to start training from")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="speckle2MINST", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=40, help="interval betwen image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args(args=[])
print(opt)

cuda = True if torch.cuda.is_available() else False
input_shape = (opt.channels, opt.img_size, opt.img_size)

G_AB = Generator(input_shape)
G_BA = Generator(input_shape)
G_DL = Generator(input_shape) 

if cuda:
    G_AB.cuda()
    G_BA.cuda()
    G_DL.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("E:/Generator/G_AB_%d.pth" % opt.epoch))
    G_BA.load_state_dict(torch.load("E:/Generator/G_BA_%d.pth" % opt.epoch))
    G_DL.load_state_dict(torch.load("E:/DL/G_DL.pth"))
    #(torch.load("E:/cv_simulation_data/GAN/saved_models/zixiangguan-GAN/Z/G_DL.pth"))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    G_DL.apply(weights_init_normal)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Image transformations
transform = [
    transforms.ToPILImage(),
    transforms.Resize((opt.img_size, opt.img_size), InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
]

class ImgDataset(Dataset):
    def __init__(self, x, y=None, z=None, transform=None):
        self.x = x
        self.y = y
        self.z = z
        if z is not None:
            self.z = z
        self.transform = transforms.Compose(transform)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index,:,:,:]
        X = np.squeeze(X)
        Y0 = self.y[index,:,:,:]
        Y0 = np.squeeze(Y0)
        
        X = self.transform(X)
        Y = self.transform(Y0)
        
        if self.z is not None:
            Z = self.z[index,:,:,:]
            Z = np.squeeze(Z)
            Z = self.transform(Z)
            return X, Y, Z
        
        else: 
            return X, Y
        
G_AB.eval()
G_BA.eval()

x_train = np.load('E:/test_image/test_image_zxg_3.npy')
print("train_image:", x_train.shape) 
y_ground = np.load('E:/test_image/ground_truth_image_zxg_3.npy')
print("ground_truth_image:", y_ground.shape) 
test_set = ImgDataset(x = x_train, y=y_ground, transform=transform)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, drop_last=True)

pred_img = np.ones((1000,opt.img_size, opt.img_size))

with torch.no_grad():
    for i, data in enumerate(test_loader):
        real_A = Variable(data[0].type(FloatTensor))
        test_pred = G_AB(real_A)
        test_pred2 = test_pred.cpu().data.numpy().squeeze()
        pred_img[opt.batch_size*i:opt.batch_size*i+opt.batch_size,:] = test_pred2

y_pred = pred_img.astype(np.float32)
y_pred2 = np.ones((1000,opt.img_size, opt.img_size)).astype(np.float32)
for i in range(1000):
    img = y_pred[i,:]
    img2 = 255*(img+1)/2
    y_pred2[i,:] = img2
y_pred3 = y_pred2.reshape(1000,opt.img_size,opt.img_size,1).astype(np.uint8)

z_ground = np.load('E:/test_image/ground_truth_image_64.npy')
test_set = ImgDataset(x=x_train, y=y_pred3, z=z_ground, transform = transform)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, drop_last=True)

G_DL.eval()

for i, data in enumerate(test_loader):
    real_A = Variable(data[0].type(FloatTensor))
    real_B = Variable(data[1].type(FloatTensor))
    real_C = Variable(data[2].type(FloatTensor))
    fake_C = G_DL(real_B)
    real_C = make_grid(real_C, nrow=10, normalize=True)
    real_B = make_grid(real_B, nrow=10, normalize=True)
    real_A = make_grid(real_A, nrow=10, normalize=True)
    fake_C = make_grid(fake_C, nrow=10, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, real_B,fake_C, real_C), 1)
    save_image(image_grid, "E:/images/test_result/3/%d.png" % i, normalize=False)
