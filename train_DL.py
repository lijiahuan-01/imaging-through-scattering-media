
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
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--decay_epoch", type=int, default=0, help="epoch from which to start lr decay")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="speckle2MINST", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=40, help="interval betwen image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
opt = parser.parse_args(args=[])
print(opt)

cuda = True if torch.cuda.is_available() else False
input_shape = (opt.channels, opt.img_size, opt.img_size)

criterion_GAN = torch.nn.MSELoss()
# Initialize generator and discriminator
G_DL = Generator(input_shape)                                            


if cuda:
    G_DL.cuda()
    criterion_GAN.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_DL.load_state_dict(torch.load("E:/DL/G_DL.pth"))
else:
    # Initialize weights
    G_DL.apply(weights_init_normal)
    
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# Optimizers
optimizer_G_DL = torch.optim.Adam(
    G_DL.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)


# Learning rate update schedulers
lr_scheduler_G_DL = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G_DL, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

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
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        self.transform = transforms.Compose(transform)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index,:,:,:]
        X = np.squeeze(X)
        Y0 = self.y[index,:,:,:]
        Y0 = np.squeeze(Y0)
        
        
        if self.transform :
            X = self.transform(X)
            Y = self.transform(Y0)

            return X, Y
        
x_train_DL = np.load('E:/train_image/ground_truth_image_zxg.npy')
print("train_image:", x_train_DL.shape) 
y_ground_DL = np.load('E:/train_image/ground_truth_image_64.npy')
print("ground_truth_image:", y_ground_DL.shape) 
train_set_DL = ImgDataset(x_train_DL, y_ground_DL, transform)
train_loader_DL = DataLoader(train_set_DL, batch_size=opt.batch_size, shuffle=True, drop_last=True)

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    epoch_start_time = time.time()
    for i, data in enumerate(train_loader_DL):
        Batch_start_time = time.time()
        # Set model input
        imgs_A = Variable(data[0].type(FloatTensor))
        imgs_B = Variable(data[1].type(FloatTensor))
                
        G_DL.train()
        optimizer_G_DL.zero_grad()
        #loss_GAN
        fake_B = G_DL(imgs_A)
        loss_GAN_B = criterion_GAN(fake_B, imgs_B)
        
        loss_G = loss_GAN_B
               
        loss_G.backward()
        optimizer_G_DL.step()
            
        batches_done = epoch * len(train_loader_DL) + i
        batches_left = opt.n_epochs * len(train_loader_DL) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log    identity: %f
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d  %2.2f sec(s)][G loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_loader_DL),
                time.time()-Batch_start_time,
                loss_G.item(),
                time_left,
                )
           )
        
        if batches_done % opt.sample_interval == 0:
                real_A = make_grid(imgs_A, nrow=20, normalize=True)
                real_B = make_grid(imgs_B, nrow=20, normalize=True)
                fake_B = make_grid(fake_B, nrow=20, normalize=True)
                # Arange images along y-axis
                image_grid_B = torch.cat((real_A, real_B, fake_B), 1)
                save_image(image_grid_B, "E:/images/train_DL/G1/%d.png" % batches_done, normalize=False)
            

    # Update learning rates               
    lr_scheduler_G_DL.step()
      
    epoch_time = time.time()-epoch_start_time
    print("\r[Epoch %d/%d] %.2f sec(s)"%(epoch, opt.n_epochs, epoch_time,))
    
G_DL.eval()
torch.save(G_DL.state_dict(), "E:/DL/G_DL.pth")
