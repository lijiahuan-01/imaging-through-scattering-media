
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
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
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
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args(args=[])
print(opt)


cuda = True if torch.cuda.is_available() else False
input_shape = (opt.channels, opt.img_size, opt.img_size)

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
# Initialize generator and discriminator
G_AB = Generator(input_shape)
G_BA = Generator(input_shape)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB.cuda()
    G_BA.cuda()
    D_A.cuda()
    D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("E:/Generator/G_AB_%d.pth" % opt.epoch))
    G_BA.load_state_dict(torch.load("E:/Generator/G_BA_%d.pth" % opt.epoch))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class ReplayBuffer:
    def __init__(self, max_size=100):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

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
        
x_train= np.load('E:/train_image/train_image_zxg.npy')
print("train_image:", x_train.shape) 
y_ground = np.load('E:/train_image/ground_truth_image_zxg.npy')
print("ground_truth_image:", y_ground.shape) 
train_set = ImgDataset(x_train, y_ground, transform)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True)

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    epoch_start_time = time.time()
    for i, data in enumerate(train_loader):
        Batch_start_time = time.time()
        # Set model input
        imgs_A = Variable(data[0].type(FloatTensor))
        imgs_B = Variable(data[1].type(FloatTensor))
        
        # Adversarial ground truths
        valid = Variable(FloatTensor(np.ones((imgs_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(FloatTensor(np.zeros((imgs_A.size(0), *D_A.output_shape))), requires_grad=False)
        
        
        G_AB.train()
        G_BA.train()
        optimizer_G.zero_grad()
        #loss_GAN
        fake_B = G_AB(imgs_A)
        loss_GAN_A = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(imgs_B)
        loss_GAN_B = criterion_GAN(D_A(fake_A), valid)
        
        loss_GAN = loss_GAN_A + loss_GAN_B
        
        #loss_cycle
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, imgs_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, imgs_B)
        
        loss_cycle = loss_cycle_A +  loss_cycle_B
        
        
        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle
        
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(imgs_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_B(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()
        
        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(imgs_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D =  loss_D_B + loss_D_A 
            
        batches_done = epoch * len(train_loader) + i
        batches_left = opt.n_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log    identity: %f
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d  %2.2f sec(s)][D loss: %f] [G loss: %f,GAN: %f,cycle: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_loader),
                time.time()-Batch_start_time,
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                time_left,
                )
           )
        
        if batches_done % opt.sample_interval == 0:
                real_B = make_grid(imgs_B, nrow=20, normalize=True)
                fake_B = make_grid(fake_B, nrow=20, normalize=True)
                recov_B = make_grid(recov_B, nrow=20, normalize=True)
                # Arange images along y-axis
                image_grid_B = torch.cat(( real_B, recov_B,fake_B), 1)
                save_image(image_grid_B, "E:/images/train_G/G_B/%d.png" % batches_done, normalize=False)
            
                real_A = make_grid(imgs_A, nrow=20, normalize=True)
                fake_A = make_grid(fake_A, nrow=20, normalize=True)
                recov_A = make_grid(recov_A, nrow=20, normalize=True)
                # Arange images along y-axis
                image_grid_A = torch.cat(( real_A, recov_A,fake_A), 1)
                save_image(image_grid_A, "E:/images/train_G/G_A/%d.png" % batches_done, normalize=False)

    # Update learning rates               
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
       
    epoch_time = time.time()-epoch_start_time
    print("\r[Epoch %d/%d] %.2f sec(s)"%(epoch, opt.n_epochs, epoch_time,))
    
    
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "E:/Generator/G_AB_%d.pth" % epoch)
        torch.save(G_BA.state_dict(), "E:/Generator/G_BA_%d.pth" % epoch)
        
        
G_AB.eval()
G_BA.eval()
