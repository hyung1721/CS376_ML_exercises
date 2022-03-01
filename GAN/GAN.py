# Reference | https://deep-learning-study.tistory.com/639
# Implement GAN using PyTorch

import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator create images from noise
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.nz = params['noise_count']
        self.img_size = params['img_size']
        self.model = nn.Sequential(
            *self._fc_layer(self.nz, 128, normalize=False),
            *self._fc_layer(128, 256),
            *self._fc_layer(256, 512),
            *self._fc_layer(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_size))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_size)
        return img
    
    def _fc_layer(self, in_channels, out_channels, normalize=True):
        layers = []
        layers.append(nn.Linear(in_channels, out_channels))
        
        if normalize:
            layers.append(nn.BatchNorm1d(out_channels, 0.8))
        
        layers.append(nn.LeakyReLU(0.2))
        return layers

# Discriminator classifies images
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.img_size = params['img_size']
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_size)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

# Initialize weights
def initialize_weights(model):
    classname = model.__class__.__name__

    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.dat, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

if __name__ == '__main__':
    # Set data path
    data_path = '../ml_data'
    os.makedirs(data_path, exist_ok=True)

    # Load MNIST dataset
    train_dataset = datasets.MNIST(data_path,
                                train=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize([0.5], [0.5])]),
                                download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Binary Cross Entrophy for loss function
    loss_func = nn.BCELoss()

    learning_rate = 2e-4
    beta1 = 0.5
    params = {'noise_count': 100, 'img_size': (1, 28, 28)}

    # Generator, Discriminator model
    model_gen = Generator(params).to(device)
    model_dis = Discriminator(params).to(device)

    # Optimizer for each model
    opt_generator     = optim.Adam(model_gen.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    opt_discriminator = optim.Adam(model_dis.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    real_label = 1
    fake_label = 0

    noise_count = params['noise_count']

    num_epochs  = 100
    iteration = 0

    loss_history = {'gen': [], 'dis': []}

    # Training
    start_time = time.time()
    model_gen.train()
    model_dis.train()

    for epoch in range(num_epochs):
        for xb, _ in train_dataloader:
            batch_size = xb.size(0)

            xb = xb.to(device)
            
            real = torch.Tensor(batch_size, 1).fill_(1.0).to(device)
            fake = torch.Tensor(batch_size, 1).fill_(0.0).to(device)

            # Generator part
            model_gen.zero_grad()
            noise = torch.randn(batch_size, noise_count, device=device)
            output_gen = model_gen(noise) # generate fake image
            output_dis = model_dis(output_gen) # discriminate fake image

            loss_gen = loss_func(output_dis, real)
            loss_gen.backward()
            opt_generator.step()

            # Discriminator part
            model_dis.zero_grad()
            output_real = model_dis(xb) # discriminate real images
            output_fake = model_dis(output_gen.detach()) # discriminate fake images

            loss_real = loss_func(output_real, real)
            loss_fake = loss_func(output_fake, fake)
            loss_dis = (loss_real + loss_fake) / 2

            loss_dis.backward()
            opt_discriminator.step()

            loss_history['gen'].append(loss_gen.item())
            loss_history['dis'].append(loss_dis.item())
            
            iteration += 1
            if iteration % 1000 == 0:
                print('Epoch: %.0f, Gen_Loss: %.6f, Dis_Loss: %.6f, time: %.2f min'
                    % (epoch, loss_gen.item(), loss_dis.item(), (time.time() - start_time) / 60))

    # Loss history
    plt.figure(figsize=(10, 5))
    plt.title('Loss Progress')
    plt.plot(loss_history['gen'], label = 'Gen. Loss')
    plt.plot(loss_history['dis'], label = 'Dis. Loss')
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save weights
    model_path = '../ml_model/GAN'
    os.makedirs(model_path, exist_ok=True)

    weights_gen_path = os.path.join(model_path, 'weights_gen.pt')
    weights_dis_path = os.path.join(model_path, 'weights_dis.pt')

    torch.save(model_gen.state_dict(), weights_gen_path)
    torch.save(model_dis.state_dict(), weights_dis_path)