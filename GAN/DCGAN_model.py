# Reference
# Paper - https://arxiv.org/abs/1511.06434 
# Implementation -  https://www.youtube.com/watch?v=IZtv9s_Wx9I&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=25
# Implementation of Deep Convolutional GAN

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels, features_dis):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            # According to the paper, BatchNorm is not applied to the input layer
            # of Discriminator
            self.conv_block(img_channels    , features_dis    , 4, 2, 1, batchnorm=False),
            self.conv_block(features_dis    , features_dis * 2, 4, 2, 1),
            self.conv_block(features_dis * 2, features_dis * 4, 4, 2, 1),
            self.conv_block(features_dis * 4, features_dis * 8, 4, 2, 1),
            self.conv_block(features_dis * 8, 1               , 4, 2, 0, batchnorm=False, leakyrelu=False),
            nn.Sigmoid()
        )
    
    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=True, leakyrelu=True):
        block = nn.Sequential()
        block.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
        )
        if batchnorm:
            block.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        if leakyrelu:
            # According to the paper, use LeakyReLU activation in discriminator for all layers
            block.add_module("leakyrelu", nn.LeakyReLU(0.2))
        return block

    def forward(self, x):
        return self.dis(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_gen):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.conv_block(z_dim            , features_gen * 16, 4, 1, 0),
            self.conv_block(features_gen * 16, features_gen * 8 , 4, 2, 1),
            self.conv_block(features_gen * 8 , features_gen * 4 , 4, 2, 1),
            self.conv_block(features_gen * 4 , features_gen * 2 , 4, 2, 1),
            self.conv_block(features_gen * 2 , img_channels     , 4, 2, 1, batchnorm=False, relu=False),
            # According to the paper, BatchNorm is not applied to the output layer
            # of Generator
            nn.Tanh()
        )
    
    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=True, relu=True):
        block = nn.Sequential()
        block.add_module(
            "conv",
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
        )
        if batchnorm:
            block.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        if relu:
            # According to the paper, use ReLU activation in generator for all layers
            # except for the output, which uses Tanh.
            block.add_module("relu", nn.ReLU())
        return block
    
    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100

    dis_sample = torch.randn((N, in_channels, H, W))
    dis = Discriminator(in_channels, 8)
    initialize_weights(dis)
    print(dis(dis_sample).shape)
    print(dis)

    gen_sample = torch.randn((N, z_dim, 1, 1))
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    print(gen(gen_sample).shape)
    print(gen)

# test()