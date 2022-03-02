# Reference | https://www.youtube.com/watch?v=uQc4Fs7yx5I&t=39s
# Implementation of GoogLeNet (InceptionNet)

from time import sleep
import torch
import torch.nn as nn

class GoogLeNet(nn.Module):
    def __init__(
        self,
        aux_logits = True,
        num_classes = 1000
    ):
        
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = conv_block(
            in_channels = 3,
            out_channels = 64,
            kernel_size = (7,7),
            stride = (2,2),
            padding = (3,3)
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        )
        self.conv2 = conv_block(
            in_channels = 64,
            out_channels = 192,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        )

        self.inception3a = Inception_module(
            in_channels = 192,
            filters_1x1 = 64,
            filters_3x3_reduce = 96,
            filters_3x3 = 128,
            filters_5x5_reduce = 16,
            filters_5x5 = 32,
            filters_pool = 32
        )
        self.inception3b = Inception_module(
            in_channels = 256,
            filters_1x1 = 128,
            filters_3x3_reduce = 128,
            filters_3x3 = 192,
            filters_5x5_reduce = 32,
            filters_5x5 = 96,
            filters_pool = 64
        )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        )

        self.inception4a = Inception_module(
            in_channels = 480,
            filters_1x1 = 192,
            filters_3x3_reduce = 96,
            filters_3x3 = 208,
            filters_5x5_reduce = 16,
            filters_5x5 = 48,
            filters_pool = 64
        )
        self.inception4b = Inception_module(
            in_channels = 512,
            filters_1x1 = 160,
            filters_3x3_reduce = 112,
            filters_3x3 = 224,
            filters_5x5_reduce = 64,
            filters_5x5 = 64,
            filters_pool = 64
        )
        self.inception4c = Inception_module(
            in_channels = 512,
            filters_1x1 = 128,
            filters_3x3_reduce = 128,
            filters_3x3 = 256,
            filters_5x5_reduce = 24,
            filters_5x5 = 64,
            filters_pool = 64
        )
        self.inception4d = Inception_module(
            in_channels = 512,
            filters_1x1 = 112,
            filters_3x3_reduce = 144,
            filters_3x3 = 288,
            filters_5x5_reduce = 32,
            filters_5x5 = 64,
            filters_pool = 64
        )
        self.inception4e = Inception_module(
            in_channels = 528,
            filters_1x1 = 256,
            filters_3x3_reduce = 160,
            filters_3x3 = 320,
            filters_5x5_reduce = 32,
            filters_5x5 = 128,
            filters_pool = 128
        )
        self.maxpool4 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1)

        self.inception5a = Inception_module(
            in_channels = 832,
            filters_1x1 = 256,
            filters_3x3_reduce = 160,
            filters_3x3 = 320,
            filters_5x5_reduce = 32,
            filters_5x5 = 128,
            filters_pool = 128
        )
        self.inception5b = Inception_module(
            in_channels = 832,
            filters_1x1 = 384,
            filters_3x3_reduce = 192,
            filters_3x3 = 384,
            filters_5x5_reduce = 48,
            filters_5x5 = 128,
            filters_pool = 128
        )

        self.avgpool = nn.AvgPool2d(
            kernel_size = 7,
            stride = 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(
            in_features = 1024,
            out_features = 1000
        )

        if self.aux_logits:
            self.aux1 = aux_block(
                in_channels = 512,
                num_classes = num_classes
            )
            self.aux2 = aux_block(
                in_channels = 528,
                num_classes = num_classes
            )
        else:
            self.aux1 = self.aux1 = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x

class Inception_module(nn.Module):
    def __init__(
        self,
        in_channels,
        filters_1x1,
        filters_3x3_reduce, filters_3x3,
        filters_5x5_reduce, filters_5x5,
        filters_pool
    ):
        
        super(Inception_module, self).__init__()

        # First branch has 1x1 convolutions
        self.branch1 = conv_block(
            in_channels = in_channels,
            out_channels = filters_1x1,
            kernel_size = 1
        )

        # Second branch has 1x1 convolutions and 3x3 convolutions
        self.branch2 = nn.Sequential(
            conv_block(
                in_channels = in_channels,
                out_channels = filters_3x3_reduce,
                kernel_size = 1
            ),
            conv_block(
                in_channels = filters_3x3_reduce,
                out_channels = filters_3x3,
                kernel_size = 3,
                stride = 1,
                padding = 1
            )
        )

        # Third branch has 1x1 convolutions and 5x5 convolutions        
        self.branch3 = nn.Sequential(
            conv_block(
                in_channels = in_channels,
                out_channels = filters_5x5_reduce,
                kernel_size = 1
            ),
            conv_block(
                in_channels = filters_5x5_reduce,
                out_channels = filters_5x5,
                kernel_size = 5,
                padding = 2
            )
        )

        # Fourth branch has 3x3 max pooling and 1x1 convolutions
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            conv_block(
                in_channels = in_channels,
                out_channels = filters_pool,
                kernel_size = 1
            )
        )
    
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class aux_block(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(aux_block, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = GoogLeNet(aux_logits=True, num_classes=1000)
    print(model(x)[2].shape)
    for param in model.parameters():
        print(type(param), param.size())
