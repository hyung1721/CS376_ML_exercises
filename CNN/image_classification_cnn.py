# Reference | Lecture from School of Computing, KAIST
# 2021 Fall Semester 
# CS376 Machine Learing Lecture
# Assignment #3: Image Classification with CNN

import os

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms

from pathlib import Path

torch.manual_seed(470)
torch.cuda.manual_seed(470)

max_epoch = 100
learning_rate = 0.1
batch_size = 128
device = 'cuda'

output_dim = 10
training_process = True

# Data Pipeline
data_dir = os.path.join('./', 'my_data')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
)

train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truch')

# Neural Networks
class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        self.features = self.create_layers()
        self.classifier = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def create_layers(self):
        layer_infos = [32, 32, 32, 32,
                       'M',
                       64, 64, 64, 64,
                       'M',
                       128, 128, 128,
                       'M',
                       256, 256,
                       'M',
                       256, 256,
                       'M']
        layers = []
        in_channels = 3

        for info in layer_infos:
            if info == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, info, kernel_size=3, padding=1),
                           nn.BatchNorm2d(info),
                           nn.ReLU(inplace=True)]
                in_channels = info
        
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

my_classifier = MyClassifier()
my_classifier = my_classifier.to(device)
# print(my_classifier)

optimizer = optim.Adam(my_classifier.parameters(), lr=learning_rate)

# Load pre-trained weights
ckpt_dir = os.path.join('./', 'checkpoints')

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

best_acc = 0
ckpt_path = os.path.join(ckpt_dir, 'lastest.pt')
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)
    try:
        my_classifier.load_state_dict(ckpt['my_classifier'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_acc = ckpt['best_acc']
    except RuntimeError as e:
        print("Wrong checkpoint")
    else:
        print("checkpoint is loaded")
        print("current best accuracy : %.2f" % best_acc)

name = 'cs376'
ckpt_dir = 'ckpts'
ckpt_reload = '10'
gpu = True
log_dir = 'logs'
log_iter = 100

result_dir = Path('./') / 'result' / name
ckpt_dir = result_dir / ckpt_dir
ckpt_dir.mkdir(parents=True, exist_ok=True)
log_dir = result_dir / log_dir
log_dir.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'

# Train the network
writer = SummaryWriter(log_dir)

if training_process:
    it = 0
    train_losses = []
    test_losses  = []

    for epoch in range(max_epoch):
        my_classifier.train()

        for inputs, labels in train_dataloader:
            it += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = my_classifier(inputs)

            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=1) == labels).float().mean()

            if it % 2000 == 0 and writer is not None:
                writer.add_scalar('Train_loss', loss.item(), it)
                writer.add_scalar('Train_accuracy', acc.item(), it)
                print('[epoch:{}, iteration:{}] train loss : {:.4f} train accuracy : {:.4f}'.format(epoch+1, it, loss.item(), acc.item()))
            
        train_losses.append(loss)

        n = 0.
        test_loss = 0
        test_acc  = 0

        my_classifier.eval()

        for test_inputs, test_labels in test_dataloader:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)

            logits = my_classifier(test_inputs)
            test_loss += F.cross_entropy(logits, test_labels, reduction='sum').item()
            test_acc  += (logits.argmax(dim=1) == test_labels).float().sum().item()
            n += test_inputs.size(0)
        
        test_loss /= n
        test_acc  /= n
        test_losses.append(test_loss)

        writer.add_scalar('Test_loss', test_loss, it)
        writer.add_scalar('Test_accuracy', test_acc, it)

        print('[epoch:{}, iteration:{}] test_loss : {:.4f} test accuracy : {:.4f}'.format(epoch+1, it, test_loss, test_acc)) 

        writer.flush()

        if test_acc > best_acc:
            best_acc = test_acc
            # Note: optimizer also has states ! don't forget to save them as well.
            ckpt = {'my_classifier':my_classifier.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'best_acc':best_acc}
            torch.save(ckpt, ckpt_path)