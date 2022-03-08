# Reference | https://www.youtube.com/watch?v=jGst43P-TJA&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=6
# RNN example

import sys
import os
# To import other code in ../base/
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from base.pytorch_model_train import accuracy

# CUDA configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 5

INPUT_SIZE = 28
NUM_CLASSES = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 2
SEQUENCE_LENGTH = 28

### Model ###
class Bi_LSTM_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Bi_LSTM_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # multipying 2 -> one for forward, one for backward
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0)) # not use second output (hidden_state, cell_state)
        out = self.fc(out[:, -1, :])
        return out

#############

# Transformation
transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

# Load MNIST dataset for train and test
train_dataset = datasets.MNIST(root='../ml_data', train=True,  transform=transform, download=True)
test_dataset  = datasets.MNIST(root='../ml_data', train=False, transform=transform, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True)

# Model
model = Bi_LSTM_RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

# Training
model.train()
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data   = data.to(device).squeeze(1)
        target = target.to(device)

        # Forward
        pred = model(data)
        loss = loss_func(pred, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_dataloader)} Loss: {loss:.4f}")

accuracy(test_dataloader, model, squeeze=True)