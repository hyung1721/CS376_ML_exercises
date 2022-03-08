# Reference | https://www.youtube.com/watch?v=Gl2WXLIMvKA&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=5
# RNN example

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # We can change RNN to GRU or LSTM
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.gru = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc  = nn.Linear(hidden_size * SEQUENCE_LENGTH, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # If we use LSTM, we need cell state
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        # If we use LSTM, we need to add argument c0
        # out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
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
simplenet = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(simplenet.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

# Training
simplenet.train()
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data   = data.to(device).squeeze(1)
        target = target.to(device)

        # Forward
        pred = simplenet(data)
        loss = loss_func(pred, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_dataloader)} Loss: {loss:.4f}")
