# The Base code for the neural network and its training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Print accuracy
def accuracy(dataloader, model, squeeze=False):
    if dataloader.dataset.train:
        print("accuracy on training data")
    else:
        print("accuary on test data")
    
    correct = 0
    samples = 0

    # evaluation mode
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            if squeeze:
                x = x.to(device).squeeze(1)
            else:
                x = x.to(device)
                x = x.reshape(x.shape[0], -1)
            y = y.to(device)

            

            scores = model(x)
            _, prediction = scores.max(1)
            correct += (prediction == y).sum()
            samples += prediction.size(0)
        
        print(f"{correct} correct from {samples} samples - accuracy {float(correct)/float(samples)*100:.2f}%")
    
    model.train()

# CUDA configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
NUM_EPOCHS = 5

### Model ###
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    
    def forward(self, x):
        return self.model(x)

#############

if __name__=="__main__":
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
    simplenet = SimpleNet().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(simplenet.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    # Training
    simplenet.train()
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data   = data.to(device)
            target = target.to(device)

            data = data.reshape(data.shape[0], -1)

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

    accuracy(test_dataloader, simplenet)