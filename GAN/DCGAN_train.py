import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from DCGAN_model import Discriminator, Generator, initialize_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
IMG_CHANNELS = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURE_DIS, FEATURE_GEN = 64, 64

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMG_CHANNELS)],
            [0.5 for _ in range(IMG_CHANNELS)]
        )
    ]
)

dataset = datasets.MNIST(root="../ml_data", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model_gen = Generator(Z_DIM, IMG_CHANNELS, FEATURE_GEN).to(device)
model_dis = Discriminator(IMG_CHANNELS, FEATURE_DIS).to(device)

initialize_weights(model_gen)
initialize_weights(model_dis)

opt_gen = optim.Adam(model_gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_dis = optim.Adam(model_dis.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
loss_function = nn.BCELoss()

fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)

model_gen.train()
model_dis.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        real = real.to(device)
        fake = model_gen(noise)

        # Train Discriminator
        dis_real = model_dis(real).reshape(-1)
        dis_fake = model_dis(fake).reshape(-1)
        loss_real = loss_function(dis_real, torch.zeros_like(dis_real))
        loss_fake = loss_function(dis_fake, torch.zeros_like(dis_fake))
        loss_dis = (loss_real + loss_fake) / 2

        model_dis.zero_grad()
        loss_dis.backward(retain_graph=True)
        opt_dis.step()

        # Train Generator
        output = model_dis(fake).reshape(-1)
        loss_gen = loss_function(output, torch.zeros_like(output))
        
        model_gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} Discriminator Loss: {loss_dis:.4f}, Generator Loss: {loss_gen:.4f}")

