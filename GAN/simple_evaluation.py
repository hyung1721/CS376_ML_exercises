import os
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import to_pil_image

from GAN import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = {'noise_count': 100, 'img_size': (1, 28, 28)}

if __name__ == '__main__':
    # Set path to model
    model_path = '../ml_model/GAN'
    weights_gen_path = os.path.join(model_path, 'weights_gen.pt')

    # Load generator model
    weights = torch.load(weights_gen_path)
    model_gen = Generator(params).to(device)
    model_gen.load_state_dict(weights)

    # Evaulation mode
    model_gen.eval()

    with torch.no_grad():
        fixed_noise = torch.randn(16, 100, device=device)
        fake_image  = model_gen(fixed_noise).detach().cpu()
    
    print(fake_image.shape)

    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(to_pil_image(0.5 * fake_image[i] + 0.5), cmap='gray')
        plt.axis('off')
    
    plt.show()