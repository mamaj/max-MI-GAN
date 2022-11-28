from pathlib import Path
import socket
from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import torch.distributions as td

DATAPATH = Path.home() / '.datasets'

def display(x, ax=None):
    """
    x (Tensor): B, C, W, H
    """
    x = x.detach().cpu()
    x = make_grid(x)[0]
    print(x.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=(x.shape[1]/30, x.shape[0]/30))
    ax.imshow(x, cmap='gray')
    ax.axis('off')
    

def load_mnist(binary=False, size=28):
    transforms_list = [
        transforms.Resize(size),
        transforms.ToTensor(),
        (lambda x: (x > 0.5).to(x.dtype)) 
        ]
    if binary:
        transforms_list.append(
            lambda x: (x > 0.5).to(x.dtype)
            )
    else:
        transforms_list.extend([
            transforms.Normalize(0.5, 0.5),  # [-1, 1]
            lambda x: x + torch.randn(x.shape) * 1e-2,
            lambda x: x.clip(-1, 1)
            ])
    
    
    train_ds = datasets.MNIST(
        root=DATAPATH,
        train=True,
        download=True,
        transform=transforms.Compose(transforms_list)
        )
    
    test_ds = datasets.MNIST(
        root=DATAPATH,
        train=False,
        download=True,
        transform=transforms.Compose(transforms_list)
        )
    
    return train_ds, test_ds
        


def plot_latent_images(model, n, digit_size=28, ax=None):
    """Plots n x n digit images decoded from the latent space."""
    
    norm = td.Normal(0, 1)
    grid_x = norm.icdf(torch.linspace(0.05, 0.95, n))
    grid_y = norm.icdf(torch.linspace(0.05, 0.95, n))
    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = torch.tensor([[xi, yi]])
            x = model(z)
            digit = torch.reshape(x[0], (digit_size, digit_size))
            image[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit.detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(image, cmap='Greys_r')
    ax.axis('Off')
    return im



class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((-1, *self.shape))


def create_digit_grid(dataset, cols=6):
    labels = dataset.targets
    result = []
    for digit in range(10):
        idx = np.where(labels == digit)[0]
        selected_idx = np.random.choice(idx, cols)
        result += [dataset[i][0] for i in selected_idx]
    return torch.cat(result, dim=0).unsqueeze(1)


def get_logdir(name='runs_maximgan'):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    return os.path.join(
        name, current_time + '_' + socket.gethostname())
