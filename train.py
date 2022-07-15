import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence as kl
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange

from model import MaxIMGAN
from utils import display, load_mnist, plot_latent_images

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparams
latent_dim = 128
batch_size = 64
epochs = 20
encoder_iters = 1
lr = 0.0002
adam_betas=(0.5, 0.999)
beta = 0.0005


x_size = 20

mnist, _ = load_mnist()
mnist_lowres, _ = load_mnist(size=x_size)

y_loader = DataLoader(mnist, shuffle=True, batch_size=batch_size, drop_last=True)
x_loader = DataLoader(mnist_lowres, shuffle=True, batch_size=batch_size, drop_last=True)


model = MaxIMGAN(input_shape=(1, x_size, x_size), output_shape=(1, 28, 28))
model.to(DEVICE)


enc_optim = torch.optim.Adam(lr=lr, params=model.generator.parameters(), betas=adam_betas)
disc_optim = torch.optim.Adam(lr=lr, params=model.discriminator.parameters(), betas=adam_betas)


disc_loss_list, enc_loss_list = [], []

for epoch in trange(epochs):
    
    for (x, _), (y_real, _) in tqdm(zip(x_loader, y_loader), total=len(x_loader)):
        # train the discriminator
        y_real = y_real.to(DEVICE)
        x = x.to(DEVICE)
        
        y_fake = model(x).sample().clip(-1, 1)
        
        y = torch.cat((y_real, y_fake))
        labels = torch.cat((
            torch.ones((batch_size, 1)),
            torch.zeros((batch_size, 1))
            )).to(DEVICE)
        
        disc_logit = model.discriminator(y)
        disc_loss = F.binary_cross_entropy_with_logits(disc_logit, labels)
    
        disc_optim.zero_grad()
        disc_loss.backward()
        disc_optim.step()
        disc_loss_list.append(disc_loss.item())
        
        
        
        # train the encoder
        y_fake_dist = model(x)
        y_fake = y_fake_dist.rsample().clip(-1, 1)
        
        disc_logit = model.discriminator(y_fake)
        label = torch.ones((batch_size, 1)).to(DEVICE)
        encoder_loss = (
            F.binary_cross_entropy_with_logits(disc_logit, label) 
            - beta * y_fake_dist.log_prob(y_fake).mean()
            )
        
        enc_optim.zero_grad()
        encoder_loss.backward()
        enc_optim.step()
        enc_loss_list.append(encoder_loss.item())
        
    if epoch % 25 == 0 and epoch != 0:
        torch.save(model, f'checkpoints/model_{epoch}.pt')



# display samples
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
display(model(x).sample(), axs[0])
display(x, axs[1])

# display loss
fig, ax = plt.subplots()
ax.plot(disc_loss_list, 'r', label='disc')
ax2 = ax.twiny()
ax2.plot(enc_loss_list, 'b', label='gen')
fig.legend(loc='right')