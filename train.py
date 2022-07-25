import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence as kl
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange
from storch.method import ScoreFunction
import storch

from model import MaxIMGAN
from utils import display, load_mnist, plot_latent_images

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Hyperparams
latent_dim = 128
n_mixtures = 4
batch_size = 64
epochs = 100
lr = 0.0002
adam_betas=(0.5, 0.999)
beta = 0.0001


# load data
x_size = 20

mnist, _ = load_mnist()
mnist_lowres, _ = load_mnist(size=x_size)

y_loader = DataLoader(mnist, shuffle=True, batch_size=batch_size, drop_last=True)
x_loader = DataLoader(mnist_lowres, shuffle=True, batch_size=batch_size, drop_last=True)


# Model
model = MaxIMGAN(input_shape=(1, x_size, x_size),
                 output_shape=(1, 28, 28),
                 n_mixtures=n_mixtures)

model.to(DEVICE)

# Optimizers
gen_optim = torch.optim.Adam(lr=lr, params=model.generator.parameters(), betas=adam_betas)
disc_optim = torch.optim.Adam(lr=lr, params=model.discriminator.parameters(), betas=adam_betas)


disc_loss_list, gen_loss_list = [], []
score_func = ScoreFunction("e", n_samples=200, baseline_factory='moving_average')


for epoch in trange(epochs):
    for (x, _), (y_real, _) in tqdm(zip(x_loader, y_loader), total=len(x_loader)):
        
        
        # train the discriminator
        y_real = y_real.to(DEVICE)
        x = x.to(DEVICE)
        
        y_fake = model(x).sample()
        
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
        
        
        
        
        
        # train the generator
        y_fake_dist = model(x)
        y_fake = score_func(y_fake_dist)
        # y_fake = y_fake_dist.rsample()
        
        
        disc_logit = model.discriminator(y_fake)
        label = torch.ones(disc_logit.shape).to(DEVICE)
        gen_loss = (
            F.binary_cross_entropy_with_logits(disc_logit, label, reduction='none')[:, :, 0] 
            # - beta * y_fake_dist.log_prob(y_fake).mean()
            # + beta * y_fake_dist.entropy().mean()
            - beta * y_fake_dist.log_prob(y_fake)
            ).mean(-1)
        
        gen_optim.zero_grad()
        storch.add_cost(gen_loss, "gen_loss")
        storch.backward()
        gen_optim.step()
        gen_loss_list.append(gen_loss.detach_tensor().mean().item())
        
        
        
        
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
ax2.plot(gen_loss_list, 'b', label='gen')
fig.legend(loc='right')
