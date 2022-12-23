# Modifiing info gan to take images as codes.
# taken from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/infogan/infogan.py

import argparse
import os
import numpy as np
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.tensorboard import SummaryWriter

from utils import create_digit_grid, get_logdir, show, display


# ---------------------------------------------------------------------------- #
#                                    CONFIG                                    #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--dz", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--dy", type=int, default=32, help="size of each image dimension")
parser.add_argument("--dx", type=int, default=8, help="size of each smaller image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
parser.add_argument("--lambda_info", type=float, default=0.1, help="recons loss weight")
opt = parser.parse_args()
print(opt)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE)

# ---------------------------------------------------------------------------- #
#                                    MODELS                                    #
# ---------------------------------------------------------------------------- #
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def discriminator_block(in_filters, out_filters, bn=True):
    """ Returns layers of each discriminator block.
        out_size = ceil(in_size/2)
    """
    block = [
        nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1), 
        nn.LeakyReLU(0.2),
        nn.Dropout2d(0.25)
    ]
    if bn:
        block.append(nn.BatchNorm2d(out_filters, 0.8))
    return block


def generator_block(in_filters, out_filters, upsample=True):
    """ Returns layers of each generator block.
        out_size = in_size * 2 
    """
    block = [
        nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding='same'),
        nn.BatchNorm2d(out_filters, 0.8),
        nn.LeakyReLU(0.2),
    ]
    if upsample:
        block.insert(0, nn.Upsample(scale_factor=2))
        
    return block


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = opt.dz + opt.dx ** 2
        self.preconv_size = opt.dy // 4  # Initial size before 2 upsampling
        
        self.preconv_lin = nn.Linear(input_dim, 128 * self.preconv_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            *generator_block(128, 128),
            *generator_block(128, 64),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding='same'),
            nn.Tanh(),
        )


    def forward(self, z, x):
        gen_in = torch.cat((z, x.flatten(1)), -1)
        conv_in = self.preconv_lin(gen_in)
        conv_in = conv_in.view(conv_in.shape[0], 
                               128, self.preconv_size, self.preconv_size)
        conv_out = self.conv_blocks(conv_in)
        return conv_out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
        )
        self.adv_conv_blocks = nn.Sequential(
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        self.x_conv_blocks = nn.Sequential(
            nn.BatchNorm2d(32),
            *generator_block(32, 32, upsample=False),
            nn.Conv2d(32, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

        # The height and width of downsampled image
        ds_size = opt.dy // (2 ** 4)

        # Output layers
        self.adv_lin = nn.Linear(128 * ds_size ** 2, 1)
        

    def forward(self, img):
        img = self.conv_blocks(img)

        x = self.x_conv_blocks(img)
        adv = self.adv_conv_blocks(img)
        validity = self.adv_lin(adv.flatten(start_dim=1))
        
        return validity, x


# ---------------------------------------------------------------------------- #
#                                Loss functions                                #
# ---------------------------------------------------------------------------- #
adversarial_loss = torch.nn.MSELoss()
small_img_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

generator.to(DEVICE)
discriminator.to(DEVICE)

generator.apply(init_weights)
discriminator.apply(init_weights)

# ---------------------------------------------------------------------------- #
#                                 DATA LOADERS                                 #
# ---------------------------------------------------------------------------- #
DATASET_FOLDER = '../data/mnist'
os.makedirs(DATASET_FOLDER, exist_ok=True)

y_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        DATASET_FOLDER,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(opt.dy),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

x_loader = torch.utils.data.DataLoader(
    x_ds := datasets.MNIST(
        DATASET_FOLDER,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(opt.dx),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# ---------------------------------------------------------------------------- #
#                                  OPTIMIZERS                                  #
# ---------------------------------------------------------------------------- #
optimizer_gen = torch.optim.Adam(
    generator.parameters(),
    lr=opt.lr, betas=(opt.b1, opt.b2))

optimizer_disc = torch.optim.Adam(
    discriminator.parameters(), 
    lr=opt.lr, betas=(opt.b1, opt.b2))

optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), 
    lr=opt.lr, betas=(opt.b1, opt.b2)
)


COLS = 8
grid = create_digit_grid(x_ds, cols=COLS)
static_z = torch.randn((COLS * 10, opt.dz), device=DEVICE)
grid = grid.to(DEVICE)

def write_tboard(writer, it, d_loss, g_loss, info_loss):
    writer.add_scalar('loss/d', d_loss, it)
    writer.add_scalar('loss/g', g_loss, it)
    writer.add_scalar('loss/info', info_loss, it)
    
    generator.eval()
    y = generator(static_z, grid)
    writer.add_image('images/x', make_grid(grid, normalize=True), it)
    writer.add_image('images/y', make_grid(y, normalize=True), it)


# ---------------------------------------------------------------------------- #
#                                   TRAINING                                   #
# ---------------------------------------------------------------------------- #

writer = SummaryWriter(get_logdir(name='mod_infogan'))

it = 0
for epoch in range(opt.n_epochs):
    for i, ((y, _), (x, _)) in enumerate(zip(y_loader, x_loader)):
        it += 1
        
        generator.train()
        
        batch_size = y.shape[0]

        # Adversarial ground truths
        valid = torch.ones((batch_size, 1), requires_grad=False)
        fake = torch.zeros((batch_size, 1), requires_grad=False)

        # Configure input
        y = y.to(DEVICE)
        x = x.to(DEVICE)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_gen.zero_grad()
        
        # Sample noise for as generator input
        z = (torch.randn((batch_size, opt.dz)))

        # Generate a batch of images
        gen_imgs = generator(z, x)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_gen.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_disc.zero_grad()

        # Loss for real images
        real_pred, _ = discriminator(y)
        disc_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _ = discriminator(gen_imgs.detach())
        disc_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (disc_real_loss + disc_fake_loss) / 2

        d_loss.backward()
        optimizer_disc.step()
        
        # ------------------
        # Information Loss
        # ------------------
        z = (torch.randn((batch_size, opt.dz)))
        gen_imgs = generator(z, x)
        _, x_hat = discriminator(gen_imgs)

        info_loss = opt.lambda_info * small_img_loss(x, x_hat)
        info_loss.backward()
        optimizer_info.step()

        print(
            f"[Epoch {epoch}/{opt.n_epochs}] [iter {i}]"
            + f"[D loss: {d_loss.item():.3f}]"
            + f"[G loss: {g_loss.item():.3f}]"
            + f"[info loss: {info_loss.item():3f}]"
        )
        
        if it % opt.sample_interval == 0:
            write_tboard(writer, it, d_loss.item(), g_loss.item(), info_loss.item())
            