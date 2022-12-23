# Modifiing info gan to take images as codes.
# taken from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/infogan/infogan.py


import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.tensorboard import SummaryWriter

from utils import create_digit_grid, get_logdir, show, display, dict2mdtable

# os.makedirs("images/static/", exist_ok=True)
# os.makedirs("images/varying_c1/", exist_ok=True)
# os.makedirs("images/varying_c2/", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--small_img_size", type=int, default=8, help="size of each smaller image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
parser.add_argument("--lambda_rec", type=float, default=0.1, help="recons loss weight")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.small_img_size ** 2

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=False),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=False),
            
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, small_img):
        small_img = torch.flatten(small_img, start_dim=1)
        gen_input = torch.cat((noise, small_img), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=False),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                *discriminator_block(opt.channels, 16, bn=False),
                *discriminator_block(16, 32),
            ),
            nn.Sequential(
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )
        ])

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        # self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())
        # self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.code_dim))
        
        self.make_small = nn.Sequential(
            # nn.BatchNorm2d(32),

            # nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=False),

            # nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8, 0.8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(8, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )


    def forward(self, img):
        out = self.conv_blocks[0](img)
        small_img = self.make_small(out)
        
        out = self.conv_blocks[1](out)
        validity = self.adv_layer(torch.flatten(out, start_dim=1))
        
        # label = self.aux_layer(out)
        # latent_code = self.latent_layer(out)
        
        return validity, small_img


# Loss functions
adversarial_loss = torch.nn.MSELoss()
small_img_loss = torch.nn.MSELoss()
# categorical_loss = torch.nn.CrossEntropyLoss()
# continuous_loss = torch.nn.MSELoss()

# Loss weights
# lambda_cat = 1
# lambda_con = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    small_img_loss.cuda()
    # categorical_loss.cuda()
    # continuous_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
DATASET_FOLDER = 'data/mnist'
os.makedirs(DATASET_FOLDER, exist_ok=True)

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        DATASET_FOLDER,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

small_dataloader = torch.utils.data.DataLoader(
    small_ds := datasets.MNIST(
        DATASET_FOLDER,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(opt.small_img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling
# static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
# static_label = to_categorical(
#     np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
# )
# static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))


# def sample_image(n_row, batches_done):
#     """Saves a grid of generated digits ranging from 0 to n_classes"""
#     # Static sample
#     z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
#     static_sample = generator(z, static_label, static_code)
#     save_image(static_sample.data, "images/static/%d.png" % batches_done, nrow=n_row, normalize=True)

#     # Get varied c1 and c2
#     zeros = np.zeros((n_row ** 2, 1))
#     c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
#     c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
#     c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
#     sample1 = generator(static_z, static_label, c1)
#     sample2 = generator(static_z, static_label, c2)
#     save_image(sample1.data, "images/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
#     save_image(sample2.data, "images/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)


cols = 8
grid = create_digit_grid(small_ds, cols=cols)
static_z = Variable(FloatTensor(np.random.normal(0, 1, (cols * 10, opt.latent_dim))))
if cuda:
    grid = grid.cuda()

def write_tboard(writer, it, d_loss, g_loss, info_loss):
    writer.add_scalar('loss/g', g_loss, it)
    writer.add_scalar('loss/d', d_loss, it)
    writer.add_scalar('loss/info', info_loss, it)
    
    generator.eval()
    y = generator(static_z, grid)
    _, x_hat = discriminator(y)

    writer.add_image('images/x', make_grid(grid, normalize=True), it)
    writer.add_image('images/x_hat', make_grid(x_hat, normalize=True), it)
    writer.add_image('images/y', make_grid(y, normalize=True), it)


# ----------
#  Training
# ----------

writer = SummaryWriter(get_logdir(name='mod_infogan'))
writer.add_text('params', dict2mdtable(vars(opt)), 1)

it = 0
for epoch in range(opt.n_epochs):
    for i, ((imgs, _), (small_imgs, _)) in enumerate(zip(dataloader, small_dataloader)):
        it += 1
        
        generator.train()
        
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        small_imgs = Variable(small_imgs.type(FloatTensor))
        # labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        # label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
        # code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))
        

        # Generate a batch of images
        gen_imgs = generator(z, small_imgs)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        
        # ------------------
        # Information Loss
        # ------------------
        optimizer_info.zero_grad()

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_imgs = generator(z, small_imgs)
        _, small_imgs_hat = discriminator(gen_imgs)

        info_loss = opt.lambda_rec * small_img_loss(small_imgs, small_imgs_hat)
        info_loss.backward()
        optimizer_info.step()


        # # ------------------
        # # Information Loss
        # # ------------------

        # optimizer_info.zero_grad()

        # # Sample labels
        # sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

        # # Ground truth labels
        # gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

        # # Sample noise, labels and code as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        # label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
        # code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        # gen_imgs = generator(z, label_input, code_input)
        # _, pred_label, pred_code = discriminator(gen_imgs)

        # info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
        #     pred_code, code_input
        # )

        # info_loss.backward()
        # optimizer_info.step()

        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
        )
        # batches_done = epoch * len(dataloader) + i
        if it % opt.sample_interval == 0:
            # sample_image(n_row=10, batches_done=batches_done)
            write_tboard(writer, it, d_loss.item(), g_loss.item(), info_loss.item())
            