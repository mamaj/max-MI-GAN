import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((-1, *self.shape))


class MaxIMGAN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.output_dim, 256), nn.LeakyReLU(0.2), # nn.Dropout(0.3),
            nn.Linear(256, 256), nn.LeakyReLU(0.2), # nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        # TODO needs to be a separate module with .sample() that clips.
        self.generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 2 * self.output_dim),
            Reshape(2, *self.output_shape)
        )
        
    @property
    def output_dim(self):
        return np.prod(self.output_shape)
        
    @property
    def input_dim(self):
        return np.prod(self.input_shape)
            
    def forward(self, x):
        loc_logstd = self.generator(x)
        loc = loc_logstd[:, 0, ...]
        scale = torch.exp(loc_logstd[:, 1, ...])
        
        # TODO: make this a mixture
        # do I need to make these discrete? PixelCNN++ uses discrete, why?
        return td.Independent(
            td.Normal(loc=loc, scale=scale),
            reinterpreted_batch_ndims=len(self.output_shape) 
        )
    
