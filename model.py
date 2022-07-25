import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np


class Generator(nn.Module):
    def __init__(self, input_shape, output_shape, ncomps=1):
        super().__init__()
        self.input_shape = input_shape
        self.input_dim = np.prod(self.input_shape)
        self.output_shape = output_shape
        self.output_dim = np.prod(self.output_shape)
        self.ncomps = ncomps
        
        if ncomps == 1:
            layers_out_dim = 2 * self.output_dim
            self.output_dist = self.normal_dist
        else:
            layers_out_dim = 3 * self.ncomps * self.output_dim
            self.output_dist = self.mixture_dist
            
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 256),nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, layers_out_dim),
        )
    
    def mixture_dist(self, x):
        x = x.view(-1, *self.output_shape, 3 * self.ncomps)
        comp_logits, locs, log_scales = torch.tensor_split(x, 3, dim=-1)
        scales = log_scales.exp()

        return (
            td.Independent(
                td.MixtureSameFamily(
                    mixture_distribution=td.Categorical(logits=comp_logits),
                    component_distribution=td.Normal(locs, scales)
                    ),
                reinterpreted_batch_ndims=len(self.output_shape)
                )
        )
        
    def normal_dist(self, x):
        x = x.view(-1, *self.output_shape, 2)
        loc, log_scale = x[..., 0], x[..., 1]

        return (
            td.Independent(
                td.Normal(
                    loc = loc,
                    scale = log_scale.exp() + 1e-5
                ),
                reinterpreted_batch_ndims=len(self.output_shape)
                )
        )


    def beta_dist(self, x):
        x = x.view(-1, *self.output_shape, 2)
        a, b = x[..., 0], x[..., 1]
        
        f = td.AffineTransform(-1, 2)
        dist = (
            td.Independent(
                td.TransformedDistribution(td.Beta(a.abs(), b.abs()), f),
                reinterpreted_batch_ndims=len(self.output_shape)
                )
        )
        
        dist.entropy = lambda: dist.base_dist.base_dist.entropy().sum((1, 2, 3))
        return dist
        
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        return self.output_dist(x)
        
        
    def sample(self, x, n_samples, clip=False):
        samples = self(x).sample(n_samples)
        if clip:
            return samples.clip(-1, 1)
        else:
            return samples

    
    

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_dim = np.prod(input_shape)
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.LeakyReLU(0.2), # nn.Dropout(0.3),
            nn.Linear(256, 256), nn.LeakyReLU(0.2), # nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x.flatten(start_dim=-3)
        return self.layers(x)
        
        
        
        
class MaxIMGAN(nn.Module):
    def __init__(self, input_shape, output_shape, n_mixtures):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_dim = np.prod(self.output_shape)
        self.input_dim = np.prod(self.input_shape)
        
        self.generator = Generator(input_shape, output_shape, n_mixtures)
        self.discriminator = Discriminator(output_shape)
        
    def forward(self, x):
        return self.generator(x)
        
        # TODO: make this a mixture
        # do I need to make these discrete? PixelCNN++ uses discrete, why?