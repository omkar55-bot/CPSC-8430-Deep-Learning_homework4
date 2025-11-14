"""
DCGAN (Deep Convolutional GAN) implementation
Based on: https://arxiv.org/abs/1511.06434
"""
import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator for CIFAR-10 (32x32 images)
    Following baseline architecture: Dense -> Reshape -> ConvTranspose with kernel=5
    """
    def __init__(self, latent_dim=100, num_channels=3, gen_features=64):
        super(DCGANGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Dense layer to get 4*4*512 (following baseline)
        self.dense = nn.Linear(latent_dim, 4 * 4 * gen_features * 8)
        
        # Starting size: 4x4
        self.main = nn.Sequential(
            # Reshape to (gen_features*8) x 4 x 4
            nn.BatchNorm2d(gen_features * 8, momentum=0.9),
            nn.ReLU(True),
            # State: (gen_features*8) x 4 x 4
            
            nn.ConvTranspose2d(gen_features * 8, gen_features * 4, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(gen_features * 4, momentum=0.9),
            nn.ReLU(True),
            # State: (gen_features*4) x 8 x 8
            
            nn.ConvTranspose2d(gen_features * 4, gen_features * 2, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(gen_features * 2, momentum=0.9),
            nn.ReLU(True),
            # State: (gen_features*2) x 16 x 16
            
            nn.ConvTranspose2d(gen_features * 2, num_channels, 5, 2, 2, output_padding=1, bias=False),
            nn.Tanh()
            # Output: num_channels x 32 x 32
        )
    
    def forward(self, input):
        # Dense -> Reshape (following baseline)
        x = self.dense(input)
        x = x.view(x.size(0), -1, 4, 4)
        return self.main(x)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator for CIFAR-10 (32x32 images)
    Following baseline architecture: Conv2D with kernel=5
    """
    def __init__(self, num_channels=3, disc_features=64):
        super(DCGANDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: num_channels x 32 x 32
            nn.Conv2d(num_channels, disc_features, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (disc_features) x 16 x 16
            
            nn.Conv2d(disc_features, disc_features * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(disc_features * 2, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (disc_features*2) x 8 x 8
            
            nn.Conv2d(disc_features * 2, disc_features * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(disc_features * 4, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (disc_features*4) x 4 x 4
            
            nn.Conv2d(disc_features * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


def weights_init(m):
    """Initialize weights according to DCGAN paper"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

