"""
WGAN (Wasserstein GAN) implementation with Gradient Penalty
Based on: https://arxiv.org/abs/1701.07875
"""
import torch
import torch.nn as nn
import torch.autograd as autograd


class WGANGenerator(nn.Module):
    """
    WGAN Generator for CIFAR-10 (32x32 images)
    Same architecture as DCGAN but without batch norm in last layer
    """
    def __init__(self, latent_dim=100, num_channels=3, gen_features=64):
        super(WGANGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, gen_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_features * 8),
            nn.ReLU(True),
            # State: (gen_features*8) x 4 x 4
            
            nn.ConvTranspose2d(gen_features * 8, gen_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 4),
            nn.ReLU(True),
            # State: (gen_features*4) x 8 x 8
            
            nn.ConvTranspose2d(gen_features * 4, gen_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 2),
            nn.ReLU(True),
            # State: (gen_features*2) x 16 x 16
            
            nn.ConvTranspose2d(gen_features * 2, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: num_channels x 32 x 32
        )
    
    def forward(self, input):
        input = input.view(input.size(0), self.latent_dim, 1, 1)
        return self.main(input)


class WGANDiscriminator(nn.Module):
    """
    WGAN Critic (Discriminator) for CIFAR-10
    Note: WGAN uses a critic instead of discriminator (no sigmoid)
    """
    def __init__(self, num_channels=3, disc_features=64):
        super(WGANDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: num_channels x 32 x 32
            nn.Conv2d(num_channels, disc_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (disc_features) x 16 x 16
            
            nn.Conv2d(disc_features, disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (disc_features*2) x 8 x 8
            
            nn.Conv2d(disc_features * 2, disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (disc_features*4) x 4 x 4
            
            nn.Conv2d(disc_features * 4, 1, 4, 1, 0, bias=False)
            # Output: 1 x 1 x 1 (no sigmoid for WGAN)
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """
    Compute gradient penalty for WGAN-GP
    
    Args:
        critic: WGAN critic network
        real_samples: Real images
        fake_samples: Generated images
        device: Device to run on
        
    Returns:
        Gradient penalty value
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Calculate critic scores
    critic_interpolates = critic(interpolates)
    
    # Get gradients
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def weights_init(m):
    """Initialize weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

