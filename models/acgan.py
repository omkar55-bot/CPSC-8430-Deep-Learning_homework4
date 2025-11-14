"""
ACGAN (Auxiliary Classifier GAN) implementation
Based on: https://arxiv.org/abs/1610.09585
"""
import torch
import torch.nn as nn


class ACGANGenerator(nn.Module):
    """
    ACGAN Generator for CIFAR-10
    Takes noise and class label as input
    """
    def __init__(self, latent_dim=100, num_classes=10, num_channels=3, gen_features=64):
        super(ACGANGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding for class label (following baseline: Dense(256, 'relu'))
        self.label_emb = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.ReLU(True)
        )
        
        # Dense layer to get 4*4*512 (following baseline)
        self.dense = nn.Linear(latent_dim + 256, 4 * 4 * gen_features * 8)
        
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
    
    def forward(self, noise, labels):
        # Convert labels to one-hot and embed (following baseline)
        labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=noise.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        label_emb = self.label_emb(labels_onehot)
        
        # Concatenate noise and label embedding
        gen_input = torch.cat([noise, label_emb], dim=1)
        
        # Dense -> Reshape (following baseline)
        x = self.dense(gen_input)
        x = x.view(x.size(0), -1, 4, 4)
        return self.main(x)


class ACGANDiscriminator(nn.Module):
    """
    ACGAN Discriminator for CIFAR-10
    Outputs: real/fake probability and class prediction
    """
    def __init__(self, num_classes=10, num_channels=3, disc_features=64):
        super(ACGANDiscriminator, self).__init__()
        
        self.num_classes = num_classes
        
        # Text embedding (following baseline: Dense(256, 'relu'))
        self.label_emb = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.ReLU(True)
        )
        
        # Shared feature extractor (following baseline: kernel=5)
        self.conv_layers = nn.Sequential(
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
            nn.LeakyReLU(0.2, inplace=True)
            # State: (disc_features*4) x 4 x 4
        )
        
        # Real/Fake branch (following baseline: concatenate with tiled embedding)
        self.adv_layer = nn.Sequential(
            nn.Conv2d(disc_features * 4 + 256, disc_features * 4, 1, 1, 0, bias=False),  # kernel=1 as in baseline
            nn.Flatten(),
            nn.Linear(disc_features * 4 * 4 * 4, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary classifier branch
        self.aux_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(disc_features * 4 * 4 * 4, num_classes)
        )
    
    def forward(self, input, labels=None):
        # Get image features
        image_feat = self.conv_layers(input)  # (batch, disc_features*4, 4, 4)
        
        # Embed labels and tile (following baseline)
        # Always create embedding - use zero embedding if labels not provided
        if labels is not None:
            labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=input.device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            text_emb = self.label_emb(labels_onehot)  # (batch, 256)
        else:
            # Use zero embedding when labels not provided
            text_emb = torch.zeros(input.size(0), 256, device=input.device)
        
        text_emb = text_emb.view(text_emb.size(0), 256, 1, 1)
        tiled_emb = text_emb.repeat(1, 1, 4, 4)  # Tile to (batch, 256, 4, 4)
        
        # Concatenate image features with tiled embedding
        combined = torch.cat([image_feat, tiled_emb], dim=1)  # (batch, disc_features*4 + 256, 4, 4)
        
        # Real/Fake prediction
        validity = self.adv_layer(combined).squeeze(1)
        
        # Class prediction (from image features only)
        label = self.aux_layer(image_feat)
        
        return validity, label


def weights_init(m):
    """Initialize weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Embedding') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

