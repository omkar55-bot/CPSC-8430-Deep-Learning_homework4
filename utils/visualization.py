"""
Visualization utilities for GAN training
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def save_generated_images(generator, device, num_images=64, latent_dim=100, 
                         save_path='generated_images.png', num_classes=10, 
                         gan_type='dcgan', class_labels=None):
    """
    Generate and save images from the generator
    
    Args:
        generator: Generator model
        device: Device to run on
        num_images: Number of images to generate
        latent_dim: Dimension of latent noise vector
        save_path: Path to save the image
        num_classes: Number of classes (for ACGAN)
        gan_type: Type of GAN ('dcgan', 'wgan', 'acgan')
        class_labels: Specific class labels to generate (for ACGAN)
    """
    generator.eval()
    
    with torch.no_grad():
        # Generate random noise
        if gan_type == 'acgan':
            noise = torch.randn(num_images, latent_dim, device=device)
            if class_labels is None:
                # Generate one image per class
                labels = torch.arange(num_classes, device=device).repeat(num_images // num_classes)
                if num_images % num_classes != 0:
                    labels = torch.cat([labels, torch.arange(num_images % num_classes, device=device)])
            else:
                labels = torch.tensor(class_labels, device=device)
            fake_images = generator(noise, labels)
        else:
            noise = torch.randn(num_images, latent_dim, device=device)
            fake_images = generator(noise)
    
    # Denormalize images from [-1, 1] to [0, 1]
    fake_images = (fake_images + 1) / 2.0
    fake_images = torch.clamp(fake_images, 0, 1)
    
    # Create grid
    grid = make_grid(fake_images, nrow=8, normalize=False, padding=2)
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    
    # Save image
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    generator.train()


def plot_training_curves(history, save_path='training_curves.png'):
    """
    Plot training curves for generator and discriminator/critic losses
    
    Args:
        history: Dictionary with 'g_loss' and 'd_loss' or 'c_loss' lists
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot generator loss
    axes[0].plot(history['g_loss'], label='Generator Loss')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Generator Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot discriminator/critic loss (handle both naming conventions)
    disc_key = 'c_loss' if 'c_loss' in history else 'd_loss'
    disc_label = 'Critic Loss' if disc_key == 'c_loss' else 'Discriminator Loss'
    axes[1].plot(history[disc_key], label=disc_label)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(disc_label)
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_best_images(generator, device, latent_dim=100, num_images=10, 
                    save_dir='./results', gan_type='dcgan', num_classes=10):
    """
    Save the best generated images for submission
    
    Args:
        generator: Generator model
        device: Device to run on
        latent_dim: Dimension of latent noise vector
        num_images: Number of best images to save
        save_dir: Directory to save images
        gan_type: Type of GAN
        num_classes: Number of classes (for ACGAN)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    generator.eval()
    
    with torch.no_grad():
        # Generate multiple batches and select best ones
        all_images = []
        all_scores = []
        
        for _ in range(10):  # Generate 10 batches
            if gan_type == 'acgan':
                noise = torch.randn(64, latent_dim, device=device)
                labels = torch.randint(0, num_classes, (64,), device=device)
                fake_images = generator(noise, labels)
            else:
                noise = torch.randn(64, latent_dim, device=device)
                fake_images = generator(noise)
            
            # Simple heuristic: select images with highest variance (more diverse)
            variance = torch.var(fake_images.view(64, -1), dim=1)
            all_images.append(fake_images)
            all_scores.append(variance)
        
        # Concatenate and select top images
        all_images = torch.cat(all_images, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        _, top_indices = torch.topk(all_scores, num_images)
        best_images = all_images[top_indices]
    
    # Save individual images
    best_images = (best_images + 1) / 2.0
    best_images = torch.clamp(best_images, 0, 1)
    
    for i, img in enumerate(best_images):
        img_np = img.cpu().numpy().transpose((1, 2, 0))
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.axis('off')
        plt.savefig(f'{save_dir}/best_image_{i+1}.png', bbox_inches='tight', dpi=150)
        plt.close()
    
    # Save grid
    grid = make_grid(best_images, nrow=5, normalize=False, padding=2)
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    plt.figure(figsize=(12, 6))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.savefig(f'{save_dir}/best_10_images.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    generator.train()

