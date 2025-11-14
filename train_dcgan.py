"""
Training script for DCGAN on CIFAR-10
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from models.dcgan import DCGANGenerator, DCGANDiscriminator, weights_init
from utils.data_loader import get_cifar10_dataloader
from utils.visualization import save_generated_images, plot_training_curves
import config


def train_dcgan():
    """Train DCGAN on CIFAR-10"""
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Initialize device
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_dataloader(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Initialize models
    print("Initializing models...")
    generator = DCGANGenerator(
        latent_dim=config.LATENT_DIM,
        num_channels=config.NUM_CHANNELS,
        gen_features=config.GEN_FEATURES
    ).to(device)
    
    discriminator = DCGANDiscriminator(
        num_channels=config.NUM_CHANNELS,
        disc_features=config.DISC_FEATURES
    ).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    
    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2)
    )
    
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2)
    )
    
    # Labels
    real_label = 1.0
    fake_label = 0.0
    
    # Training history
    history = {'g_loss': [], 'd_loss': []}
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f'{config.OUTPUT_DIR}/dcgan_logs')
    
    print("Starting training...")
    print(f"Number of epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    
    # Training loop
    iteration = 0
    for epoch in range(config.NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch_idx, (real_images, _) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # =====================
            # Train Discriminator
            # =====================
            discriminator.zero_grad()
            
            # Train with real images
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train with fake images
            noise = torch.randn(batch_size, config.LATENT_DIM, device=device)
            fake_images = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_images.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizer_D.step()
            
            # =====================
            # Train Generator
            # =====================
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = discriminator(fake_images)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()
            
            # Update history
            history['g_loss'].append(errG.item())
            history['d_loss'].append(errD.item())
            
            # Logging
            if iteration % config.LOG_INTERVAL == 0:
                writer.add_scalar('Loss/Generator', errG.item(), iteration)
                writer.add_scalar('Loss/Discriminator', errD.item(), iteration)
                writer.add_scalar('Score/D_x', D_x, iteration)
                writer.add_scalar('Score/D_G_z', D_G_z2, iteration)
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f'{errG.item():.4f}',
                'D_loss': f'{errD.item():.4f}',
                'D(x)': f'{D_x:.4f}',
                'D(G(z))': f'{D_G_z2:.4f}'
            })
            
            iteration += 1
        
        # Save generated images at intervals
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            save_generated_images(
                generator,
                device,
                num_images=64,
                latent_dim=config.LATENT_DIM,
                save_path=f'{config.OUTPUT_DIR}/dcgan_epoch_{epoch+1}.png',
                gan_type='dcgan'
            )
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_loss': errG.item(),
                'd_loss': errD.item(),
            }, f'{config.CHECKPOINT_DIR}/dcgan_epoch_{epoch+1}.pth')
    
    # Save final models and training curves
    print("Saving final models...")
    torch.save(generator.state_dict(), f'{config.CHECKPOINT_DIR}/dcgan_generator_final.pth')
    torch.save(discriminator.state_dict(), f'{config.CHECKPOINT_DIR}/dcgan_discriminator_final.pth')
    
    plot_training_curves(history, save_path=f'{config.RESULTS_DIR}/dcgan_training_curves.png')
    
    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    train_dcgan()

