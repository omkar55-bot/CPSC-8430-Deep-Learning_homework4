"""
Training script for WGAN-GP on CIFAR-10
"""
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from models.wgan import WGANGenerator, WGANDiscriminator, compute_gradient_penalty, weights_init
from utils.data_loader import get_cifar10_dataloader
from utils.visualization import save_generated_images, plot_training_curves
import config


def train_wgan():
    """Train WGAN-GP on CIFAR-10"""
    
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
    generator = WGANGenerator(
        latent_dim=config.LATENT_DIM,
        num_channels=config.NUM_CHANNELS,
        gen_features=config.GEN_FEATURES
    ).to(device)
    
    critic = WGANDiscriminator(
        num_channels=config.NUM_CHANNELS,
        disc_features=config.DISC_FEATURES
    ).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    critic.apply(weights_init)
    
    # Optimizers (use RMSprop as recommended in WGAN paper)
    optimizer_G = optim.RMSprop(generator.parameters(), lr=config.LEARNING_RATE)
    optimizer_C = optim.RMSprop(critic.parameters(), lr=config.LEARNING_RATE)
    
    # Training history
    history = {'g_loss': [], 'c_loss': []}
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f'{config.OUTPUT_DIR}/wgan_logs')
    
    print("Starting training...")
    print(f"Number of epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Critic iterations: {config.WGAN_CRITIC_ITERATIONS}")
    print(f"Gradient penalty lambda: {config.WGAN_LAMBDA_GP}")
    
    # Training loop
    iteration = 0
    for epoch in range(config.NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch_idx, (real_images, _) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # =====================
            # Train Critic (more iterations than generator)
            # =====================
            for _ in range(config.WGAN_CRITIC_ITERATIONS):
                critic.zero_grad()
                
                # Real images
                real_output = critic(real_images)
                errC_real = -real_output.mean()
                
                # Fake images
                noise = torch.randn(batch_size, config.LATENT_DIM, device=device)
                fake_images = generator(noise)
                fake_output = critic(fake_images.detach())
                errC_fake = fake_output.mean()
                
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(
                    critic, real_images, fake_images, device
                )
                
                # Critic loss
                errC = errC_real + errC_fake + config.WGAN_LAMBDA_GP * gradient_penalty
                errC.backward()
                optimizer_C.step()
            
            # =====================
            # Train Generator
            # =====================
            generator.zero_grad()
            
            noise = torch.randn(batch_size, config.LATENT_DIM, device=device)
            fake_images = generator(noise)
            fake_output = critic(fake_images)
            errG = -fake_output.mean()
            errG.backward()
            optimizer_G.step()
            
            # Update history
            history['g_loss'].append(errG.item())
            history['c_loss'].append(errC.item())
            
            # Logging
            if iteration % config.LOG_INTERVAL == 0:
                writer.add_scalar('Loss/Generator', errG.item(), iteration)
                writer.add_scalar('Loss/Critic', errC.item(), iteration)
                writer.add_scalar('Score/Real', real_output.mean().item(), iteration)
                writer.add_scalar('Score/Fake', fake_output.mean().item(), iteration)
                writer.add_scalar('Gradient_Penalty', gradient_penalty.item(), iteration)
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f'{errG.item():.4f}',
                'C_loss': f'{errC.item():.4f}',
                'GP': f'{gradient_penalty.item():.4f}'
            })
            
            iteration += 1
        
        # Save generated images at intervals
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            save_generated_images(
                generator,
                device,
                num_images=64,
                latent_dim=config.LATENT_DIM,
                save_path=f'{config.OUTPUT_DIR}/wgan_epoch_{epoch+1}.png',
                gan_type='wgan'
            )
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_C_state_dict': optimizer_C.state_dict(),
                'g_loss': errG.item(),
                'c_loss': errC.item(),
            }, f'{config.CHECKPOINT_DIR}/wgan_epoch_{epoch+1}.pth')
    
    # Save final models and training curves
    print("Saving final models...")
    torch.save(generator.state_dict(), f'{config.CHECKPOINT_DIR}/wgan_generator_final.pth')
    torch.save(critic.state_dict(), f'{config.CHECKPOINT_DIR}/wgan_critic_final.pth')
    
    plot_training_curves(history, save_path=f'{config.RESULTS_DIR}/wgan_training_curves.png')
    
    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    train_wgan()

