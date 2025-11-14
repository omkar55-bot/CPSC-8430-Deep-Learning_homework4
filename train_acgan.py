"""
Training script for ACGAN on CIFAR-10
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from models.acgan import ACGANGenerator, ACGANDiscriminator, weights_init
from utils.data_loader import get_cifar10_dataloader
from utils.visualization import save_generated_images, plot_training_curves
import config


def train_acgan():
    """Train ACGAN on CIFAR-10"""
    
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
    generator = ACGANGenerator(
        latent_dim=config.LATENT_DIM,
        num_classes=config.NUM_CLASSES,
        num_channels=config.NUM_CHANNELS,
        gen_features=config.GEN_FEATURES
    ).to(device)
    
    discriminator = ACGANDiscriminator(
        num_classes=config.NUM_CLASSES,
        num_channels=config.NUM_CHANNELS,
        disc_features=config.DISC_FEATURES
    ).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    
    # Optimizers
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
    history = {'g_loss': [], 'd_loss': [], 'd_real_acc': [], 'd_fake_acc': [], 'd_class_acc': []}
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f'{config.OUTPUT_DIR}/acgan_logs')
    
    print("Starting training...")
    print(f"Number of epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Classification loss weight (alpha): {config.ACGAN_ALPHA}")
    
    # Training loop
    iteration = 0
    for epoch in range(config.NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch_idx, (real_images, real_labels) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            
            # =====================
            # Train Discriminator
            # =====================
            discriminator.zero_grad()
            
            # Train with real images
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            validity, pred_label = discriminator(real_images, real_labels)
            
            # Adversarial loss for real images
            errD_real_adv = adversarial_loss(validity, label)
            
            # Classification loss for real images
            errD_real_aux = auxiliary_loss(pred_label, real_labels)
            
            errD_real = errD_real_adv + config.ACGAN_ALPHA * errD_real_aux
            errD_real.backward()
            
            # Train with fake images
            noise = torch.randn(batch_size, config.LATENT_DIM, device=device)
            fake_labels = torch.randint(0, config.NUM_CLASSES, (batch_size,), device=device)
            fake_images = generator(noise, fake_labels)
            
            label.fill_(fake_label)
            validity, pred_label = discriminator(fake_images.detach(), fake_labels)
            
            # Adversarial loss for fake images
            errD_fake_adv = adversarial_loss(validity, label)
            
            # Classification loss for fake images (should be wrong)
            errD_fake_aux = auxiliary_loss(pred_label, fake_labels)
            
            errD_fake = errD_fake_adv + config.ACGAN_ALPHA * errD_fake_aux
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizer_D.step()
            
            # Calculate accuracies
            d_real_acc = (validity > 0.5).float().mean().item()
            d_fake_acc = (validity < 0.5).float().mean().item()
            d_class_acc = (pred_label.argmax(dim=1) == real_labels).float().mean().item()
            
            # =====================
            # Train Generator
            # =====================
            generator.zero_grad()
            
            label.fill_(real_label)  # fake labels are real for generator cost
            validity, pred_label = discriminator(fake_images, fake_labels)
            
            # Adversarial loss
            errG_adv = adversarial_loss(validity, label)
            
            # Classification loss (generator wants correct classification)
            errG_aux = auxiliary_loss(pred_label, fake_labels)
            
            errG = errG_adv + config.ACGAN_ALPHA * errG_aux
            errG.backward()
            optimizer_G.step()
            
            # Update history
            history['g_loss'].append(errG.item())
            history['d_loss'].append(errD.item())
            history['d_real_acc'].append(d_real_acc)
            history['d_fake_acc'].append(d_fake_acc)
            history['d_class_acc'].append(d_class_acc)
            
            # Logging
            if iteration % config.LOG_INTERVAL == 0:
                writer.add_scalar('Loss/Generator', errG.item(), iteration)
                writer.add_scalar('Loss/Discriminator', errD.item(), iteration)
                writer.add_scalar('Loss/G_Adversarial', errG_adv.item(), iteration)
                writer.add_scalar('Loss/G_Auxiliary', errG_aux.item(), iteration)
                writer.add_scalar('Accuracy/D_Real', d_real_acc, iteration)
                writer.add_scalar('Accuracy/D_Fake', d_fake_acc, iteration)
                writer.add_scalar('Accuracy/D_Class', d_class_acc, iteration)
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f'{errG.item():.4f}',
                'D_loss': f'{errD.item():.4f}',
                'D_real_acc': f'{d_real_acc:.2f}',
                'D_class_acc': f'{d_class_acc:.2f}'
            })
            
            iteration += 1
        
        # Save generated images at intervals
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            save_generated_images(
                generator,
                device,
                num_images=64,
                latent_dim=config.LATENT_DIM,
                save_path=f'{config.OUTPUT_DIR}/acgan_epoch_{epoch+1}.png',
                gan_type='acgan',
                num_classes=config.NUM_CLASSES
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
            }, f'{config.CHECKPOINT_DIR}/acgan_epoch_{epoch+1}.pth')
    
    # Save final models and training curves
    print("Saving final models...")
    torch.save(generator.state_dict(), f'{config.CHECKPOINT_DIR}/acgan_generator_final.pth')
    torch.save(discriminator.state_dict(), f'{config.CHECKPOINT_DIR}/acgan_discriminator_final.pth')
    
    plot_training_curves(history, save_path=f'{config.RESULTS_DIR}/acgan_training_curves.png')
    
    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    train_acgan()

