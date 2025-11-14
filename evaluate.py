"""
Evaluation script to generate best images and compare GANs
"""
import torch
import os
import argparse

from models.dcgan import DCGANGenerator
from models.wgan import WGANGenerator
from models.acgan import ACGANGenerator
from utils.visualization import save_best_images, save_generated_images
import config


def load_model(gan_type, checkpoint_path, device):
    """Load trained model"""
    if gan_type == 'dcgan':
        generator = DCGANGenerator(
            latent_dim=config.LATENT_DIM,
            num_channels=config.NUM_CHANNELS,
            gen_features=config.GEN_FEATURES
        ).to(device)
    elif gan_type == 'wgan':
        generator = WGANGenerator(
            latent_dim=config.LATENT_DIM,
            num_channels=config.NUM_CHANNELS,
            gen_features=config.GEN_FEATURES
        ).to(device)
    elif gan_type == 'acgan':
        generator = ACGANGenerator(
            latent_dim=config.LATENT_DIM,
            num_classes=config.NUM_CLASSES,
            num_channels=config.NUM_CHANNELS,
            gen_features=config.GEN_FEATURES
        ).to(device)
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")
    
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    
    return generator


def evaluate_gan(gan_type, checkpoint_path, output_dir):
    """Evaluate a trained GAN and save best images"""
    device = config.DEVICE
    
    print(f"Loading {gan_type.upper()} model from {checkpoint_path}...")
    generator = load_model(gan_type, checkpoint_path, device)
    
    # Create output directory
    gan_output_dir = os.path.join(output_dir, gan_type)
    os.makedirs(gan_output_dir, exist_ok=True)
    
    # Save best 10 images
    print(f"Generating best images for {gan_type.upper()}...")
    save_best_images(
        generator,
        device,
        latent_dim=config.LATENT_DIM,
        num_images=10,
        save_dir=gan_output_dir,
        gan_type=gan_type,
        num_classes=config.NUM_CLASSES
    )
    
    # Generate sample grid
    print(f"Generating sample grid for {gan_type.upper()}...")
    save_generated_images(
        generator,
        device,
        num_images=64,
        latent_dim=config.LATENT_DIM,
        save_path=os.path.join(gan_output_dir, 'sample_grid.png'),
        gan_type=gan_type,
        num_classes=config.NUM_CLASSES
    )
    
    print(f"Evaluation complete for {gan_type.upper()}!")
    print(f"Results saved to {gan_output_dir}")


def compare_all_gans():
    """Compare all three GANs"""
    device = config.DEVICE
    output_dir = config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    gan_types = ['dcgan', 'wgan', 'acgan']
    checkpoint_paths = {
        'dcgan': f'{config.CHECKPOINT_DIR}/dcgan_generator_final.pth',
        'wgan': f'{config.CHECKPOINT_DIR}/wgan_generator_final.pth',
        'acgan': f'{config.CHECKPOINT_DIR}/acgan_generator_final.pth'
    }
    
    print("Comparing all GANs...")
    print("=" * 50)
    
    for gan_type in gan_types:
        checkpoint_path = checkpoint_paths[gan_type]
        if os.path.exists(checkpoint_path):
            evaluate_gan(gan_type, checkpoint_path, output_dir)
        else:
            print(f"Warning: Checkpoint not found for {gan_type}: {checkpoint_path}")
    
    print("=" * 50)
    print("Comparison complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained GAN models')
    parser.add_argument('--gan', type=str, choices=['dcgan', 'wgan', 'acgan', 'all'],
                       default='all', help='Which GAN to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (if evaluating single GAN)')
    parser.add_argument('--output_dir', type=str, default=config.RESULTS_DIR,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.gan == 'all':
        compare_all_gans()
    else:
        if args.checkpoint is None:
            args.checkpoint = f'{config.CHECKPOINT_DIR}/{args.gan}_generator_final.pth'
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            print("Please train the model first or provide a valid checkpoint path.")
        else:
            evaluate_gan(args.gan, args.checkpoint, args.output_dir)

