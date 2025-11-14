"""
Performance comparison script for HW4
Generates comparison metrics and visualizations for DCGAN, WGAN, and ACGAN
"""
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.dcgan import DCGANGenerator
from models.wgan import WGANGenerator
from models.acgan import ACGANGenerator
from utils.visualization import save_generated_images
import config


def load_trained_model(gan_type, checkpoint_path, device):
    """Load a trained model"""
    if gan_type == 'dcgan':
        model = DCGANGenerator(
            latent_dim=config.LATENT_DIM,
            num_channels=config.NUM_CHANNELS,
            gen_features=config.GEN_FEATURES
        ).to(device)
    elif gan_type == 'wgan':
        model = WGANGenerator(
            latent_dim=config.LATENT_DIM,
            num_channels=config.NUM_CHANNELS,
            gen_features=config.GEN_FEATURES
        ).to(device)
    elif gan_type == 'acgan':
        model = ACGANGenerator(
            latent_dim=config.LATENT_DIM,
            num_classes=config.NUM_CLASSES,
            num_channels=config.NUM_CHANNELS,
            gen_features=config.GEN_FEATURES
        ).to(device)
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model
    else:
        return None


def compute_inception_score_approx(generator, device, num_samples=5000, gan_type='dcgan'):
    """
    Approximate Inception Score (IS) - simplified version
    Note: Full IS requires Inception network, this is a placeholder
    """
    generator.eval()
    all_images = []
    
    with torch.no_grad():
        num_batches = num_samples // 64
        for _ in range(num_batches):
            noise = torch.randn(64, config.LATENT_DIM, device=device)
            if gan_type == 'acgan':
                labels = torch.randint(0, config.NUM_CLASSES, (64,), device=device)
                images = generator(noise, labels)
            else:
                images = generator(noise)
            all_images.append(images.cpu())
    
    # Placeholder: In real implementation, you'd use Inception network
    # For now, return a simple diversity metric
    all_images = torch.cat(all_images, dim=0)
    diversity = torch.std(all_images.view(len(all_images), -1), dim=0).mean().item()
    
    return {
        'diversity': diversity,
        'note': 'Full IS requires Inception network - this is a simplified metric'
    }


def generate_comparison_report():
    """Generate comprehensive comparison report"""
    device = config.DEVICE
    results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    gan_types = ['dcgan', 'wgan', 'acgan']
    checkpoint_paths = {
        'dcgan': f'{config.CHECKPOINT_DIR}/dcgan_generator_final.pth',
        'wgan': f'{config.CHECKPOINT_DIR}/wgan_generator_final.pth',
        'acgan': f'{config.CHECKPOINT_DIR}/acgan_generator_final.pth'
    }
    
    print("=" * 60)
    print("GAN Performance Comparison Report")
    print("=" * 60)
    
    results = {}
    
    # Load models and generate comparison images
    for gan_type in gan_types:
        print(f"\nProcessing {gan_type.upper()}...")
        checkpoint_path = checkpoint_paths[gan_type]
        
        model = load_trained_model(gan_type, checkpoint_path, device)
        if model is None:
            print(f"  ⚠️  Model not found: {checkpoint_path}")
            print(f"     Please train {gan_type} first using train_{gan_type}.py")
            continue
        
        # Generate sample images for comparison
        print(f"  ✓ Generating comparison images...")
        save_generated_images(
            model,
            device,
            num_images=64,
            latent_dim=config.LATENT_DIM,
            save_path=f'{results_dir}/{gan_type}_comparison_grid.png',
            gan_type=gan_type,
            num_classes=config.NUM_CLASSES
        )
        
        # Compute metrics (simplified)
        print(f"  ✓ Computing metrics...")
        metrics = compute_inception_score_approx(model, device, gan_type=gan_type)
        results[gan_type] = {
            'model': model,
            'metrics': metrics,
            'checkpoint_exists': True
        }
        
        print(f"  ✓ {gan_type.upper()} processed successfully")
    
    # Create comparison visualization
    print("\n" + "=" * 60)
    print("Creating comparison visualization...")
    create_comparison_visualization(results, results_dir)
    
    # Generate text report
    print("Generating text report...")
    generate_text_report(results, results_dir)
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)


def create_comparison_visualization(results, output_dir):
    """Create side-by-side comparison of generated images"""
    device = config.DEVICE
    num_models = len([r for r in results.values() if r.get('checkpoint_exists', False)])
    
    if num_models == 0:
        print("  ⚠️  No trained models found for comparison")
        return
    
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 6))
    if num_models == 1:
        axes = [axes]
    
    idx = 0
    for gan_type, result in results.items():
        if not result.get('checkpoint_exists', False):
            continue
        
        model = result['model']
        model.eval()
        
        with torch.no_grad():
            noise = torch.randn(16, config.LATENT_DIM, device=device)
            if gan_type == 'acgan':
                labels = torch.randint(0, config.NUM_CLASSES, (16,), device=device)
                images = model(noise, labels)
            else:
                images = model(noise)
            
            # Denormalize
            images = (images + 1) / 2.0
            images = torch.clamp(images, 0, 1)
            
            # Create grid
            from torchvision.utils import make_grid
            grid = make_grid(images, nrow=4, normalize=False, padding=2)
            grid_np = grid.cpu().numpy().transpose((1, 2, 0))
            
            axes[idx].imshow(grid_np)
            axes[idx].set_title(f'{gan_type.upper()}\nGenerated Samples', fontsize=14, fontweight='bold')
            axes[idx].axis('off')
            idx += 1
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gan_comparison_side_by_side.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparison visualization saved")


def generate_text_report(results, output_dir):
    """Generate text report with comparison metrics"""
    report_path = f'{output_dir}/performance_comparison_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GAN Performance Comparison Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Dataset: CIFAR-10\n")
        f.write(f"Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}\n")
        f.write(f"Number of Classes: {config.NUM_CLASSES}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        
        for gan_type, result in results.items():
            f.write(f"{gan_type.upper()}:\n")
            f.write("-" * 40 + "\n")
            
            if not result.get('checkpoint_exists', False):
                f.write("  Status: Not trained\n")
            else:
                f.write("  Status: ✓ Trained\n")
                metrics = result.get('metrics', {})
                if 'diversity' in metrics:
                    f.write(f"  Image Diversity: {metrics['diversity']:.4f}\n")
                if 'note' in metrics:
                    f.write(f"  Note: {metrics['note']}\n")
            
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("COMPARISON NOTES\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. Visual Quality: Compare the generated images in the comparison grids\n")
        f.write("2. Training Stability: Check training curves in results/ directory\n")
        f.write("3. Convergence: Review loss curves to see which model converged faster\n")
        f.write("4. Best Images: See results/<gan_type>/best_10_images.png for each model\n\n")
        
        f.write("For full Inception Score (IS) and FID metrics, you would need to:\n")
        f.write("- Use a pretrained Inception network for IS\n")
        f.write("- Use a pretrained Inception network for FID\n")
        f.write("- Compare against real CIFAR-10 statistics\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("BASELINE COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write("Compare your results to:\n")
        f.write("1. Original DCGAN paper results on CIFAR-10\n")
        f.write("2. Original WGAN paper results\n")
        f.write("3. Original ACGAN paper results\n")
        f.write("4. Baseline architecture shown in assignment (adapted for CIFAR-10)\n\n")
        
        f.write("Key differences to note:\n")
        f.write("- Architecture variations (kernel size, layer structure)\n")
        f.write("- Training hyperparameters\n")
        f.write("- Number of training epochs\n")
        f.write("- Image quality and diversity\n")
    
    print(f"  ✓ Text report saved to {report_path}")


if __name__ == '__main__':
    generate_comparison_report()

