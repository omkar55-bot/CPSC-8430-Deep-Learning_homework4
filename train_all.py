"""
Script to train all GAN models sequentially
"""
import subprocess
import sys
import os

def train_all():
    """Train all three GAN models"""
    scripts = [
        ('DCGAN', 'train_dcgan.py'),
        ('WGAN', 'train_wgan.py'),
        ('ACGAN', 'train_acgan.py')
    ]
    
    print("=" * 60)
    print("Training All GAN Models on CIFAR-10")
    print("=" * 60)
    
    for name, script in scripts:
        print(f"\n{'=' * 60}")
        print(f"Training {name}...")
        print(f"{'=' * 60}\n")
        
        try:
            result = subprocess.run(
                [sys.executable, script],
                check=True
            )
            print(f"\n{name} training completed successfully!\n")
        except subprocess.CalledProcessError as e:
            print(f"\nError training {name}: {e}")
            print("Continuing with next model...\n")
        except KeyboardInterrupt:
            print(f"\n\nTraining interrupted during {name}")
            print("Exiting...")
            sys.exit(1)
    
    print("=" * 60)
    print("All models trained successfully!")
    print("=" * 60)
    print("\nRun 'python evaluate.py --gan all' to generate results.")


if __name__ == '__main__':
    train_all()

