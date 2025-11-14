"""
Configuration file for GAN training
"""
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configuration
DATASET = 'CIFAR10'
DATA_DIR = './data'
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10

# Model configuration
LATENT_DIM = 100
GEN_FEATURES = 64
DISC_FEATURES = 64

# Training configuration
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# WGAN specific
WGAN_CRITIC_ITERATIONS = 5
WGAN_LAMBDA_GP = 10
WGAN_CLIP_VALUE = 0.01

# ACGAN specific
ACGAN_ALPHA = 1.0  # Weight for classification loss

# Training settings
NUM_WORKERS = 4
SAVE_INTERVAL = 10
LOG_INTERVAL = 100

# Output directories
OUTPUT_DIR = './outputs'
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'

