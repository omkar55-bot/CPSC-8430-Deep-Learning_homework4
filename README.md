# GAN Training on CIFAR-10

This project implements and trains three types of Generative Adversarial Networks (GANs) on the CIFAR-10 dataset:

1. **DCGAN** (Deep Convolutional GAN) - [Paper](https://arxiv.org/abs/1511.06434)
2. **WGAN** (Wasserstein GAN with Gradient Penalty) - [Paper](https://arxiv.org/abs/1701.07875)
3. **ACGAN** (Auxiliary Classifier GAN) - [Paper](https://arxiv.org/abs/1610.09585)

## Project Structure

```
DLHW4/
├── models/
│   ├── __init__.py
│   ├── dcgan.py          # DCGAN implementation
│   ├── wgan.py           # WGAN implementation
│   └── acgan.py          # ACGAN implementation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # CIFAR-10 data loading utilities
│   └── visualization.py  # Image generation and visualization utilities
├── config.py             # Configuration parameters
├── train_dcgan.py        # Training script for DCGAN
├── train_wgan.py         # Training script for WGAN
├── train_acgan.py        # Training script for ACGAN
├── evaluate.py           # Evaluation and comparison script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DLHW4
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train each GAN model separately:

#### DCGAN
```bash
python train_dcgan.py
```

#### WGAN
```bash
python train_wgan.py
```

#### ACGAN
```bash
python train_acgan.py
```

### Evaluation

After training, generate the best images and compare models:

```bash
# Evaluate all trained models
python evaluate.py --gan all

# Evaluate a specific model
python evaluate.py --gan dcgan --checkpoint checkpoints/dcgan_generator_final.pth
```

## Configuration

Edit `config.py` to modify training parameters:

- `BATCH_SIZE`: Batch size for training (default: 128)
- `NUM_EPOCHS`: Number of training epochs (default: 100)
- `LEARNING_RATE`: Learning rate for optimizers (default: 0.0002)
- `LATENT_DIM`: Dimension of noise vector (default: 100)
- `WGAN_CRITIC_ITERATIONS`: Number of critic updates per generator update (default: 5)
- `WGAN_LAMBDA_GP`: Gradient penalty coefficient (default: 10)

## Model Architectures

### DCGAN
- **Generator**: Transposed convolutions with batch normalization and ReLU activations
- **Discriminator**: Convolutions with batch normalization and LeakyReLU activations
- **Loss**: Binary cross-entropy loss

### WGAN
- **Generator**: Similar to DCGAN
- **Critic**: Similar to DCGAN discriminator but without sigmoid (outputs raw scores)
- **Loss**: Wasserstein distance with gradient penalty

### ACGAN
- **Generator**: Takes noise and class label as input
- **Discriminator**: Outputs both real/fake prediction and class prediction
- **Loss**: Combined adversarial loss and classification loss

## Output Structure

After training, the following directories will be created:

- `outputs/`: Generated images during training
- `checkpoints/`: Model checkpoints
- `results/`: Final evaluation results and best images

## Results

The evaluation script will generate:
- 10 best generated images per GAN
- Sample grids showing generated images
- Training curves showing loss progression

## Dataset

The CIFAR-10 dataset will be automatically downloaded to `./data/` on first run.

CIFAR-10 consists of 60,000 32x32 color images in 10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

## References

- DCGAN: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- WGAN: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- ACGAN: [Conditional Image Synthesis With Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585)
- CIFAR-10: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html)

## Notes

- Training on GPU is highly recommended for faster convergence
- Models are saved at regular intervals (every 10 epochs by default)
- TensorBoard logs are saved in `outputs/<gan_type>_logs/` for monitoring training

