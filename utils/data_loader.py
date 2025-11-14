"""
Data loading utilities for CIFAR-10
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_dataloader(data_dir, batch_size=128, num_workers=4, train=True):
    """
    Get CIFAR-10 data loader
    
    Args:
        data_dir: Directory to store/load CIFAR-10 data
        batch_size: Batch size for training
        num_workers: Number of worker threads
        train: Whether to load training or test set
        
    Returns:
        DataLoader for CIFAR-10
    """
    # Normalize to [-1, 1] for tanh output
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_cifar10_classes():
    """Get CIFAR-10 class names"""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

