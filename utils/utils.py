"""
Utility functions for CycleGAN training.

Includes:
- ReplayBuffer: Stores generated images to stabilize discriminator training
- Image saving and visualization
- Checkpoint save/load
- Learning rate scheduling
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt


class ReplayBuffer:
    """
    Replay Buffer for storing generated images.
    
    This helps stabilize GAN training by showing the discriminator
    a history of generated images rather than just the latest ones.
    
    Strategy:
    - 50% chance: return the new image directly
    - 50% chance: return a random old image from buffer (and store new one)
    
    This prevents the discriminator from overfitting to the generator's
    current output distribution.
    """
    
    def __init__(self, max_size: int = 50):
        """
        Args:
            max_size: Maximum number of images to store (default: 50)
        """
        self.max_size = max_size
        self.buffer = []
    
    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        """
        Add images to buffer and return images for discriminator.
        
        Args:
            images: Batch of generated images (N, C, H, W)
            
        Returns:
            Batch of images (mix of new and buffered)
        """
        result = []
        
        for image in images:
            image = image.unsqueeze(0)  # Add batch dimension
            
            if len(self.buffer) < self.max_size:
                # Buffer not full - add image and return it
                self.buffer.append(image.clone())
                result.append(image)
            else:
                # Buffer full - 50% chance to swap
                if random.random() > 0.5:
                    # Return random old image, store new one
                    idx = random.randint(0, self.max_size - 1)
                    old_image = self.buffer[idx].clone()
                    self.buffer[idx] = image.clone()
                    result.append(old_image)
                else:
                    # Return new image directly
                    result.append(image)
        
        return torch.cat(result, dim=0)
    
    def __len__(self) -> int:
        return len(self.buffer)


def save_checkpoint(
    epoch: int,
    G_AB: nn.Module,
    G_BA: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    path: str,
    scheduler_G: Optional[object] = None,
    scheduler_D: Optional[object] = None,
) -> None:
    """
    Save training checkpoint.
    
    Args:
        epoch: Current epoch number
        G_AB, G_BA: Generator models
        D_A, D_B: Discriminator models
        optimizer_G, optimizer_D: Optimizers
        path: Path to save checkpoint
        scheduler_G, scheduler_D: Optional LR schedulers
    """
    checkpoint = {
        'epoch': epoch,
        'G_AB_state_dict': G_AB.state_dict(),
        'G_BA_state_dict': G_BA.state_dict(),
        'D_A_state_dict': D_A.state_dict(),
        'D_B_state_dict': D_B.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }
    
    if scheduler_G is not None:
        checkpoint['scheduler_G_state_dict'] = scheduler_G.state_dict()
    if scheduler_D is not None:
        checkpoint['scheduler_D_state_dict'] = scheduler_D.state_dict()
    
    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    G_AB: nn.Module,
    G_BA: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    optimizer_G: Optional[torch.optim.Optimizer] = None,
    optimizer_D: Optional[torch.optim.Optimizer] = None,
    scheduler_G: Optional[object] = None,
    scheduler_D: Optional[object] = None,
    device: str = 'cuda',
) -> int:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint file
        G_AB, G_BA: Generator models
        D_A, D_B: Discriminator models
        optimizer_G, optimizer_D: Optional optimizers
        scheduler_G, scheduler_D: Optional LR schedulers
        device: Device to load to
        
    Returns:
        epoch: Epoch number from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    
    G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
    G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
    D_A.load_state_dict(checkpoint['D_A_state_dict'])
    D_B.load_state_dict(checkpoint['D_B_state_dict'])
    
    if optimizer_G is not None and 'optimizer_G_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    if optimizer_D is not None and 'optimizer_D_state_dict' in checkpoint:
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    if scheduler_G is not None and 'scheduler_G_state_dict' in checkpoint:
        scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
    if scheduler_D is not None and 'scheduler_D_state_dict' in checkpoint:
        scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}: {path}")
    
    return epoch


def save_generator_only(
    G_AB: nn.Module,
    path: str,
) -> None:
    """
    Save only the Photo->Monet generator for inference/submission.
    
    Args:
        G_AB: Generator that converts photos to Monet style
        path: Path to save
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(G_AB.state_dict(), path)
    print(f"Generator saved: {path}")


def load_generator_only(
    G_AB: nn.Module,
    path: str,
    device: str = 'cuda',
) -> nn.Module:
    """
    Load generator weights for inference.
    
    Args:
        G_AB: Generator model
        path: Path to saved weights
        device: Device to load to
        
    Returns:
        Generator with loaded weights
    """
    G_AB.load_state_dict(torch.load(path, map_location=device))
    G_AB.eval()
    print(f"Generator loaded: {path}")
    return G_AB


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] to [0, 1] range."""
    return tensor * 0.5 + 0.5


def save_image(
    tensor: torch.Tensor,
    path: str,
    normalize: bool = True,
) -> None:
    """
    Save a tensor as an image.
    
    Args:
        tensor: Image tensor (C, H, W) or (N, C, H, W)
        path: Path to save
        normalize: If True, convert from [-1, 1] to [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image
    
    if normalize:
        tensor = denormalize(tensor)
    
    tensor = torch.clamp(tensor, 0, 1)
    
    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    vutils.save_image(tensor, path)


def save_image_grid(
    tensors: List[torch.Tensor],
    path: str,
    nrow: int = 4,
    normalize: bool = True,
    titles: Optional[List[str]] = None,
) -> None:
    """
    Save multiple images as a grid.
    
    Args:
        tensors: List of image tensors
        path: Path to save
        nrow: Number of images per row
        normalize: If True, convert from [-1, 1] to [0, 1]
        titles: Optional titles (not used in grid, just for reference)
    """
    # Stack tensors
    if tensors[0].dim() == 4:
        tensors = [t[0] for t in tensors]
    
    grid_tensor = torch.stack(tensors)
    
    if normalize:
        grid_tensor = denormalize(grid_tensor)
    
    grid_tensor = torch.clamp(grid_tensor, 0, 1)
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid_tensor, path, nrow=nrow, padding=2)


def visualize_results(
    real_A: torch.Tensor,
    fake_B: torch.Tensor,
    reconstructed_A: torch.Tensor,
    real_B: torch.Tensor,
    fake_A: torch.Tensor,
    reconstructed_B: torch.Tensor,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize CycleGAN results.
    
    Shows:
    Row 1: Real A -> Fake B -> Reconstructed A
    Row 2: Real B -> Fake A -> Reconstructed B
    
    Args:
        real_A: Real image from domain A (Photo)
        fake_B: Generated image in domain B (Monet)
        reconstructed_A: Reconstructed image in domain A
        real_B: Real image from domain B (Monet)
        fake_A: Generated image in domain A (Photo)
        reconstructed_B: Reconstructed image in domain B
        save_path: Optional path to save figure
        show: Whether to display the figure
    """
    def to_numpy(tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]
        tensor = denormalize(tensor)
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.cpu().permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Row 1: A -> B -> A
    axes[0, 0].imshow(to_numpy(real_A))
    axes[0, 0].set_title('Real Photo')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(to_numpy(fake_B))
    axes[0, 1].set_title('Generated Monet')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(to_numpy(reconstructed_A))
    axes[0, 2].set_title('Reconstructed Photo')
    axes[0, 2].axis('off')
    
    # Row 2: B -> A -> B
    axes[1, 0].imshow(to_numpy(real_B))
    axes[1, 0].set_title('Real Monet')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(to_numpy(fake_A))
    axes[1, 1].set_title('Generated Photo')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(to_numpy(reconstructed_B))
    axes[1, 2].set_title('Reconstructed Monet')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    decay_start_epoch: int,
    decay_epochs: int,
) -> lr_scheduler.LambdaLR:
    """
    Get learning rate scheduler with linear decay.
    
    Learning rate stays constant until decay_start_epoch,
    then linearly decays to 0 over decay_epochs.
    
    Args:
        optimizer: Optimizer to schedule
        n_epochs: Total number of epochs
        decay_start_epoch: Epoch to start decay
        decay_epochs: Number of epochs to decay over
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < decay_start_epoch:
            return 1.0
        else:
            # Linear decay from 1.0 to 0.0
            return 1.0 - (epoch - decay_start_epoch) / decay_epochs
    
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_requires_grad(models: List[nn.Module], requires_grad: bool) -> None:
    """
    Set requires_grad for all parameters in models.
    
    Useful for freezing discriminator when training generator and vice versa.
    
    Args:
        models: List of models
        requires_grad: Whether to require gradients
    """
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(
    G_AB: nn.Module,
    G_BA: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
) -> None:
    """Print summary of all models."""
    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Generator A->B: {count_parameters(G_AB):,} parameters")
    print(f"Generator B->A: {count_parameters(G_BA):,} parameters")
    print(f"Discriminator A: {count_parameters(D_A):,} parameters")
    print(f"Discriminator B: {count_parameters(D_B):,} parameters")
    print(f"Total: {count_parameters(G_AB) + count_parameters(G_BA) + count_parameters(D_A) + count_parameters(D_B):,} parameters")
    print("=" * 50)


def test_utils():
    """Test utility functions."""
    print("Testing Utility Functions...")
    print("-" * 50)
    
    # Test ReplayBuffer
    print("\n1. Testing ReplayBuffer:")
    buffer = ReplayBuffer(max_size=10)
    fake_images = torch.randn(4, 3, 256, 256)
    output = buffer.push_and_pop(fake_images)
    print(f"   Input shape: {fake_images.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Buffer size: {len(buffer)}")
    
    # Test LR scheduler
    print("\n2. Testing LR Scheduler:")
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = get_lr_scheduler(optimizer, n_epochs=30, decay_start_epoch=15, decay_epochs=15)
    
    lrs = []
    for epoch in range(30):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    print(f"   Epoch 0 LR: {lrs[0]:.6f}")
    print(f"   Epoch 14 LR: {lrs[14]:.6f}")
    print(f"   Epoch 20 LR: {lrs[20]:.6f}")
    print(f"   Epoch 29 LR: {lrs[29]:.6f}")
    
    # Test set_requires_grad
    print("\n3. Testing set_requires_grad:")
    model = nn.Linear(10, 10)
    print(f"   Before: requires_grad = {model.weight.requires_grad}")
    set_requires_grad([model], False)
    print(f"   After: requires_grad = {model.weight.requires_grad}")
    
    print("\n✅ All utility functions working!")


if __name__ == "__main__":
    test_utils()
