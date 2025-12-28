"""
Dataset classes for CycleGAN training.
Handles loading unpaired images from Monet and Photo domains.
"""

import os
import random
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Callable, List

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """
    Dataset for unpaired image-to-image translation.
    Loads images from two domains (e.g., Monet paintings and real photos).
    
    Since CycleGAN uses unpaired data, images from domain A and B are
    randomly paired during training.
    """
    
    def __init__(
        self,
        root_A: str,
        root_B: str,
        transform: Optional[Callable] = None,
        mode: str = "train",
        unaligned: bool = True,
    ):
        """
        Args:
            root_A: Path to domain A images (e.g., Monet paintings)
            root_B: Path to domain B images (e.g., real photos)
            transform: Torchvision transforms to apply
            mode: "train" or "test"
            unaligned: If True, randomly pair images from A and B
        """
        self.transform = transform
        self.unaligned = unaligned
        self.mode = mode
        
        # Get all image paths
        self.files_A = sorted(self._get_image_files(root_A))
        self.files_B = sorted(self._get_image_files(root_B))
        
        if len(self.files_A) == 0:
            raise ValueError(f"No images found in {root_A}")
        if len(self.files_B) == 0:
            raise ValueError(f"No images found in {root_B}")
            
        print(f"Found {len(self.files_A)} images in domain A (Monet)")
        print(f"Found {len(self.files_B)} images in domain B (Photos)")
    
    def _get_image_files(self, directory: str) -> List[str]:
        """Get all image files from a directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for f in os.listdir(directory):
            if Path(f).suffix.lower() in valid_extensions:
                image_files.append(os.path.join(directory, f))
        
        return image_files
    
    def __getitem__(self, index: int) -> dict:
        """
        Returns a dict with:
            - 'A': Image from domain A (Monet)
            - 'B': Image from domain B (Photo)
        """
        # Get image from domain A
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        
        # Get image from domain B (random if unaligned)
        if self.unaligned:
            idx_B = random.randint(0, len(self.files_B) - 1)
        else:
            idx_B = index % len(self.files_B)
        img_B = Image.open(self.files_B[idx_B]).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        
        return {'A': img_A, 'B': img_B}
    
    def __len__(self) -> int:
        """Return the maximum of the two domain sizes."""
        return max(len(self.files_A), len(self.files_B))


class SingleDomainDataset(Dataset):
    """
    Dataset for loading images from a single domain.
    Useful for inference when generating images.
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: Path to images
            transform: Torchvision transforms to apply
        """
        self.transform = transform
        self.files = sorted(self._get_image_files(root))
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root}")
        
        print(f"Found {len(self.files)} images")
    
    def _get_image_files(self, directory: str) -> List[str]:
        """Get all image files from a directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for f in os.listdir(directory):
            if Path(f).suffix.lower() in valid_extensions:
                image_files.append(os.path.join(directory, f))
        
        return image_files
    
    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.files[index]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img
    
    def __len__(self) -> int:
        return len(self.files)


def get_transforms(img_size: int = 256, mode: str = "train") -> transforms.Compose:
    """
    Get transforms for training or testing.
    
    Training transforms include:
        - Resize to slightly larger than target
        - Random crop to target size
        - Random horizontal flip
        - Convert to tensor
        - Normalize to [-1, 1]
    
    Test transforms:
        - Resize to target size
        - Convert to tensor
        - Normalize to [-1, 1]
    """
    if mode == "train":
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.12), Image.BICUBIC),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    return transform


def get_dataloaders(
    monet_path: str,
    photo_path: str,
    img_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test dataloaders.
    
    Returns:
        train_loader: DataLoader for training
        photo_loader: DataLoader for photos only (for inference)
    """
    # Training dataloader (paired A and B)
    train_transform = get_transforms(img_size, mode="train")
    train_dataset = ImageDataset(
        root_A=monet_path,
        root_B=photo_path,
        transform=train_transform,
        mode="train",
        unaligned=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    # Photo-only dataloader for inference
    test_transform = get_transforms(img_size, mode="test")
    photo_dataset = SingleDomainDataset(
        root=photo_path,
        transform=test_transform,
    )
    
    photo_loader = DataLoader(
        photo_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, photo_loader


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from [-1, 1] to [0, 1] range.
    Useful for visualization and saving images.
    """
    return tensor * 0.5 + 0.5


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to PIL Image.
    Assumes tensor is in [-1, 1] range with shape (C, H, W) or (B, C, H, W).
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image if batched
    
    # Denormalize and clamp
    tensor = denormalize(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and then PIL
    img_np = tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype('uint8')
    
    return Image.fromarray(img_np)


# For testing the dataset
if __name__ == "__main__":
    # Test with dummy paths (update these for actual testing)
    print("Dataset module loaded successfully!")
    print("\nAvailable functions:")
    print("  - ImageDataset: For training with paired domains")
    print("  - SingleDomainDataset: For inference")
    print("  - get_transforms: Get train/test transforms")
    print("  - get_dataloaders: Create train and test loaders")
    print("  - denormalize: Convert [-1,1] to [0,1]")
    print("  - tensor_to_image: Convert tensor to PIL Image")
