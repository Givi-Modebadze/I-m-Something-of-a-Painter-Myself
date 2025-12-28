"""
Inference Script for CycleGAN.

Generates Monet-style images from photos for Kaggle submission.
Outputs a zip file with 7,000-10,000 images sized 256x256.

Usage:
    # In Colab/Kaggle notebook, run:
    from inference import generate_submission
    generate_submission(generator_path, photo_dir, output_path, num_images)
"""

import os
import zipfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Import our modules
from models import get_generator
from data.dataset import SingleDomainDataset, get_transforms, denormalize


def load_generator(
    checkpoint_path: str,
    generator_type: str = "unet",
    device: str = "cuda",
) -> nn.Module:
    """
    Load trained generator for inference.
    
    Args:
        checkpoint_path: Path to generator weights (.pth file)
        generator_type: "unet" or "resnet"
        device: Device to load model on
        
    Returns:
        Generator model in eval mode
    """
    # Create generator
    generator = get_generator(
        generator_type,
        in_channels=3,
        out_channels=3,
    )
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    
    # Set to eval mode and move to device
    generator = generator.to(device)
    generator.eval()
    
    print(f"Generator loaded from: {checkpoint_path}")
    print(f"Generator type: {generator_type}")
    print(f"Device: {device}")
    
    return generator


def generate_images(
    generator: nn.Module,
    photo_dir: str,
    output_dir: str,
    num_images: int = 7500,
    img_size: int = 256,
    device: str = "cuda",
    batch_size: int = 1,
) -> int:
    """
    Generate Monet-style images from photos.
    
    Args:
        generator: Trained generator model
        photo_dir: Directory containing source photos
        output_dir: Directory to save generated images
        num_images: Number of images to generate (7000-10000 for Kaggle)
        img_size: Output image size (256 for Kaggle)
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        Number of images generated
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup transform
    transform = get_transforms(img_size, mode="test")
    
    # Load photos
    dataset = SingleDomainDataset(photo_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    print(f"Generating {num_images} images...")
    print(f"Source photos: {len(dataset)}")
    
    count = 0
    
    with torch.no_grad():
        for i, photos in enumerate(tqdm(dataloader, desc="Generating")):
            if count >= num_images:
                break
            
            photos = photos.to(device)
            
            # Generate Monet-style images
            fake_monets = generator(photos)
            
            # Save each image in batch
            for j, fake_monet in enumerate(fake_monets):
                if count >= num_images:
                    break
                
                # Denormalize from [-1, 1] to [0, 1]
                img = denormalize(fake_monet)
                img = torch.clamp(img, 0, 1)
                
                # Convert to PIL Image
                img_np = img.cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype('uint8')
                pil_img = Image.fromarray(img_np)
                
                # Save as JPEG
                img_path = os.path.join(output_dir, f"{count:05d}.jpg")
                pil_img.save(img_path, "JPEG", quality=95)
                
                count += 1
    
    print(f"Generated {count} images in {output_dir}")
    return count


def create_submission_zip(
    image_dir: str,
    zip_path: str = "images.zip",
) -> str:
    """
    Create submission zip file from generated images.
    
    Args:
        image_dir: Directory containing generated images
        zip_path: Output zip file path
        
    Returns:
        Path to created zip file
    """
    print(f"Creating submission zip: {zip_path}")
    
    # Get all jpg files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img_file in tqdm(image_files, desc="Zipping"):
            img_path = os.path.join(image_dir, img_file)
            zipf.write(img_path, img_file)
    
    # Get zip file size
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
    
    print(f"Submission zip created: {zip_path}")
    print(f"Contains {len(image_files)} images")
    print(f"Size: {zip_size:.1f} MB")
    
    return zip_path


def generate_submission(
    generator_path: str,
    photo_dir: str,
    output_dir: str = "./generated_images",
    zip_path: str = "images.zip",
    num_images: int = 7500,
    generator_type: str = "unet",
    device: str = None,
) -> str:
    """
    Complete submission generation pipeline.
    
    Args:
        generator_path: Path to trained generator weights
        photo_dir: Directory containing source photos
        output_dir: Directory to save generated images
        zip_path: Output zip file path
        num_images: Number of images to generate
        generator_type: "unet" or "resnet"
        device: Device to run on (auto-detect if None)
        
    Returns:
        Path to submission zip file
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 50)
    print("CycleGAN Submission Generator")
    print("=" * 50)
    
    # Load generator
    generator = load_generator(generator_path, generator_type, device)
    
    # Generate images
    count = generate_images(
        generator=generator,
        photo_dir=photo_dir,
        output_dir=output_dir,
        num_images=num_images,
        device=device,
    )
    
    # Create submission zip
    zip_file = create_submission_zip(output_dir, zip_path)
    
    print("=" * 50)
    print("Submission ready!")
    print(f"Upload {zip_path} to Kaggle")
    print("=" * 50)
    
    return zip_file


def visualize_samples(
    generator_path: str,
    photo_dir: str,
    num_samples: int = 5,
    generator_type: str = "unet",
    device: str = None,
):
    """
    Visualize some sample generations before full submission.
    
    Args:
        generator_path: Path to trained generator weights
        photo_dir: Directory containing source photos
        num_samples: Number of samples to visualize
        generator_type: "unet" or "resnet"
        device: Device to run on
    """
    import matplotlib.pyplot as plt
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load generator
    generator = load_generator(generator_path, generator_type, device)
    
    # Setup transform
    transform = get_transforms(256, mode="test")
    dataset = SingleDomainDataset(photo_dir, transform=transform)
    
    # Generate samples
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            photo = dataset[i].unsqueeze(0).to(device)
            fake_monet = generator(photo)
            
            # Convert to display format
            photo_np = denormalize(photo[0]).cpu().permute(1, 2, 0).numpy()
            monet_np = denormalize(fake_monet[0]).cpu().permute(1, 2, 0).numpy()
            
            photo_np = photo_np.clip(0, 1)
            monet_np = monet_np.clip(0, 1)
            
            axes[i, 0].imshow(photo_np)
            axes[i, 0].set_title("Original Photo")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(monet_np)
            axes[i, 1].set_title("Generated Monet")
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


# For Kaggle notebook submission
def kaggle_inference(
    generator_path: str,
    generator_type: str = "unet",
):
    """
    Kaggle-specific inference function.
    
    Designed to run in Kaggle notebook environment.
    Reads from /kaggle/input and writes to current directory.
    """
    import os
    
    # Kaggle paths
    photo_dir = "/kaggle/input/gan-getting-started/photo_jpg"
    output_dir = "/kaggle/working/generated_images"
    zip_path = "/kaggle/working/images.zip"
    
    # Check if we're in Kaggle environment
    if not os.path.exists("/kaggle"):
        print("Not in Kaggle environment. Using local paths.")
        photo_dir = "./gan-getting-started/photo_jpg"
        output_dir = "./generated_images"
        zip_path = "./images.zip"
    
    # Generate submission
    generate_submission(
        generator_path=generator_path,
        photo_dir=photo_dir,
        output_dir=output_dir,
        zip_path=zip_path,
        num_images=7500,
        generator_type=generator_type,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Monet-style images")
    parser.add_argument('--generator', type=str, required=True,
                        help='Path to generator weights')
    parser.add_argument('--photos', type=str, required=True,
                        help='Path to photo directory')
    parser.add_argument('--output', type=str, default='./generated_images',
                        help='Output directory')
    parser.add_argument('--zip', type=str, default='images.zip',
                        help='Output zip file')
    parser.add_argument('--num-images', type=int, default=7500,
                        help='Number of images to generate')
    parser.add_argument('--type', type=str, default='unet',
                        choices=['unet', 'resnet'],
                        help='Generator type')
    args = parser.parse_args()
    
    generate_submission(
        generator_path=args.generator,
        photo_dir=args.photos,
        output_dir=args.output,
        zip_path=args.zip,
        num_images=args.num_images,
        generator_type=args.type,
    )
