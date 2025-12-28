"""
Configuration file for CycleGAN training.
All hyperparameters and paths are defined here for easy experimentation.
"""

import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path


@dataclass
class Config:
    """Main configuration class for CycleGAN training."""
    
    # ============== Paths ==============
    # Update these paths based on your environment
    # For Colab: /content/drive/MyDrive/...
    # For Kaggle: /kaggle/input/gan-getting-started/...
    data_root: str = "/kaggle/input/gan-getting-started"
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    sample_dir: str = "./samples"
    
    # ============== Data ==============
    monet_dir: str = "monet_jpg"  # Subdirectory name
    photo_dir: str = "photo_jpg"  # Subdirectory name
    img_size: int = 256
    img_channels: int = 3
    
    # ============== Model ==============
    # Generator type: "unet" or "resnet"
    generator_type: str = "unet"
    
    # U-Net specific
    unet_features: Tuple[int, ...] = (64, 128, 256, 512, 512, 512, 512, 512)
    
    # ResNet specific
    resnet_blocks: int = 9  # 6 for 128x128, 9 for 256x256
    ngf: int = 64  # Number of generator filters
    
    # Discriminator
    ndf: int = 64  # Number of discriminator filters
    n_layers_D: int = 3  # Number of layers in discriminator
    
    # ============== Training ==============
    epochs: int = 30
    batch_size: int = 1  # CycleGAN typically uses batch_size=1
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Learning rate scheduler
    lr_decay_start: int = 15  # Start decaying after this epoch
    lr_decay_epochs: int = 15  # Decay over this many epochs
    
    # ============== Loss Weights ==============
    lambda_cycle: float = 10.0  # Cycle consistency loss weight
    lambda_identity: float = 0.5  # Identity loss weight (relative to cycle)
    # Actual identity weight = lambda_identity * lambda_cycle = 5.0
    
    # Adversarial loss type: "lsgan", "vanilla", "hinge"
    adversarial_loss_type: str = "lsgan"
    
    # ============== Replay Buffer ==============
    use_replay_buffer: bool = True
    buffer_size: int = 50
    
    # ============== Logging ==============
    use_wandb: bool = True
    wandb_project: str = "cyclegan-monet"
    wandb_entity: Optional[str] = None  # Your WandB username
    experiment_name: str = "unet_baseline"
    
    # Logging frequency
    log_freq: int = 100  # Log losses every N iterations
    sample_freq: int = 500  # Save sample images every N iterations
    checkpoint_freq: int = 5  # Save checkpoint every N epochs
    
    # ============== Hardware ==============
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    pin_memory: bool = True
    
    # ============== Reproducibility ==============
    seed: int = 42
    
    # ============== Inference ==============
    num_images_to_generate: int = 7500  # For Kaggle submission (7000-10000)
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.sample_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def monet_path(self) -> str:
        return str(Path(self.data_root) / self.monet_dir)
    
    @property
    def photo_path(self) -> str:
        return str(Path(self.data_root) / self.photo_dir)
    
    def get_experiment_config(self) -> dict:
        """Return config as dict for WandB logging."""
        return {
            "generator_type": self.generator_type,
            "adversarial_loss": self.adversarial_loss_type,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "lambda_cycle": self.lambda_cycle,
            "lambda_identity": self.lambda_identity,
            "img_size": self.img_size,
            "ngf": self.ngf,
            "ndf": self.ndf,
        }


# Preset configurations for experiments
def get_unet_config() -> Config:
    """Configuration for U-Net generator experiment."""
    return Config(
        generator_type="unet",
        experiment_name="unet_generator",
    )


def get_resnet_config() -> Config:
    """Configuration for ResNet generator experiment."""
    return Config(
        generator_type="resnet",
        experiment_name="resnet_generator",
    )


def get_debug_config() -> Config:
    """Configuration for quick debugging runs."""
    return Config(
        epochs=2,
        log_freq=10,
        sample_freq=50,
        checkpoint_freq=1,
        experiment_name="debug_run",
        use_wandb=False,
    )


# For Colab environment
def get_colab_config() -> Config:
    """Configuration optimized for Google Colab."""
    return Config(
        data_root="/content/gan-getting-started",
        checkpoint_dir="/content/drive/MyDrive/cyclegan_checkpoints",
        output_dir="/content/outputs",
        sample_dir="/content/samples",
        num_workers=2,
    )
