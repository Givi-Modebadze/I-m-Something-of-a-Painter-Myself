"""
CycleGAN Training Script.

This script handles the complete training pipeline:
1. Initialize models, optimizers, losses
2. Training loop with alternating G and D updates
3. Logging to WandB
4. Checkpointing
5. Sample visualization

Usage:
    python train.py                     # Train with default config
    python train.py --generator unet    # Train with U-Net generator
    python train.py --generator resnet  # Train with ResNet generator
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import our modules
from configs.config import Config, get_colab_config, get_debug_config
from data.dataset import get_dataloaders, tensor_to_image
from models import get_generator, get_discriminator
from losses.losses import CycleGANLoss
from utils.utils import (
    ReplayBuffer,
    save_checkpoint,
    load_checkpoint,
    save_generator_only,
    visualize_results,
    get_lr_scheduler,
    set_requires_grad,
    print_model_summary,
    save_image_grid,
)


class CycleGANTrainer:
    """
    CycleGAN Trainer class.
    
    Handles model initialization, training loop, logging, and checkpointing.
    """
    
    def __init__(self, config: Config):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize loss functions
        self._init_losses()
        
        # Initialize replay buffers
        self.buffer_A = ReplayBuffer(config.buffer_size)
        self.buffer_B = ReplayBuffer(config.buffer_size)
        
        # Initialize WandB
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def _init_models(self) -> None:
        """Initialize generator and discriminator models."""
        config = self.config
        
        # Generators
        # G_AB: A -> B (Photo -> Monet)
        # G_BA: B -> A (Monet -> Photo)
        self.G_AB = get_generator(
            config.generator_type,
            in_channels=config.img_channels,
            out_channels=config.img_channels,
        ).to(self.device)
        
        self.G_BA = get_generator(
            config.generator_type,
            in_channels=config.img_channels,
            out_channels=config.img_channels,
        ).to(self.device)
        
        # Discriminators
        # D_A: Discriminates real/fake in domain A (Photo)
        # D_B: Discriminates real/fake in domain B (Monet)
        self.D_A = get_discriminator(
            in_channels=config.img_channels,
            ndf=config.ndf,
            n_layers=config.n_layers_D,
        ).to(self.device)
        
        self.D_B = get_discriminator(
            in_channels=config.img_channels,
            ndf=config.ndf,
            n_layers=config.n_layers_D,
        ).to(self.device)
        
        # Print model summary
        print_model_summary(self.G_AB, self.G_BA, self.D_A, self.D_B)
    
    def _init_optimizers(self) -> None:
        """Initialize optimizers and LR schedulers."""
        config = self.config
        
        # Combined generator optimizer
        self.optimizer_G = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        
        # Combined discriminator optimizer
        self.optimizer_D = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        
        # Learning rate schedulers
        self.scheduler_G = get_lr_scheduler(
            self.optimizer_G,
            n_epochs=config.epochs,
            decay_start_epoch=config.lr_decay_start,
            decay_epochs=config.lr_decay_epochs,
        )
        
        self.scheduler_D = get_lr_scheduler(
            self.optimizer_D,
            n_epochs=config.epochs,
            decay_start_epoch=config.lr_decay_start,
            decay_epochs=config.lr_decay_epochs,
        )
    
    def _init_losses(self) -> None:
        """Initialize loss functions."""
        config = self.config
        
        self.criterion = CycleGANLoss(
            lambda_cycle=config.lambda_cycle,
            lambda_identity=config.lambda_identity,
            adversarial_loss_type=config.adversarial_loss_type,
        )
    
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.experiment_name,
                config=self.config.get_experiment_config(),
            )
            print(f"WandB initialized: {wandb.run.name}")
        except ImportError:
            print("WandB not installed. Logging disabled.")
            self.config.use_wandb = False
        except Exception as e:
            print(f"WandB initialization failed: {e}")
            self.config.use_wandb = False
    
    def train_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            real_A: Real images from domain A (Photos)
            real_B: Real images from domain B (Monet)
            
        Returns:
            Dictionary of loss values
        """
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        
        # ==================== Train Generators ====================
        set_requires_grad([self.D_A, self.D_B], False)
        self.optimizer_G.zero_grad()
        
        # Generate fake images
        fake_B = self.G_AB(real_A)  # Photo -> Monet
        fake_A = self.G_BA(real_B)  # Monet -> Photo
        
        # Cycle reconstruction
        reconstructed_A = self.G_BA(fake_B)  # Photo -> Monet -> Photo
        reconstructed_B = self.G_AB(fake_A)  # Monet -> Photo -> Monet
        
        # Identity mapping (optional but helps preserve color)
        identity_A = self.G_BA(real_A)  # Photo -> Photo (should be identity)
        identity_B = self.G_AB(real_B)  # Monet -> Monet (should be identity)
        
        # Discriminator outputs for fake images
        D_A_fake = self.D_A(fake_A)
        D_B_fake = self.D_B(fake_B)
        
        # Calculate generator loss
        loss_G, loss_G_dict = self.criterion.generator_loss(
            D_A_fake=D_A_fake,
            D_B_fake=D_B_fake,
            real_A=real_A,
            real_B=real_B,
            reconstructed_A=reconstructed_A,
            reconstructed_B=reconstructed_B,
            identity_A=identity_A,
            identity_B=identity_B,
        )
        
        # Backprop generators
        loss_G.backward()
        self.optimizer_G.step()
        
        # ==================== Train Discriminators ====================
        set_requires_grad([self.D_A, self.D_B], True)
        self.optimizer_D.zero_grad()
        
        # Use replay buffer for fake images
        fake_A_buffer = self.buffer_A.push_and_pop(fake_A.detach())
        fake_B_buffer = self.buffer_B.push_and_pop(fake_B.detach())
        
        # Discriminator A (distinguishes real/fake Photos)
        D_A_real = self.D_A(real_A)
        D_A_fake = self.D_A(fake_A_buffer)
        loss_D_A, loss_D_A_dict = self.criterion.discriminator_loss(D_A_real, D_A_fake)
        
        # Discriminator B (distinguishes real/fake Monets)
        D_B_real = self.D_B(real_B)
        D_B_fake = self.D_B(fake_B_buffer)
        loss_D_B, loss_D_B_dict = self.criterion.discriminator_loss(D_B_real, D_B_fake)
        
        # Total discriminator loss
        loss_D = loss_D_A + loss_D_B
        
        # Backprop discriminators
        loss_D.backward()
        self.optimizer_D.step()
        
        # Combine all losses for logging
        losses = {
            **loss_G_dict,
            'D_A_total': loss_D_A_dict['D_total'],
            'D_B_total': loss_D_B_dict['D_total'],
            'D_total': loss_D.item(),
        }
        
        return losses, {
            'real_A': real_A,
            'real_B': real_B,
            'fake_A': fake_A.detach(),
            'fake_B': fake_B.detach(),
            'reconstructed_A': reconstructed_A.detach(),
            'reconstructed_B': reconstructed_B.detach(),
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
            
        Returns:
            Average losses for the epoch
        """
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()
        
        epoch_losses = {}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.config.epochs}")
        
        for i, batch in enumerate(pbar):
            real_A = batch['A']
            real_B = batch['B']
            
            # Training step
            losses, images = self.train_step(real_A, real_B)
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{losses['G_total']:.3f}",
                'D': f"{losses['D_total']:.3f}",
            })
            
            # Log to WandB
            if self.config.use_wandb and self.global_step % self.config.log_freq == 0:
                import wandb
                wandb.log({
                    **{f"train/{k}": v for k, v in losses.items()},
                    'train/lr': self.optimizer_G.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': self.global_step,
                })
            
            # Save sample images
            if self.global_step % self.config.sample_freq == 0:
                self._save_samples(images, epoch, self.global_step)
            
            self.global_step += 1
        
        # Average losses
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _save_samples(self, images: Dict[str, torch.Tensor], epoch: int, step: int) -> None:
        """Save sample images during training."""
        sample_path = os.path.join(
            self.config.sample_dir,
            f"epoch_{epoch}_step_{step}.png"
        )
        
        visualize_results(
            real_A=images['real_A'],
            fake_B=images['fake_B'],
            reconstructed_A=images['reconstructed_A'],
            real_B=images['real_B'],
            fake_A=images['fake_A'],
            reconstructed_B=images['reconstructed_B'],
            save_path=sample_path,
            show=False,
        )
        
        # Log to WandB
        if self.config.use_wandb:
            import wandb
            wandb.log({
                "samples": wandb.Image(sample_path),
                "step": step,
            })
    
    def train(self, dataloader: DataLoader, resume_from: Optional[str] = None) -> None:
        """
        Full training loop.
        
        Args:
            dataloader: Training dataloader
            resume_from: Optional path to checkpoint to resume from
        """
        # Resume from checkpoint if provided
        if resume_from is not None:
            self.current_epoch = load_checkpoint(
                resume_from,
                self.G_AB, self.G_BA,
                self.D_A, self.D_B,
                self.optimizer_G, self.optimizer_D,
                self.scheduler_G, self.scheduler_D,
                device=self.config.device,
            )
            self.current_epoch += 1  # Start from next epoch
        
        print(f"\nStarting training from epoch {self.current_epoch}")
        print(f"Total epochs: {self.config.epochs}")
        print(f"Device: {self.device}")
        print(f"Generator type: {self.config.generator_type}")
        print("-" * 50)
        
        for epoch in range(self.current_epoch, self.config.epochs):
            start_time = time.time()
            
            # Train one epoch
            epoch_losses = self.train_epoch(dataloader, epoch)
            
            # Update learning rate
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
            print(f"  G_total: {epoch_losses['G_total']:.4f}")
            print(f"  G_adv: {epoch_losses['G_adv']:.4f}")
            print(f"  G_cycle: {epoch_losses['G_cycle']:.4f}")
            print(f"  G_identity: {epoch_losses['G_identity']:.4f}")
            print(f"  D_total: {epoch_losses['D_total']:.4f}")
            print(f"  LR: {self.optimizer_G.param_groups[0]['lr']:.6f}")
            
            # Log epoch metrics to WandB
            if self.config.use_wandb:
                import wandb
                wandb.log({
                    **{f"epoch/{k}": v for k, v in epoch_losses.items()},
                    'epoch/lr': self.optimizer_G.param_groups[0]['lr'],
                    'epoch': epoch,
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_freq == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_epoch_{epoch}.pth"
                )
                save_checkpoint(
                    epoch,
                    self.G_AB, self.G_BA,
                    self.D_A, self.D_B,
                    self.optimizer_G, self.optimizer_D,
                    checkpoint_path,
                    self.scheduler_G, self.scheduler_D,
                )
                
                # Also save generator only for submission
                generator_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"generator_AB_epoch_{epoch}.pth"
                )
                save_generator_only(self.G_AB, generator_path)
        
        # Save final models
        print("\nTraining completed!")
        final_checkpoint = os.path.join(self.config.checkpoint_dir, "final_checkpoint.pth")
        save_checkpoint(
            self.config.epochs - 1,
            self.G_AB, self.G_BA,
            self.D_A, self.D_B,
            self.optimizer_G, self.optimizer_D,
            final_checkpoint,
        )
        
        final_generator = os.path.join(self.config.checkpoint_dir, "generator_AB_final.pth")
        save_generator_only(self.G_AB, final_generator)
        
        # Close WandB
        if self.config.use_wandb:
            import wandb
            wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train CycleGAN")
    parser.add_argument('--generator', type=str, default='unet', choices=['unet', 'resnet'],
                        help='Generator type (default: unet)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Use debug config (fewer epochs, more logging)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')
    args = parser.parse_args()
    
    # Get config
    if args.debug:
        config = get_debug_config()
    else:
        config = get_colab_config()
    
    # Override with command line args
    config.generator_type = args.generator
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.no_wandb:
        config.use_wandb = False
    
    config.experiment_name = f"{config.generator_type}_cyclegan"
    
    # Create dataloaders
    train_loader, _ = get_dataloaders(
        monet_path=config.monet_path,
        photo_path=config.photo_path,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    
    # Create trainer and train
    trainer = CycleGANTrainer(config)
    trainer.train(train_loader, resume_from=args.resume)


if __name__ == "__main__":
    main()
