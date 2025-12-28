"""
Loss functions for CycleGAN.

CycleGAN uses three types of losses:
1. Adversarial Loss - Makes generated images indistinguishable from real
2. Cycle Consistency Loss - Ensures F(G(x)) ≈ x and G(F(y)) ≈ y
3. Identity Loss - Helps preserve color composition (optional but recommended)

Total Generator Loss = Adversarial + λ_cycle * Cycle + λ_identity * Identity
"""

import torch
import torch.nn as nn
from typing import Tuple


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training.
    
    Supports multiple GAN loss types:
    - 'lsgan': Least Squares GAN (MSE loss) - recommended, more stable
    - 'vanilla': Original GAN (BCE loss)
    - 'hinge': Hinge loss
    
    LSGAN is recommended for CycleGAN as it provides more stable training
    and produces higher quality results.
    """
    
    def __init__(self, loss_type: str = 'lsgan'):
        super().__init__()
        
        self.loss_type = loss_type
        
        if loss_type == 'lsgan':
            # Least Squares GAN: min (D(x) - 1)^2 + (D(G(z)))^2
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'vanilla':
            # Original GAN: -log(D(x)) - log(1 - D(G(z)))
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == 'hinge':
            # Hinge loss: handled separately in forward
            self.loss_fn = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose 'lsgan', 'vanilla', or 'hinge'.")
    
    def forward(
        self,
        pred: torch.Tensor,
        is_real: bool,
        is_discriminator: bool = True,
    ) -> torch.Tensor:
        """
        Calculate adversarial loss.
        
        Args:
            pred: Discriminator prediction
            is_real: True if this is for real images, False for fake
            is_discriminator: True if training discriminator, False for generator
            
        Returns:
            Loss value
        """
        if self.loss_type == 'lsgan':
            if is_real:
                target = torch.ones_like(pred)
            else:
                target = torch.zeros_like(pred)
            return self.loss_fn(pred, target)
        
        elif self.loss_type == 'vanilla':
            if is_real:
                target = torch.ones_like(pred)
            else:
                target = torch.zeros_like(pred)
            return self.loss_fn(pred, target)
        
        elif self.loss_type == 'hinge':
            if is_discriminator:
                if is_real:
                    # D wants to maximize: min(0, -1 + D(x))
                    return torch.mean(torch.relu(1.0 - pred))
                else:
                    # D wants to minimize fake: min(0, -1 - D(G(z)))
                    return torch.mean(torch.relu(1.0 + pred))
            else:
                # Generator wants to maximize D(G(z))
                return -torch.mean(pred)


class CycleLoss(nn.Module):
    """
    Cycle Consistency Loss.
    
    Ensures that translating an image to another domain and back
    reconstructs the original image:
        - Forward cycle: x -> G(x) -> F(G(x)) ≈ x
        - Backward cycle: y -> F(y) -> G(F(y)) ≈ y
    
    Uses L1 loss for sharper reconstructions (L2 tends to be blurry).
    """
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()
    
    def forward(
        self,
        real: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate cycle consistency loss.
        
        Args:
            real: Original image
            reconstructed: Image after full cycle (domain A -> B -> A)
            
        Returns:
            L1 loss between real and reconstructed
        """
        return self.loss_fn(real, reconstructed)


class IdentityLoss(nn.Module):
    """
    Identity Loss.
    
    Encourages the generator to be near an identity mapping when
    real samples of the target domain are provided:
        - G(y) ≈ y (G should not change images already in target domain)
        - F(x) ≈ x (F should not change images already in source domain)
    
    This helps preserve color composition and prevents unnecessary changes.
    Particularly useful for painting/photo style transfer.
    """
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()
    
    def forward(
        self,
        real: torch.Tensor,
        same: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate identity loss.
        
        Args:
            real: Real image from target domain
            same: Output of generator when given target domain image
                  (should be unchanged)
            
        Returns:
            L1 loss between real and same
        """
        return self.loss_fn(real, same)


class CycleGANLoss(nn.Module):
    """
    Combined loss for CycleGAN training.
    
    Combines all losses with configurable weights:
        Total_G = Adv_G + λ_cycle * (Cycle_A + Cycle_B) + λ_identity * (Id_A + Id_B)
        Total_D = Adv_D_A + Adv_D_B
    """
    
    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        adversarial_loss_type: str = 'lsgan',
    ):
        """
        Args:
            lambda_cycle: Weight for cycle consistency loss (default: 10.0)
            lambda_identity: Weight for identity loss relative to cycle (default: 0.5)
                            Actual weight = lambda_identity * lambda_cycle = 5.0
            adversarial_loss_type: Type of adversarial loss ('lsgan', 'vanilla', 'hinge')
        """
        super().__init__()
        
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        self.adversarial = AdversarialLoss(adversarial_loss_type)
        self.cycle = CycleLoss()
        self.identity = IdentityLoss()
    
    def generator_loss(
        self,
        # Discriminator outputs for fake images
        D_A_fake: torch.Tensor,  # D_A(G_BA(B))
        D_B_fake: torch.Tensor,  # D_B(G_AB(A))
        # Cycle reconstructions
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        reconstructed_A: torch.Tensor,  # G_BA(G_AB(A))
        reconstructed_B: torch.Tensor,  # G_AB(G_BA(B))
        # Identity outputs (optional)
        identity_A: torch.Tensor = None,  # G_BA(A) - should be A
        identity_B: torch.Tensor = None,  # G_AB(B) - should be B
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate total generator loss.
        
        Returns:
            total_loss: Combined loss for backpropagation
            loss_dict: Dictionary with individual loss values for logging
        """
        # Adversarial losses - generators want discriminators to think fakes are real
        loss_adv_A = self.adversarial(D_A_fake, is_real=True, is_discriminator=False)
        loss_adv_B = self.adversarial(D_B_fake, is_real=True, is_discriminator=False)
        loss_adv = loss_adv_A + loss_adv_B
        
        # Cycle consistency losses
        loss_cycle_A = self.cycle(real_A, reconstructed_A)
        loss_cycle_B = self.cycle(real_B, reconstructed_B)
        loss_cycle = loss_cycle_A + loss_cycle_B
        
        # Identity losses (if provided)
        loss_identity = torch.tensor(0.0, device=real_A.device)
        if identity_A is not None and identity_B is not None:
            loss_id_A = self.identity(real_A, identity_A)
            loss_id_B = self.identity(real_B, identity_B)
            loss_identity = loss_id_A + loss_id_B
        
        # Total generator loss
        total_loss = (
            loss_adv +
            self.lambda_cycle * loss_cycle +
            self.lambda_cycle * self.lambda_identity * loss_identity
        )
        
        loss_dict = {
            'G_adv': loss_adv.item(),
            'G_adv_A': loss_adv_A.item(),
            'G_adv_B': loss_adv_B.item(),
            'G_cycle': loss_cycle.item(),
            'G_cycle_A': loss_cycle_A.item(),
            'G_cycle_B': loss_cycle_B.item(),
            'G_identity': loss_identity.item(),
            'G_total': total_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def discriminator_loss(
        self,
        D_real: torch.Tensor,
        D_fake: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate discriminator loss for one discriminator.
        
        Args:
            D_real: Discriminator output for real images
            D_fake: Discriminator output for fake images (detached)
            
        Returns:
            total_loss: Combined loss for backpropagation
            loss_dict: Dictionary with individual loss values
        """
        loss_real = self.adversarial(D_real, is_real=True, is_discriminator=True)
        loss_fake = self.adversarial(D_fake, is_real=False, is_discriminator=True)
        
        # Average of real and fake losses
        total_loss = (loss_real + loss_fake) * 0.5
        
        loss_dict = {
            'D_real': loss_real.item(),
            'D_fake': loss_fake.item(),
            'D_total': total_loss.item(),
        }
        
        return total_loss, loss_dict


def test_losses():
    """Test all loss functions."""
    print("Testing Loss Functions...")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy tensors
    real_A = torch.randn(1, 3, 256, 256, device=device)
    real_B = torch.randn(1, 3, 256, 256, device=device)
    fake_A = torch.randn(1, 3, 256, 256, device=device)
    fake_B = torch.randn(1, 3, 256, 256, device=device)
    
    # Discriminator outputs (PatchGAN: 30x30)
    D_A_real = torch.randn(1, 1, 30, 30, device=device)
    D_A_fake = torch.randn(1, 1, 30, 30, device=device)
    D_B_real = torch.randn(1, 1, 30, 30, device=device)
    D_B_fake = torch.randn(1, 1, 30, 30, device=device)
    
    # Test individual losses
    print("\n1. Testing Adversarial Loss (LSGAN):")
    adv_loss = AdversarialLoss('lsgan')
    loss_real = adv_loss(D_A_real, is_real=True)
    loss_fake = adv_loss(D_A_fake, is_real=False)
    print(f"   Real loss: {loss_real.item():.4f}")
    print(f"   Fake loss: {loss_fake.item():.4f}")
    
    print("\n2. Testing Cycle Loss:")
    cycle_loss = CycleLoss()
    loss = cycle_loss(real_A, fake_A)  # Should be high (different images)
    print(f"   Cycle loss (different): {loss.item():.4f}")
    loss = cycle_loss(real_A, real_A)  # Should be 0 (same image)
    print(f"   Cycle loss (same): {loss.item():.4f}")
    
    print("\n3. Testing Identity Loss:")
    id_loss = IdentityLoss()
    loss = id_loss(real_A, fake_A)
    print(f"   Identity loss: {loss.item():.4f}")
    
    print("\n4. Testing Combined CycleGAN Loss:")
    criterion = CycleGANLoss(lambda_cycle=10.0, lambda_identity=0.5)
    
    # Generator loss
    g_loss, g_dict = criterion.generator_loss(
        D_A_fake=D_A_fake,
        D_B_fake=D_B_fake,
        real_A=real_A,
        real_B=real_B,
        reconstructed_A=fake_A,
        reconstructed_B=fake_B,
        identity_A=fake_A,
        identity_B=fake_B,
    )
    print(f"   Generator total loss: {g_dict['G_total']:.4f}")
    print(f"   - Adversarial: {g_dict['G_adv']:.4f}")
    print(f"   - Cycle: {g_dict['G_cycle']:.4f}")
    print(f"   - Identity: {g_dict['G_identity']:.4f}")
    
    # Discriminator loss
    d_loss, d_dict = criterion.discriminator_loss(D_A_real, D_A_fake)
    print(f"\n   Discriminator loss: {d_dict['D_total']:.4f}")
    print(f"   - Real: {d_dict['D_real']:.4f}")
    print(f"   - Fake: {d_dict['D_fake']:.4f}")
    
    print("\n✅ All loss functions working!")


if __name__ == "__main__":
    test_losses()
