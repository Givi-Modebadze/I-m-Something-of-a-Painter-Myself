"""
PatchGAN Discriminator for CycleGAN.

Instead of classifying the entire image as real/fake, PatchGAN classifies
NxN patches of the image. This encourages the generator to produce
high-frequency details and textures.

The 70x70 PatchGAN is the default - it looks at 70x70 receptive field patches.
"""

import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv -> InstanceNorm -> LeakyReLU
    
    First layer doesn't use normalization (as per original CycleGAN).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_norm: bool = True,
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_norm,  # No bias when using normalization
            )
        ]
        
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator (70x70 receptive field).
    
    Architecture:
        C64 -> C128 -> C256 -> C512 -> output
        
    Where Ck = Conv-InstanceNorm-LeakyReLU with k filters.
    First C64 doesn't use InstanceNorm.
    
    Input: (N, 3, 256, 256)
    Output: (N, 1, 30, 30) - each value represents real/fake for a patch
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        ndf: Number of filters in first conv layer (default: 64)
        n_layers: Number of conv layers (default: 3 for 70x70 PatchGAN)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.ndf = ndf
        self.n_layers = n_layers
        
        # Build discriminator layers
        layers = []
        
        # First layer: no normalization
        layers.append(
            ConvBlock(in_channels, ndf, kernel_size=4, stride=2, padding=1, use_norm=False)
        )
        
        # Middle layers: stride=2 downsampling with normalization
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 512 filters
            layers.append(
                ConvBlock(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    use_norm=True,
                )
            )
        
        # Second-to-last layer: stride=1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(
            ConvBlock(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                use_norm=True,
            )
        )
        
        # Final layer: 1-channel output (no norm, no activation)
        layers.append(
            nn.Conv2d(
                ndf * nf_mult,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights with normal distribution."""
        classname = m.__class__.__name__
        if classname == 'Conv2d':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname == 'InstanceNorm2d':
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (N, C, H, W)
            
        Returns:
            Patch predictions (N, 1, H', W')
        """
        return self.model(x)


def test_discriminator():
    """Test the discriminator with sample input."""
    print("Testing PatchGAN Discriminator...")
    print("-" * 50)
    
    # Create discriminator
    D = PatchGANDiscriminator(in_channels=3, ndf=64, n_layers=3)
    
    # Count parameters
    num_params = sum(p.numel() for p in D.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    output = D(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [1, 1, 30, 30]
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Verify output size
    assert output.shape == (1, 1, 30, 30), f"Expected (1, 1, 30, 30), got {output.shape}"
    print("\n✅ Discriminator test passed!")
    
    return D


if __name__ == "__main__":
    test_discriminator()
