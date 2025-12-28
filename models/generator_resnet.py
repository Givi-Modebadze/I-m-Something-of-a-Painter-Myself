"""
ResNet Generator for CycleGAN.

This is the original generator architecture from the CycleGAN paper.
Uses residual blocks instead of skip connections.

Architecture:
    c7s1-64 -> d128 -> d256 -> R256 x 9 -> u128 -> u64 -> c7s1-3
    
Where:
    c7s1-k: 7x7 Conv-InstanceNorm-ReLU with k filters and stride 1
    dk: 3x3 Conv-InstanceNorm-ReLU with k filters and stride 2 (downsample)
    Rk: Residual block with k filters
    uk: 3x3 ConvTranspose-InstanceNorm-ReLU with k filters and stride 2 (upsample)
"""

import torch
import torch.nn as nn
from typing import List


class ResidualBlock(nn.Module):
    """
    Residual Block with two 3x3 convolutions.
    
    Architecture: Conv -> Norm -> ReLU -> Conv -> Norm -> + input
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet-based Generator for image-to-image translation.
    
    The network consists of:
    - Initial 7x7 convolution
    - 2 downsampling layers
    - 9 residual blocks (for 256x256 images)
    - 2 upsampling layers
    - Final 7x7 convolution
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        ngf: Number of filters in first conv layer (default: 64)
        n_residual_blocks: Number of residual blocks (default: 9 for 256x256)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        n_residual_blocks: int = 9,
    ):
        super().__init__()
        
        self.ngf = ngf
        self.n_residual_blocks = n_residual_blocks
        
        # ============== Initial Convolution ==============
        # c7s1-64: 7x7 conv, 64 filters, stride 1
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )
        
        # ============== Downsampling ==============
        # d128, d256
        self.down1 = self._downsample_block(ngf, ngf * 2)      # 64 -> 128
        self.down2 = self._downsample_block(ngf * 2, ngf * 4)  # 128 -> 256
        
        # ============== Residual Blocks ==============
        residual_blocks = []
        for _ in range(n_residual_blocks):
            residual_blocks.append(ResidualBlock(ngf * 4))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # ============== Upsampling ==============
        # u128, u64
        self.up1 = self._upsample_block(ngf * 4, ngf * 2)  # 256 -> 128
        self.up2 = self._upsample_block(ngf * 2, ngf)      # 128 -> 64
        
        # ============== Final Convolution ==============
        # c7s1-3: 7x7 conv, 3 filters, stride 1
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh(),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _downsample_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Downsampling block: Conv (stride 2) -> InstanceNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _upsample_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Upsampling block: ConvTranspose (stride 2) -> InstanceNorm -> ReLU"""
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _init_weights(self, m):
        """Initialize weights with normal distribution."""
        classname = m.__class__.__name__
        if classname == 'Conv2d' or classname == 'ConvTranspose2d':
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
            x: Input image (N, C, H, W)
            
        Returns:
            Output image (N, C, H, W)
        """
        # Initial
        x = self.initial(x)     # (N, 64, 256, 256)
        
        # Downsample
        x = self.down1(x)       # (N, 128, 128, 128)
        x = self.down2(x)       # (N, 256, 64, 64)
        
        # Residual blocks
        x = self.residual_blocks(x)  # (N, 256, 64, 64)
        
        # Upsample
        x = self.up1(x)         # (N, 128, 128, 128)
        x = self.up2(x)         # (N, 64, 256, 256)
        
        # Final
        x = self.final(x)       # (N, 3, 256, 256)
        
        return x


def test_generator():
    """Test the ResNet generator with sample input."""
    print("Testing ResNet Generator...")
    print("-" * 50)
    
    # Create generator
    G = ResNetGenerator(in_channels=3, out_channels=3, ngf=64, n_residual_blocks=9)
    
    # Count parameters
    num_params = sum(p.numel() for p in G.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    output = G(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Verify output
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    assert output.min() >= -1 and output.max() <= 1, "Output should be in [-1, 1]"
    print("\n✅ ResNet Generator test passed!")
    
    return G


def compare_generators():
    """Compare U-Net and ResNet generators."""
    from generator_unet import UNetGeneratorSimple
    
    print("\n" + "=" * 50)
    print("Generator Comparison")
    print("=" * 50)
    
    unet = UNetGeneratorSimple()
    resnet = ResNetGenerator()
    
    unet_params = sum(p.numel() for p in unet.parameters())
    resnet_params = sum(p.numel() for p in resnet.parameters())
    
    print(f"\nU-Net Generator:   {unet_params:,} parameters")
    print(f"ResNet Generator:  {resnet_params:,} parameters")
    print(f"Difference:        {abs(unet_params - resnet_params):,} parameters")
    
    # Test speed (rough comparison)
    import time
    x = torch.randn(1, 3, 256, 256)
    
    # U-Net
    start = time.time()
    for _ in range(10):
        _ = unet(x)
    unet_time = (time.time() - start) / 10
    
    # ResNet
    start = time.time()
    for _ in range(10):
        _ = resnet(x)
    resnet_time = (time.time() - start) / 10
    
    print(f"\nU-Net avg forward pass:   {unet_time*1000:.2f} ms")
    print(f"ResNet avg forward pass:  {resnet_time*1000:.2f} ms")


if __name__ == "__main__":
    test_generator()
    
    # Uncomment to compare both generators
    # compare_generators()
