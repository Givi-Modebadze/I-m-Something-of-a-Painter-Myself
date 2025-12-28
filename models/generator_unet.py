"""
U-Net Generator for CycleGAN.

U-Net architecture uses skip connections between encoder and decoder,
which helps preserve spatial information and produces sharper outputs.

Architecture:
    Encoder: C64 -> C128 -> C256 -> C512 -> C512 -> C512 -> C512 -> C512
    Decoder: CD512 -> CD512 -> CD512 -> CD512 -> CD256 -> CD128 -> CD64 -> output
    
Where:
    Ck = Conv-InstanceNorm-LeakyReLU with k filters
    CDk = ConvTranspose-InstanceNorm-Dropout-ReLU with k filters
    
Skip connections connect encoder layer i to decoder layer n-i.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class UNetEncoderBlock(nn.Module):
    """
    U-Net Encoder block: Conv -> InstanceNorm -> LeakyReLU
    
    Downsamples by factor of 2.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool = True,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_norm,
            )
        ]
        
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    """
    U-Net Decoder block: ConvTranspose -> InstanceNorm -> Dropout (optional) -> ReLU
    
    Upsamples by factor of 2.
    Input is concatenated with skip connection from encoder.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dropout: bool = False,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from corresponding encoder layer
        """
        x = self.block(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        return x


class UNetGenerator(nn.Module):
    """
    U-Net Generator for image-to-image translation.
    
    The network consists of:
    - 8 encoder blocks (downsampling)
    - 8 decoder blocks (upsampling) with skip connections
    
    For 256x256 input:
        256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 (bottleneck)
        1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        features: List of features for each encoder layer
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: Optional[List[int]] = None,
    ):
        super().__init__()
        
        if features is None:
            # Default U-Net features for 256x256 images
            features = [64, 128, 256, 512, 512, 512, 512, 512]
        
        self.features = features
        
        # ============== Encoder ==============
        self.encoders = nn.ModuleList()
        
        # First encoder: no normalization
        self.encoders.append(
            UNetEncoderBlock(in_channels, features[0], use_norm=False)
        )
        
        # Remaining encoders
        for i in range(1, len(features)):
            self.encoders.append(
                UNetEncoderBlock(features[i-1], features[i], use_norm=True)
            )
        
        # ============== Decoder ==============
        self.decoders = nn.ModuleList()
        
        # First 3 decoders use dropout (for 512 -> 512 layers)
        # Decoder input channels = features[i] (from previous) 
        # After concat with skip = features[i] + features[n-1-i]
        
        reversed_features = features[::-1]  # [512, 512, 512, 512, 256, 128, 64]
        
        for i in range(len(features) - 1):
            # Determine input channels
            if i == 0:
                in_ch = reversed_features[i]  # First decoder: no skip concat yet
            else:
                in_ch = reversed_features[i-1] + reversed_features[i]  # Previous output + skip
            
            out_ch = reversed_features[i+1] if i < len(features) - 2 else reversed_features[-1]
            
            # Use dropout for first 3 decoder layers
            use_dropout = i < 3
            
            self.decoders.append(
                UNetDecoderBlock(in_ch, out_ch, use_dropout=use_dropout)
            )
        
        # ============== Final Layer ==============
        # Takes concatenated features and outputs RGB image
        # Input: last decoder output (64) + first encoder skip (64) = 128
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                features[0] * 2,  # 64 + 64 = 128
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
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
        Forward pass with skip connections.
        
        Args:
            x: Input image (N, C, H, W)
            
        Returns:
            Output image (N, C, H, W)
        """
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder forward pass
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
        
        # Remove the bottleneck from skip connections
        # (we don't skip connect the bottleneck to itself)
        skips = encoder_outputs[:-1][::-1]  # Reverse order, exclude last
        
        # Decoder forward pass
        x = encoder_outputs[-1]  # Start from bottleneck
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[i])
        
        # Final layer
        x = self.final(x)
        
        return x


class UNetGeneratorSimple(nn.Module):
    """
    Simplified U-Net Generator - cleaner implementation.
    
    This version is more explicit about the architecture.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._encoder_block(in_channels, 64, normalize=False)  # 256 -> 128
        self.enc2 = self._encoder_block(64, 128)    # 128 -> 64
        self.enc3 = self._encoder_block(128, 256)   # 64 -> 32
        self.enc4 = self._encoder_block(256, 512)   # 32 -> 16
        self.enc5 = self._encoder_block(512, 512)   # 16 -> 8
        self.enc6 = self._encoder_block(512, 512)   # 8 -> 4
        self.enc7 = self._encoder_block(512, 512)   # 4 -> 2
        self.enc8 = self._encoder_block(512, 512, normalize=False)   # 2 -> 1 (bottleneck)
        
        # Decoder (upsampling with skip connections)
        self.dec1 = self._decoder_block(512, 512, dropout=True)     # 1 -> 2
        self.dec2 = self._decoder_block(1024, 512, dropout=True)    # 2 -> 4
        self.dec3 = self._decoder_block(1024, 512, dropout=True)    # 4 -> 8
        self.dec4 = self._decoder_block(1024, 512)                   # 8 -> 16
        self.dec5 = self._decoder_block(1024, 256)                   # 16 -> 32
        self.dec6 = self._decoder_block(512, 128)                    # 32 -> 64
        self.dec7 = self._decoder_block(256, 64)                     # 64 -> 128
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
    
    def _encoder_block(self, in_ch, out_ch, normalize=True):
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _decoder_block(self, in_ch, out_ch, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
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
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # 64 x 128 x 128
        e2 = self.enc2(e1)   # 128 x 64 x 64
        e3 = self.enc3(e2)   # 256 x 32 x 32
        e4 = self.enc4(e3)   # 512 x 16 x 16
        e5 = self.enc5(e4)   # 512 x 8 x 8
        e6 = self.enc6(e5)   # 512 x 4 x 4
        e7 = self.enc7(e6)   # 512 x 2 x 2
        e8 = self.enc8(e7)   # 512 x 1 x 1 (bottleneck)
        
        # Decoder with skip connections
        d1 = self.dec1(e8)                      # 512 x 2 x 2
        d2 = self.dec2(torch.cat([d1, e7], 1))  # 512 x 4 x 4
        d3 = self.dec3(torch.cat([d2, e6], 1))  # 512 x 8 x 8
        d4 = self.dec4(torch.cat([d3, e5], 1))  # 512 x 16 x 16
        d5 = self.dec5(torch.cat([d4, e4], 1))  # 256 x 32 x 32
        d6 = self.dec6(torch.cat([d5, e3], 1))  # 128 x 64 x 64
        d7 = self.dec7(torch.cat([d6, e2], 1))  # 64 x 128 x 128
        
        # Final with skip from e1
        out = self.final(torch.cat([d7, e1], 1))  # 3 x 256 x 256
        
        return out


def test_generator():
    """Test the U-Net generator with sample input."""
    print("Testing U-Net Generator...")
    print("-" * 50)
    
    # Test simple version (recommended)
    G = UNetGeneratorSimple(in_channels=3, out_channels=3)
    
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
    print("\n✅ U-Net Generator test passed!")
    
    return G


if __name__ == "__main__":
    test_generator()
