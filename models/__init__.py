from .discriminator import PatchGANDiscriminator
from .generator_unet import UNetGenerator, UNetGeneratorSimple
from .generator_resnet import ResNetGenerator

__all__ = [
    "PatchGANDiscriminator",
    "UNetGenerator",
    "UNetGeneratorSimple",
    "ResNetGenerator",
]


def get_generator(generator_type: str, **kwargs):
    """
    Factory function to get generator by type.
    
    Args:
        generator_type: "unet" or "resnet"
        **kwargs: Additional arguments for the generator
        
    Returns:
        Generator model
    """
    if generator_type == "unet":
        return UNetGeneratorSimple(**kwargs)
    elif generator_type == "resnet":
        return ResNetGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}. Choose 'unet' or 'resnet'.")


def get_discriminator(**kwargs):
    """
    Factory function to get discriminator.
    
    Args:
        **kwargs: Additional arguments for the discriminator
        
    Returns:
        Discriminator model
    """
    return PatchGANDiscriminator(**kwargs)
