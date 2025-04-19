"""Utility functions for creating and manipulating masks for text inpainting."""

from PIL import Image, ImageDraw


def create_text_mask(image, text_box):
    """Create a binary mask for the text region.

    Args:
        image (PIL.Image): Input image
        text_box (tuple): (x1, y1, x2, y2) coordinates for text region

    Returns:
        PIL.Image: Binary mask with white region for text area
    """
    mask = Image.new("RGB", image.size, "black")
    draw = ImageDraw.Draw(mask)
    draw.rectangle(text_box, fill="white")
    return mask


def create_context_mask(image, text_box, padding=10):
    """Create a mask with padding for context awareness.

    Args:
        image (PIL.Image): Input image
        text_box (tuple): (x1, y1, x2, y2) coordinates for text region
        padding (int): Number of pixels to pad around the text region

    Returns:
        PIL.Image: Binary mask with padding
    """
    x1, y1, x2, y2 = text_box
    padded_box = (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image.size[0], x2 + padding),
        min(image.size[1], y2 + padding),
    )
    return create_text_mask(image, padded_box)


def create_antialiased_mask(image, text_box, blur_radius=2):
    """Create an anti-aliased mask for smoother blending.

    Args:
        image (PIL.Image): Input image
        text_box (tuple): (x1, y1, x2, y2) coordinates for text region
        blur_radius (int): Radius for Gaussian blur

    Returns:
        PIL.Image: Anti-aliased mask
    """
    mask = create_text_mask(image, text_box)
    return mask.filter(ImageFilter.GaussianBlur(blur_radius))


def validate_text_box(image_size, text_box):
    """Validate and adjust text box coordinates to fit within image bounds.

    Args:
        image_size (tuple): (width, height) of the image
        text_box (tuple): (x1, y1, x2, y2) coordinates for text region

    Returns:
        tuple: Adjusted text box coordinates
    """
    x1, y1, x2, y2 = text_box
    width, height = image_size

    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )
