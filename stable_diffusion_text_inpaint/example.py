"""Example usage of the TextInpainter class.

Note: On first run, this script will download required models from Hugging Face:
1. stabilityai/stable-diffusion-2-inpainting (~4GB): Used for text inpainting
2. microsoft/trocr-large-printed (~1GB): Used for text style analysis

The style analysis system helps match the appearance of existing text by:
- Analyzing text color by finding dominant colors in the text region
- Detecting background color by sampling corner pixels
- Determining text size based on the region dimensions
- Converting these properties into natural language prompts for Stable Diffusion
  (e.g. "clear black text on white background")

This helps ensure that newly inpainted text matches the style of text in the rest
of the image, maintaining visual consistency.
"""

from PIL import Image, ImageDraw
from text_inpainter import TextInpainter
import os


def create_sample_image(size=(512, 512), color="white"):
    """Create a sample image for testing.

    Args:
        size (tuple): Image dimensions (width, height)
        color (str): Background color

    Returns:
        PIL.Image: Sample image
    """
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)

    # Add some shapes for visual reference
    draw.rectangle((50, 50, 462, 462), outline="gray", width=2)
    draw.line((50, 256, 462, 256), fill="gray", width=2)
    draw.line((256, 50, 256, 462), fill="gray", width=2)

    # Save the image
    if not os.path.exists("example.png"):
        image.save("example.png")
    return image


def single_text_example():
    """Example of inpainting a single text region."""
    # Initialize inpainter
    inpainter = TextInpainter()

    # Create or load sample image
    image = create_sample_image()

    # Define text region (centered horizontally)
    text_box = (156, 100, 356, 150)  # 200px wide, centered in 512px image

    # Simple inpainting
    result = inpainter.inpaint_text(image=image, text="Hello World", text_box=text_box)
    result.save("single_text_result.png")

    # Multiple attempts with style matching
    variations = inpainter.inpaint_text(
        image=image,
        text="Hello World",
        text_box=text_box,
        match_style=True,
        num_attempts=3,
    )

    # Save variations
    for i, img in enumerate(variations):
        img.save(f"variation_{i}.png")


def multiple_text_example():
    """Example of inpainting multiple text regions."""
    inpainter = TextInpainter()
    image = create_sample_image()

    # Define multiple text regions (vertically stacked, centered)
    text_regions = [
        ("First Text", (156, 100, 356, 150)),
        ("Second Text", (156, 200, 356, 250)),
        ("Third Text", (156, 300, 356, 350)),
    ]

    # Batch inpainting
    result = inpainter.batch_inpaint_text(image, text_regions)
    result.save("multiple_text_result.png")


def custom_parameters_example():
    """Example with custom pipeline parameters."""
    inpainter = TextInpainter()
    image = create_sample_image()
    text_box = (156, 100, 356, 150)

    # Custom parameters for more control
    result = inpainter.inpaint_text(
        image=image,
        text="Custom Text",
        text_box=text_box,
        num_inference_steps=75,  # More steps for better quality
        guidance_scale=8.5,  # Stronger prompt adherence
        negative_prompt="blurry, ugly, bad quality, error, watermark",
        match_style=True,
    )
    result.save("custom_params_result.png")


if __name__ == "__main__":
    print("Running single text example...")
    single_text_example()

    print("Running multiple text example...")
    multiple_text_example()

    print("Running custom parameters example...")
    custom_parameters_example()
