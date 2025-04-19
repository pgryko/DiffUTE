"""Main class for text inpainting using Stable Diffusion."""

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from tqdm import tqdm

from utils.mask_utils import create_context_mask, validate_text_box
from utils.style_utils import TextStyleAnalyzer, generate_style_prompt


class TextInpainter:
    def __init__(self, device="cuda"):
        """Initialize the text inpainting pipeline.

        Args:
            device (str): Device to run the model on ("cuda" or "cpu")
        """
        self.device = device
        self.model_id = "stabilityai/stable-diffusion-2-inpainting"

        # Initialize pipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.pipe = self.pipe.to(device)

        # Initialize style analyzer
        self.style_analyzer = TextStyleAnalyzer()

    def inpaint_text(
        self, image, text, text_box, match_style=True, num_attempts=1, **kwargs
    ):
        """Inpaint text in the specified region.

        Args:
            image (PIL.Image): Input image
            text (str): Text to add
            text_box (tuple): (x1, y1, x2, y2) coordinates for text region
            match_style (bool): Whether to match existing text style
            num_attempts (int): Number of generation attempts
            **kwargs: Additional arguments for the pipeline

        Returns:
            PIL.Image: Image with inpainted text
            list: All generated variations if num_attempts > 1
        """
        # Validate text box
        text_box = validate_text_box(image.size, text_box)

        # Create mask
        mask = create_context_mask(image, text_box)

        # Generate prompt
        if match_style:
            style_props = self.style_analyzer.analyze_text_region(image, text_box)
            style_prompt = generate_style_prompt(style_props)
            prompt = f"{style_prompt}, text saying '{text}'"
        else:
            prompt = f"Clear text saying '{text}'"

        # Default parameters
        params = {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, unclear text, multiple texts, watermark",
        }
        params.update(kwargs)

        # Generate multiple attempts
        results = []
        for _ in tqdm(range(num_attempts), desc="Generating variations"):
            result = self.pipe(
                prompt=prompt, image=image, mask_image=mask, **params
            ).images[0]
            results.append(result)

        return results[0] if num_attempts == 1 else results

    def batch_inpaint_text(self, image, text_regions):
        """Inpaint multiple text regions in an image.

        Args:
            image (PIL.Image): Input image
            text_regions (list): List of (text, box) tuples

        Returns:
            PIL.Image: Image with all text regions inpainted
        """
        result = image.copy()
        for text, box in text_regions:
            result = self.inpaint_text(result, text, box)
        return result


def main():
    """Example usage of TextInpainter."""
    # Initialize inpainter
    inpainter = TextInpainter()

    # Load image
    image = Image.open("example.png")

    # Define text region
    text_box = (100, 100, 300, 150)

    # Inpaint text
    result = inpainter.inpaint_text(
        image=image, text="Hello World", text_box=text_box, num_attempts=3
    )

    # Save results
    if isinstance(result, list):
        for i, img in enumerate(result):
            img.save(f"result_{i}.png")
    else:
        result.save("result.png")


if __name__ == "__main__":
    main()
