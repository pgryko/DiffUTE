"""Utility functions for analyzing and matching text styles."""

from PIL import Image
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


class TextStyleAnalyzer:
    def __init__(self):
        """Initialize the text style analyzer with TrOCR model."""
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-large-printed"
        )

    def analyze_text_region(self, image, text_box):
        """Analyze text style in the specified region.

        Args:
            image (PIL.Image): Input image
            text_box (tuple): (x1, y1, x2, y2) coordinates for text region

        Returns:
            dict: Style properties including font, color, size
        """
        # Crop the text region
        x1, y1, x2, y2 = text_box
        text_region = image.crop((x1, y1, x2, y2))

        # Ensure image is in RGB format
        if text_region.mode != 'RGB':
            text_region = text_region.convert('RGB')

        try:
            # Get text features from TrOCR
            pixel_values = self.processor(text_region, return_tensors="pt").pixel_values
            features = self.model.encoder(pixel_values).last_hidden_state
        except Exception as e:
            print(f"Warning: Style analysis failed ({str(e)}), using basic analysis only")
            features = None

        # Analyze basic properties
        style_props = {
            "size": y2 - y1,  # Approximate text height
            "width": x2 - x1,  # Region width
            "color": self._analyze_color(text_region),
            "background": self._analyze_background(text_region),
        }

        return style_props

    def _analyze_color(self, text_region):
        """Analyze the dominant text color."""
        # Convert to numpy array
        img_array = np.array(text_region)

        # Simple color analysis (can be improved)
        mean_color = np.mean(img_array, axis=(0, 1))
        return tuple(map(int, mean_color))

    def _analyze_background(self, text_region):
        """Analyze the background color."""
        img_array = np.array(text_region)

        # Assume corners are background
        corners = [
            img_array[0, 0],
            img_array[0, -1],
            img_array[-1, 0],
            img_array[-1, -1],
        ]
        bg_color = np.mean(corners, axis=0)
        return tuple(map(int, bg_color))


def generate_style_prompt(style_props):
    """Generate a text prompt based on style properties.

    Args:
        style_props (dict): Style properties from TextStyleAnalyzer

    Returns:
        str: Generated prompt for stable diffusion
    """
    # Convert RGB colors to descriptive terms
    text_color = _rgb_to_description(style_props["color"])
    bg_color = _rgb_to_description(style_props["background"])

    prompt = f"clear {text_color} text on {bg_color} background"

    # Add size information
    if style_props["size"] < 20:
        prompt = "small " + prompt
    elif style_props["size"] > 40:
        prompt = "large " + prompt

    return prompt


def _rgb_to_description(rgb):
    """Convert RGB values to color description."""
    r, g, b = rgb

    # Simple color mapping (can be expanded)
    if max(r, g, b) < 50:
        return "black"
    elif min(r, g, b) > 200:
        return "white"
    elif r > max(g, b) + 50:
        return "red"
    elif g > max(r, b) + 50:
        return "green"
    elif b > max(r, g) + 50:
        return "blue"
    else:
        return "gray"
