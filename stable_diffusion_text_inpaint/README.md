# Using Stable Diffusion for Text Inpainting

This guide explains how to use Stable Diffusion's inpainting capability to add text to specific regions in an image. While not as specialized as DiffUTE for text editing, this approach can still achieve decent results.

## Requirements

```python
pip install diffusers transformers torch
```

## Basic Implementation

```python
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import numpy as np

def create_text_mask(image, text_box):
    """Create a binary mask for the text region
    
    Args:
        image: PIL Image
        text_box: tuple of (x1, y1, x2, y2) coordinates
    """
    mask = Image.new("RGB", image.size, "black")
    draw = ImageDraw.Draw(mask)
    draw.rectangle(text_box, fill="white")
    return mask

# Load the model
model_id = "stabilityai/stable-diffusion-2-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# Load your image
image = Image.open("your_image.png")

# Define the text region (x1, y1, x2, y2)
text_box = (100, 100, 300, 150)  # Example coordinates

# Create the mask
mask = create_text_mask(image, text_box)

# Generate the inpainting
prompt = "Clear black text saying 'Hello World' on a white background"
negative_prompt = "blurry, unclear text, multiple texts, watermark"

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]
```

## Tips for Better Results

1. **Mask Preparation**:
   - Make the mask slightly larger than the text area
   - Use anti-aliasing on mask edges for smoother blending
   - Consider the text baseline and x-height in mask creation

2. **Prompt Engineering**:
   - Be specific about text style: "sharp, clear black text"
   - Mention text properties: "centered, serif font"
   - Include context: "text on a white background"

3. **Negative Prompts**:
   - "blurry, unclear text"
   - "multiple texts, overlapping text"
   - "watermark, artifacts"
   - "distorted, warped text"

4. **Parameter Tuning**:
   ```python
   # For clearer text
   result = pipe(
       prompt=prompt,
       negative_prompt=negative_prompt,
       image=image,
       mask_image=mask,
       num_inference_steps=50,  # More steps for better quality
       guidance_scale=7.5,      # Higher for more prompt adherence
       strength=0.8,            # Control how much to change
   ).images[0]
   ```

## Advanced Usage

### 1. Style Matching

To match existing text styles in the image:

```python
def match_text_style(image, text_region):
    """Analyze existing text style in the image"""
    # Add OCR or style analysis here
    return "style_description"

style = match_text_style(image, text_region)
prompt = f"Text saying 'Hello World' in style: {style}"
```

### 2. Context-Aware Masking

```python
def create_context_mask(image, text_box, padding=10):
    """Create a mask with context awareness"""
    x1, y1, x2, y2 = text_box
    padded_box = (x1-padding, y1-padding, x2+padding, y2+padding)
    mask = create_text_mask(image, padded_box)
    return mask
```

### 3. Multiple Attempts

```python
def generate_multiple_attempts(pipe, image, mask, prompt, num_attempts=3):
    """Generate multiple versions and pick the best"""
    results = []
    for _ in range(num_attempts):
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=50,
        ).images[0]
        results.append(result)
    return results
```

## Limitations

1. Less precise text control compared to DiffUTE
2. May require multiple attempts to get desired results
3. Text style matching is less reliable
4. May introduce artifacts around text regions

## Best Practices

1. **Preparation**:
   - Clean the text region thoroughly
   - Create precise masks
   - Use high-resolution images

2. **Generation**:
   - Start with lower strength values
   - Generate multiple variations
   - Use detailed prompts

3. **Post-processing**:
   - Check text clarity and alignment
   - Verify style consistency
   - Touch up edges if needed

## When to Use DiffUTE Instead

Consider using DiffUTE when:
- Precise text style matching is crucial
- Multiple text regions need editing
- Text needs to perfectly match surrounding context
- Working with complex backgrounds 