"""Utilities for finding and visualizing text regions in images."""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

def visualize_region(image_path, text_box=None):
    """Display image with optional text region highlighted.
    
    Args:
        image_path (str): Path to the image
        text_box (tuple, optional): (x1, y1, x2, y2) coordinates to highlight
    """
    # Load and display image
    img = Image.open(image_path)
    plt.figure(figsize=(12, 8))
    
    # Create a copy for drawing
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    # Draw grid lines every 50 pixels
    for x in range(0, img.width, 50):
        draw.line([(x, 0), (x, img.height)], fill='gray', width=1)
        if x % 100 == 0:  # Add labels for every 100 pixels
            draw.text((x, 5), str(x), fill='gray')
    
    for y in range(0, img.height, 50):
        draw.line([(0, y), (img.width, y)], fill='gray', width=1)
        if y % 100 == 0:  # Add labels for every 100 pixels
            draw.text((5, y), str(y), fill='gray')
    
    # If text box provided, highlight it
    if text_box:
        draw.rectangle(text_box, outline='red', width=2)
        # Add coordinate labels
        x1, y1, x2, y2 = text_box
        draw.text((x1, y1-20), f'({x1}, {y1})', fill='red')
        draw.text((x2, y2+5), f'({x2}, {y2})', fill='red')
    
    plt.imshow(draw_img)
    plt.axis('off')
    plt.show()
    
    return img.size

def detect_text_regions(image_path):
    """Automatically detect text regions using OCR.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        list: List of (text, box) tuples where box is (x1, y1, x2, y2)
    """
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use EAST text detector or Tesseract OCR
    # For now, using simple edge detection as placeholder
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter small regions
        if w > 20 and h > 10:
            regions.append((x, y, x+w, y+h))
    
    return regions

def interactive_region_select(image_path):
    """Display image and let user click points to define region.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of selected region
    """
    img = Image.open(image_path)
    
    print("\nInstructions:")
    print("1. Image will be displayed with a grid overlay")
    print("2. Use the grid lines and coordinates to identify the region")
    print("3. Close the image window when done")
    
    # Show image with grid
    size = visualize_region(image_path)
    
    # Get coordinates from user
    while True:
        try:
            print("\nEnter coordinates (x1 y1 x2 y2) separated by spaces:")
            coords = input("> ").strip().split()
            if len(coords) != 4:
                print("Please enter exactly 4 numbers")
                continue
                
            x1, y1, x2, y2 = map(int, coords)
            
            # Validate coordinates
            if not (0 <= x1 < size[0] and 0 <= x2 < size[0] and 
                   0 <= y1 < size[1] and 0 <= y2 < size[1]):
                print(f"Coordinates must be within image bounds: width={size[0]}, height={size[1]}")
                continue
                
            # Show region for confirmation
            print("\nSelected region (close window to confirm):")
            visualize_region(image_path, (x1, y1, x2, y2))
            
            confirm = input("Is this region correct? (y/n): ").lower().strip()
            if confirm == 'y':
                return (x1, y1, x2, y2)
                
        except ValueError:
            print("Please enter valid numbers")

if __name__ == "__main__":
    # Example usage
    image_path = "example.png"
    
    print("Method 1: Visual Grid Helper")
    region = interactive_region_select(image_path)
    print(f"Selected region: {region}")
    
    print("\nMethod 2: Automatic Detection")
    regions = detect_text_regions(image_path)
    print(f"Detected regions: {regions}") 