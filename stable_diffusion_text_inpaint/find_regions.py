"""Script to help find text regions in an image."""

import argparse
from utils.region_finder import interactive_region_select, detect_text_regions, visualize_region

def main():
    parser = argparse.ArgumentParser(description="Find text regions in an image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--auto", action="store_true", help="Use automatic detection")
    args = parser.parse_args()
    
    if args.auto:
        print("Detecting text regions automatically...")
        regions = detect_text_regions(args.image_path)
        print(f"\nFound {len(regions)} regions:")
        for i, region in enumerate(regions, 1):
            print(f"Region {i}: {region}")
            visualize_region(args.image_path, region)
    else:
        print("Starting interactive region selection...")
        region = interactive_region_select(args.image_path)
        print(f"\nSelected region coordinates: {region}")
        print("You can use these coordinates with the TextInpainter class.")

if __name__ == "__main__":
    main() 