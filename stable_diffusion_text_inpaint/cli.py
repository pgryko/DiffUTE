"""Command line interface for text inpainting."""

import click
from PIL import Image
import sys
from text_inpainter import TextInpainter
from utils.region_finder import interactive_region_select, detect_text_regions, visualize_region

@click.group()
def cli():
    """Text inpainting tools using Stable Diffusion."""
    pass

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--text', '-t', help='Text to add (if not provided, will be prompted)')
@click.option('--region', '-r', nargs=4, type=int, help='Text region coordinates (x1 y1 x2 y2)')
@click.option('--match-style/--no-match-style', default=True, help='Match existing text style')
@click.option('--attempts', '-a', default=1, help='Number of variations to generate')
@click.option('--output', '-o', help='Output file path (default: inpainted_<original>)')
@click.option('--device', default='cuda', help='Device to run on (cuda or cpu)')
@click.option('--interactive/--no-interactive', default=True, help='Use interactive region selection')
def inpaint(image_path, text, region, match_style, attempts, output, device, interactive):
    """Inpaint text in an image.
    
    Examples:
        # Interactive mode (recommended for first use)
        python cli.py inpaint image.png
        
        # Specify everything via command line
        python cli.py inpaint image.png -t "Hello" -r 100 50 300 100
        
        # Generate multiple variations
        python cli.py inpaint image.png -t "Hello" -a 3
    """
    try:
        # Load image
        try:
            image = Image.open(image_path)
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            click.echo(f"Error loading image: {str(e)}", err=True)
            sys.exit(1)
        
        # Get text if not provided
        if not text:
            text = click.prompt('Enter the text to add', type=str)
        
        # Get region if not provided
        if not region and interactive:
            click.echo("\nSelect the region for the text:")
            try:
                region = interactive_region_select(image_path)
            except Exception as e:
                click.echo(f"Error selecting region: {str(e)}", err=True)
                sys.exit(1)
        elif not region:
            click.echo("Error: Must provide either --region or use --interactive", err=True)
            sys.exit(1)
        
        # Initialize inpainter
        click.echo("\nInitializing Stable Diffusion (this may take a moment)...")
        try:
            inpainter = TextInpainter(device=device)
        except Exception as e:
            click.echo(f"Error initializing Stable Diffusion: {str(e)}", err=True)
            sys.exit(1)
        
        # Generate the inpainting
        click.echo(f"\nInpainting text: '{text}'")
        with click.progressbar(length=attempts, label='Generating variations') as bar:
            try:
                results = inpainter.inpaint_text(
                    image=image,
                    text=text,
                    text_box=region,
                    match_style=match_style,
                    num_attempts=attempts
                )
                bar.update(attempts)
            except Exception as e:
                click.echo(f"\nError during inpainting: {str(e)}", err=True)
                sys.exit(1)
        
        # Save results
        if not output:
            import os
            base, ext = os.path.splitext(image_path)
            output = f"{base}_inpainted{ext}"
            
        if attempts == 1:
            results = [results]  # Make it a list for consistent handling
            
        # Save all variations
        for i, result in enumerate(results):
            try:
                if attempts > 1:
                    base, ext = os.path.splitext(output)
                    save_path = f"{base}_{i+1}{ext}"
                else:
                    save_path = output
                result.save(save_path)
                click.echo(f"Saved result to: {save_path}")
            except Exception as e:
                click.echo(f"Error saving result {i+1}: {str(e)}", err=True)
                continue
                
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--auto/--no-auto', default=False, help='Use automatic detection')
def find_regions(image_path, auto):
    """Find text regions in an image.
    
    This command helps identify the coordinates of text regions for inpainting.
    """
    try:
        if auto:
            click.echo("Detecting text regions automatically...")
            regions = detect_text_regions(image_path)
            click.echo(f"\nFound {len(regions)} regions:")
            for i, region in enumerate(regions, 1):
                click.echo(f"Region {i}: {region}")
                visualize_region(image_path, region)
        else:
            click.echo("Starting interactive region selection...")
            region = interactive_region_select(image_path)
            click.echo(f"\nSelected region coordinates: {region}")
            click.echo("\nTo use these coordinates with the inpaint command:")
            click.echo(f"python cli.py inpaint {image_path} -r {region[0]} {region[1]} {region[2]} {region[3]}")
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli() 