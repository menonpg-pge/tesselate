#!/usr/bin/env python3
"""
Minimal Image Tesselation Tool
==============================

Break images into sub-images, store metadata, and reconstruct perfectly.

Core Features:
- Break images into configurable parts (quarters, custom sizes)
- Store JSON metadata tracking slice relationships
- Reconstruct original image from parts + metadata
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image


def slice_image(image_path: str, output_dir: str, slice_width: int, slice_height: int) -> Dict:
    """
    Slice an image into parts and save metadata.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save slices and metadata
        slice_width: Width of each slice
        slice_height: Height of each slice
        
    Returns:
        Dictionary containing slice metadata
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    img_array = np.array(img)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate slice coordinates
    slices = []
    slice_index = 0
    
    for y in range(0, img_height, slice_height):
        for x in range(0, img_width, slice_width):
            # Calculate slice boundaries
            x_end = min(x + slice_width, img_width)
            y_end = min(y + slice_height, img_height)
            
            # Extract slice
            slice_array = img_array[y:y_end, x:x_end]
            slice_img = Image.fromarray(slice_array)
            
            # Save slice
            slice_filename = f"slice_{slice_index}_{x}_{y}_{x_end}_{y_end}.png"
            slice_path = Path(output_dir) / slice_filename
            slice_img.save(slice_path)
            
            # Store metadata
            slices.append({
                "index": slice_index,
                "filename": slice_filename,
                "x": x, "y": y,
                "width": x_end - x,
                "height": y_end - y,
                "x_end": x_end,
                "y_end": y_end
            })
            
            slice_index += 1
            print(f"Saved: {slice_filename}")
    
    # Create metadata
    metadata = {
        "original_image": {
            "path": str(image_path),
            "width": img_width,
            "height": img_height
        },
        "slice_size": {
            "width": slice_width,
            "height": slice_height
        },
        "slices": slices,
        "total_slices": len(slices)
    }
    
    # Save metadata
    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created {len(slices)} slices")
    print(f"Metadata saved: {metadata_path}")
    
    return metadata


def reconstruct_image(metadata_path: str, output_path: str) -> str:
    """
    Reconstruct original image from slices and metadata.
    
    Args:
        metadata_path: Path to metadata JSON file
        output_path: Path for reconstructed image
        
    Returns:
        Path to reconstructed image
    """
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get original dimensions
    orig_width = metadata["original_image"]["width"]
    orig_height = metadata["original_image"]["height"]
    
    # Create blank canvas
    reconstructed = Image.new('RGB', (orig_width, orig_height))
    
    # Get slice directory
    slice_dir = Path(metadata_path).parent
    
    # Place each slice
    for slice_info in metadata["slices"]:
        slice_path = slice_dir / slice_info["filename"]
        
        if slice_path.exists():
            # Load slice
            slice_img = Image.open(slice_path)
            
            # Paste at original position
            reconstructed.paste(slice_img, (slice_info["x"], slice_info["y"]))
            print(f"Placed: {slice_info['filename']}")
    
    # Save reconstructed image
    reconstructed.save(output_path)
    print(f"Reconstructed image saved: {output_path}")
    
    return output_path


def tessellate_quarters(image_path: str, output_dir: str) -> Dict:
    """Convenience function to tessellate image into quarters."""
    img = Image.open(image_path)
    width, height = img.size
    return slice_image(image_path, output_dir, width // 2, height // 2)


def tessellate_grid(image_path: str, output_dir: str, rows: int, cols: int) -> Dict:
    """Convenience function to tessellate image into a grid."""
    img = Image.open(image_path)
    width, height = img.size
    slice_width = width // cols
    slice_height = height // rows
    return slice_image(image_path, output_dir, slice_width, slice_height)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Image Tesselation Tool")
    parser.add_argument("command", choices=["slice", "reconstruct", "quarters"], 
                       help="Operation: slice, reconstruct, or quarters")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", help="Output path/directory")
    parser.add_argument("--width", "-w", type=int, help="Slice width")
    parser.add_argument("--height", type=int, help="Slice height")
    parser.add_argument("--rows", type=int, help="Grid rows")
    parser.add_argument("--cols", type=int, help="Grid columns")
    parser.add_argument("--metadata", "-m", help="Metadata file path")
    
    args = parser.parse_args()
    
    if args.command == "slice":
        if not args.width or not args.height:
            print("Error: --width and --height required for slice command")
            return 1
        output_dir = args.output or "slices"
        slice_image(args.image, output_dir, args.width, args.height)
        
    elif args.command == "quarters":
        output_dir = args.output or "quarters"
        tessellate_quarters(args.image, output_dir)
        
    elif args.command == "reconstruct":
        if not args.metadata:
            print("Error: --metadata required for reconstruct command")
            return 1
        output_path = args.output or "reconstructed.png"
        reconstruct_image(args.metadata, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
