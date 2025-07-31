#!/usr/bin/env python3
"""
Simple Example of Image Tesselation
===================================

This example shows how to:
1. Break an image into quarters
2. Break an image into custom slices  
3. Reconstruct the original from parts + metadata
"""

from tesselate import slice_image, reconstruct_image, tessellate_quarters
from PIL import Image, ImageDraw
import os

def create_test_image(filename="test_image.png", width=800, height=600):
    """Create a simple test image with colored squares."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw colored squares
    draw.rectangle([0, 0, width//2, height//2], fill='red')
    draw.rectangle([width//2, 0, width, height//2], fill='green') 
    draw.rectangle([0, height//2, width//2, height], fill='blue')
    draw.rectangle([width//2, height//2, width, height], fill='yellow')
    
    # Add some text
    draw.text((width//2-50, height//2-10), "TEST", fill='black')
    
    img.save(filename)
    print(f"Created test image: {filename}")
    return filename

def example_1_quarters():
    """Example 1: Break image into quarters."""
    print("\n=== Example 1: Quarters ===")
    
    # Create test image
    image_path = create_test_image("example1.png", 800, 600)
    
    # Break into quarters
    metadata = tessellate_quarters(image_path, "example1_quarters")
    
    # Reconstruct
    reconstruct_image("example1_quarters/metadata.json", "example1_reconstructed.png")
    
    print("✓ Example 1 complete! Check example1_quarters/ folder")

def example_2_custom_slices():
    """Example 2: Custom slice sizes."""
    print("\n=== Example 2: Custom Slices ===")
    
    # Create test image
    image_path = create_test_image("example2.png", 1200, 800) 
    
    # Break into 200x200 slices
    metadata = slice_image(image_path, "example2_slices", 200, 200)
    
    # Reconstruct
    reconstruct_image("example2_slices/metadata.json", "example2_reconstructed.png")
    
    print("✓ Example 2 complete! Check example2_slices/ folder")

def example_3_real_image():
    """Example 3: Using a real image (if available)."""
    print("\n=== Example 3: Real Image ===")
    
    # Try to use SAHI demo image if available
    test_images = [
        "../sahi/demo/demo_data/terrain2.png",
        "../demo_basic.png",
        "example1.png"  # fallback
    ]
    
    image_path = None
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("No test image found, skipping example 3")
        return
    
    print(f"Using image: {image_path}")
    
    # Break into 256x256 slices
    metadata = slice_image(image_path, "example3_real", 256, 256)
    
    # Reconstruct
    reconstruct_image("example3_real/metadata.json", "example3_reconstructed.png")
    
    print("✓ Example 3 complete! Check example3_real/ folder")

def main():
    """Run all examples."""
    print("Image Tesselation Examples")
    print("=" * 50)
    
    example_1_quarters()
    example_2_custom_slices()  
    example_3_real_image()
    
    print("\n" + "=" * 50)
    print("All examples complete!")
    print("\nGenerated files:")
    print("- example1_quarters/ - Image broken into 4 parts")
    print("- example2_slices/ - Image broken into 200x200 slices") 
    print("- example3_real/ - Real image sliced into 256x256 parts")
    print("- *_reconstructed.png - Perfectly reconstructed images")

if __name__ == "__main__":
    main()
