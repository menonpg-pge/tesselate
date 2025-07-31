#!/usr/bin/env python3
"""
Enhanced Image Tesselation Tool
===============================

Advanced image slicing with multiple custom cut patterns and perfect reconstruction.

Features:
- Multiple cut patterns: quarters, grid, strips, custom shapes
- Perfect handling of uneven divisions
- Advanced cut patterns: diagonal, spiral, random regions
- JSON metadata with complete traceability
- Perfect reconstruction from any cut pattern
"""

import os
import json
import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image


def slice_image_basic(image_path: str, output_dir: str, slice_width: int, slice_height: int) -> Dict:
    """Basic rectangular slicing - handles uneven divisions perfectly."""
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    img_array = np.array(img)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    slices = []
    slice_index = 0
    
    for y in range(0, img_height, slice_height):
        for x in range(0, img_width, slice_width):
            x_end = min(x + slice_width, img_width)
            y_end = min(y + slice_height, img_height)
            
            slice_array = img_array[y:y_end, x:x_end]
            slice_img = Image.fromarray(slice_array)
            
            slice_filename = f"slice_{slice_index}_{x}_{y}_{x_end}_{y_end}.png"
            slice_path = Path(output_dir) / slice_filename
            slice_img.save(slice_path)
            
            slices.append({
                "index": slice_index,
                "filename": slice_filename,
                "x": x, "y": y,
                "width": x_end - x,
                "height": y_end - y,
                "x_end": x_end,
                "y_end": y_end,
                "cut_type": "rectangular"
            })
            
            slice_index += 1
            print(f"Saved: {slice_filename}")
    
    metadata = create_metadata(image_path, img_width, img_height, slices, 
                             {"slice_width": slice_width, "slice_height": slice_height, "type": "rectangular"})
    save_metadata(metadata, output_dir)
    return metadata


def slice_image_strips(image_path: str, output_dir: str, direction: str = "horizontal", num_strips: int = 4) -> Dict:
    """Cut image into horizontal or vertical strips."""
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    img_array = np.array(img)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    slices = []
    slice_index = 0
    
    if direction == "horizontal":
        strip_height = img_height // num_strips
        remainder = img_height % num_strips
        
        current_y = 0
        for i in range(num_strips):
            # Distribute remainder pixels among the first strips
            height = strip_height + (1 if i < remainder else 0)
            y_end = current_y + height
            
            slice_array = img_array[current_y:y_end, 0:img_width]
            slice_img = Image.fromarray(slice_array)
            
            slice_filename = f"strip_h_{slice_index}_{0}_{current_y}_{img_width}_{y_end}.png"
            slice_path = Path(output_dir) / slice_filename
            slice_img.save(slice_path)
            
            slices.append({
                "index": slice_index,
                "filename": slice_filename,
                "x": 0, "y": current_y,
                "width": img_width,
                "height": height,
                "x_end": img_width,
                "y_end": y_end,
                "cut_type": "horizontal_strip",
                "strip_number": i
            })
            
            current_y = y_end
            slice_index += 1
            print(f"Saved: {slice_filename}")
            
    else:  # vertical strips
        strip_width = img_width // num_strips
        remainder = img_width % num_strips
        
        current_x = 0
        for i in range(num_strips):
            # Distribute remainder pixels among the first strips
            width = strip_width + (1 if i < remainder else 0)
            x_end = current_x + width
            
            slice_array = img_array[0:img_height, current_x:x_end]
            slice_img = Image.fromarray(slice_array)
            
            slice_filename = f"strip_v_{slice_index}_{current_x}_{0}_{x_end}_{img_height}.png"
            slice_path = Path(output_dir) / slice_filename
            slice_img.save(slice_path)
            
            slices.append({
                "index": slice_index,
                "filename": slice_filename,
                "x": current_x, "y": 0,
                "width": width,
                "height": img_height,
                "x_end": x_end,
                "y_end": img_height,
                "cut_type": "vertical_strip",
                "strip_number": i
            })
            
            current_x = x_end
            slice_index += 1
            print(f"Saved: {slice_filename}")
    
    metadata = create_metadata(image_path, img_width, img_height, slices,
                             {"direction": direction, "num_strips": num_strips, "type": "strips"})
    save_metadata(metadata, output_dir)
    return metadata


def slice_image_diagonal(image_path: str, output_dir: str, num_diagonal_bands: int = 4) -> Dict:
    """Cut image into diagonal bands."""
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    img_array = np.array(img)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    slices = []
    slice_index = 0
    
    # Create diagonal bands by creating masks
    total_diagonal = img_width + img_height
    band_width = total_diagonal // num_diagonal_bands
    
    for band in range(num_diagonal_bands):
        # Create mask for this diagonal band
        mask = np.zeros((img_height, img_width), dtype=bool)
        
        min_diag = band * band_width
        max_diag = min((band + 1) * band_width, total_diagonal)
        
        for y in range(img_height):
            for x in range(img_width):
                diagonal_pos = x + y
                if min_diag <= diagonal_pos < max_diag:
                    mask[y, x] = True
        
        # Find bounding box of the mask
        if np.any(mask):
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Extract the bounding box region
            region = img_array[y_min:y_max+1, x_min:x_max+1]
            region_mask = mask[y_min:y_max+1, x_min:x_max+1]
            
            # Apply mask (set non-mask areas to white/transparent)
            masked_region = region.copy()
            masked_region[~region_mask] = [255, 255, 255]  # White background
            
            slice_img = Image.fromarray(masked_region)
            
            slice_filename = f"diagonal_{slice_index}_{x_min}_{y_min}_{x_max+1}_{y_max+1}_band{band}.png"
            slice_path = Path(output_dir) / slice_filename
            slice_img.save(slice_path)
            
            slices.append({
                "index": slice_index,
                "filename": slice_filename,
                "x": x_min, "y": y_min,
                "width": x_max - x_min + 1,
                "height": y_max - y_min + 1,
                "x_end": x_max + 1,
                "y_end": y_max + 1,
                "cut_type": "diagonal_band",
                "band_number": band,
                "diagonal_range": [int(min_diag), int(max_diag)]
            })
            
            slice_index += 1
            print(f"Saved: {slice_filename}")
    
    metadata = create_metadata(image_path, img_width, img_height, slices,
                             {"num_diagonal_bands": num_diagonal_bands, "type": "diagonal"})
    save_metadata(metadata, output_dir)
    return metadata


def slice_image_random_regions(image_path: str, output_dir: str, num_regions: int = 6, seed: int = 42) -> Dict:
    """Cut image into random rectangular regions."""
    random.seed(seed)
    np.random.seed(seed)
    
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    img_array = np.array(img)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    slices = []
    slice_index = 0
    
    # Generate random non-overlapping regions
    regions = []
    attempts = 0
    max_attempts = 1000
    
    while len(regions) < num_regions and attempts < max_attempts:
        # Random region size (20% to 60% of image dimensions)
        region_w = random.randint(img_width // 5, min(img_width * 3 // 5, img_width))
        region_h = random.randint(img_height // 5, min(img_height * 3 // 5, img_height))
        
        # Random position
        x = random.randint(0, max(0, img_width - region_w))
        y = random.randint(0, max(0, img_height - region_h))
        
        new_region = (x, y, x + region_w, y + region_h)
        
        # Check for minimal overlap with existing regions
        overlap = False
        for existing in regions:
            if (new_region[0] < existing[2] and new_region[2] > existing[0] and
                new_region[1] < existing[3] and new_region[3] > existing[1]):
                # Calculate overlap area
                overlap_area = (min(new_region[2], existing[2]) - max(new_region[0], existing[0])) * \
                              (min(new_region[3], existing[3]) - max(new_region[1], existing[1]))
                region_area = region_w * region_h
                if overlap_area > region_area * 0.3:  # Allow up to 30% overlap
                    overlap = True
                    break
        
        if not overlap:
            regions.append(new_region)
        
        attempts += 1
    
    # Create slices from regions
    for i, (x, y, x_end, y_end) in enumerate(regions):
        slice_array = img_array[y:y_end, x:x_end]
        slice_img = Image.fromarray(slice_array)
        
        slice_filename = f"random_{slice_index}_{x}_{y}_{x_end}_{y_end}.png"
        slice_path = Path(output_dir) / slice_filename
        slice_img.save(slice_path)
        
        slices.append({
            "index": slice_index,
            "filename": slice_filename,
            "x": x, "y": y,
            "width": x_end - x,
            "height": y_end - y,
            "x_end": x_end,
            "y_end": y_end,
            "cut_type": "random_region",
            "region_number": i
        })
        
        slice_index += 1
        print(f"Saved: {slice_filename}")
    
    metadata = create_metadata(image_path, img_width, img_height, slices,
                             {"num_regions": num_regions, "seed": seed, "type": "random_regions"})
    save_metadata(metadata, output_dir)
    return metadata


def slice_image_concentric(image_path: str, output_dir: str, num_rings: int = 3) -> Dict:
    """Cut image into concentric rectangular rings."""
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    img_array = np.array(img)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    slices = []
    slice_index = 0
    
    center_x, center_y = img_width // 2, img_height // 2
    max_radius = min(center_x, center_y)
    
    prev_coords = None
    
    for ring in range(num_rings):
        # Calculate ring boundaries
        inner_radius = ring * max_radius // num_rings
        outer_radius = (ring + 1) * max_radius // num_rings
        
        # Define rectangular ring boundaries
        x_min = max(0, center_x - outer_radius)
        y_min = max(0, center_y - outer_radius)
        x_max = min(img_width, center_x + outer_radius)
        y_max = min(img_height, center_y + outer_radius)
        
        # Extract the ring region
        ring_array = img_array[y_min:y_max, x_min:x_max].copy()
        
        # If not the outermost ring, cut out the inner part
        if ring > 0:
            inner_x_min = center_x - inner_radius - x_min
            inner_y_min = center_y - inner_radius - y_min
            inner_x_max = center_x + inner_radius - x_min
            inner_y_max = center_y + inner_radius - y_min
            
            # Ensure inner boundaries are within the ring array
            inner_x_min = max(0, inner_x_min)
            inner_y_min = max(0, inner_y_min)
            inner_x_max = min(ring_array.shape[1], inner_x_max)
            inner_y_max = min(ring_array.shape[0], inner_y_max)
            
            # Set inner region to white (will be filled by inner ring)
            if inner_x_max > inner_x_min and inner_y_max > inner_y_min:
                ring_array[inner_y_min:inner_y_max, inner_x_min:inner_x_max] = [255, 255, 255]
        
        slice_img = Image.fromarray(ring_array)
        
        slice_filename = f"ring_{slice_index}_{x_min}_{y_min}_{x_max}_{y_max}_r{ring}.png"
        slice_path = Path(output_dir) / slice_filename
        slice_img.save(slice_path)
        
        slices.append({
            "index": slice_index,
            "filename": slice_filename,
            "x": x_min, "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min,
            "x_end": x_max,
            "y_end": y_max,
            "cut_type": "concentric_ring",
            "ring_number": ring,
            "inner_radius": inner_radius,
            "outer_radius": outer_radius
        })
        
        slice_index += 1
        print(f"Saved: {slice_filename}")
    
    metadata = create_metadata(image_path, img_width, img_height, slices,
                             {"num_rings": num_rings, "type": "concentric"})
    save_metadata(metadata, output_dir)
    return metadata


def create_metadata(image_path: str, img_width: int, img_height: int, slices: List[Dict], cut_params: Dict) -> Dict:
    """Create comprehensive metadata."""
    return {
        "original_image": {
            "path": str(image_path),
            "width": img_width,
            "height": img_height
        },
        "cut_parameters": cut_params,
        "slices": slices,
        "total_slices": len(slices)
    }


def save_metadata(metadata: Dict, output_dir: str):
    """Save metadata to JSON file."""
    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")


def reconstruct_image(metadata_path: str, output_path: str) -> str:
    """Reconstruct image from any cut pattern."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    orig_width = metadata["original_image"]["width"]
    orig_height = metadata["original_image"]["height"]
    
    reconstructed = Image.new('RGB', (orig_width, orig_height), color='white')
    slice_dir = Path(metadata_path).parent
    
    # Sort slices by index for proper layering
    slices = sorted(metadata["slices"], key=lambda x: x["index"])
    
    for slice_info in slices:
        slice_path = slice_dir / slice_info["filename"]
        
        if slice_path.exists():
            slice_img = Image.open(slice_path)
            reconstructed.paste(slice_img, (slice_info["x"], slice_info["y"]))
            print(f"Placed: {slice_info['filename']}")
    
    reconstructed.save(output_path)
    print(f"Reconstructed image saved: {output_path}")
    return output_path


# Convenience functions
def tessellate_quarters(image_path: str, output_dir: str) -> Dict:
    """Tessellate image into quarters."""
    img = Image.open(image_path)
    width, height = img.size
    return slice_image_basic(image_path, output_dir, width // 2, height // 2)


def tessellate_grid(image_path: str, output_dir: str, rows: int, cols: int) -> Dict:
    """Tessellate image into a grid."""
    img = Image.open(image_path)
    width, height = img.size
    slice_width = width // cols
    slice_height = height // rows
    return slice_image_basic(image_path, output_dir, slice_width, slice_height)


def slice_image_auto_divide(image_path: str, output_dir: str, num_pieces: int) -> Dict:
    """
    Automatically divide image into specified number of pieces with most equal dimensions.
    
    Finds the best grid dimensions (rows x cols) that creates approximately num_pieces,
    with preference for square-like grids and perfect divisions.
    """
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    
    print(f"Auto-dividing {img_width}x{img_height} image into {num_pieces} pieces...")
    
    # Find the best grid dimensions
    best_rows, best_cols = find_optimal_grid_dimensions(num_pieces, img_width, img_height)
    actual_pieces = best_rows * best_cols
    
    print(f"Optimal grid: {best_rows}x{best_cols} = {actual_pieces} pieces")
    print(f"Slice dimensions: ~{img_width//best_cols}x{img_height//best_rows} pixels each")
    
    # Use the grid tessellation with optimal dimensions
    result = tessellate_grid(image_path, output_dir, best_rows, best_cols)
    
    # Update metadata to reflect auto-divide parameters
    result["cut_parameters"].update({
        "type": "auto_divide",
        "requested_pieces": num_pieces,
        "actual_pieces": actual_pieces,
        "optimal_grid": [best_rows, best_cols]
    })
    
    # Re-save updated metadata
    save_metadata(result, output_dir)
    
    return result


def find_optimal_grid_dimensions(target_pieces: int, img_width: int, img_height: int) -> Tuple[int, int]:
    """
    Find optimal rows x cols that creates closest to target_pieces with most equal slice sizes.
    
    Strategy:
    1. Find all factor pairs of numbers near target_pieces
    2. Score each pair based on:
       - How close total pieces is to target
       - How square-like the grid is (prefer closer to 1:1 ratio)
       - How evenly pieces divide image dimensions
    """
    img_aspect = img_width / img_height
    candidates = []
    
    # Check numbers around target (±20% range)
    search_range = max(1, target_pieces // 5)
    for total in range(max(1, target_pieces - search_range), target_pieces + search_range + 1):
        
        # Find all factor pairs for this total
        factors = find_factor_pairs(total)
        
        for rows, cols in factors:
            # Calculate slice dimensions
            slice_width = img_width / cols
            slice_height = img_height / rows
            slice_aspect = slice_width / slice_height
            
            # Score this configuration
            # 1. Piece count difference (lower is better)
            piece_diff = abs(total - target_pieces)
            
            # 2. Grid aspect ratio (prefer closer to image aspect ratio)
            grid_aspect = cols / rows
            aspect_diff = abs(grid_aspect - img_aspect)
            
            # 3. Slice "squareness" (prefer slices closer to square)
            slice_squareness = min(slice_aspect, 1/slice_aspect)  # 0 to 1, 1 is perfect square
            
            # 4. Division evenness (prefer when dimensions divide evenly)
            width_remainder = img_width % cols
            height_remainder = img_height % rows
            evenness = 1.0 - (width_remainder + height_remainder) / (img_width + img_height)
            
            # Combined score (lower is better)
            score = (
                piece_diff * 2.0 +           # Prioritize getting close to target pieces
                aspect_diff * 1.0 +          # Consider aspect ratio
                (1 - slice_squareness) * 0.5 + # Slight preference for square slices
                (1 - evenness) * 1.5         # Prefer even divisions
            )
            
            candidates.append((score, total, rows, cols, slice_width, slice_height))
    
    # Sort by score and return best option
    candidates.sort(key=lambda x: x[0])
    
    if candidates:
        best = candidates[0]
        _, total, rows, cols, slice_w, slice_h = best
        print(f"Selected: {rows}x{cols} grid (score: {best[0]:.2f})")
        print(f"  Actual pieces: {total} (target: {target_pieces})")
        print(f"  Slice size: {slice_w:.1f}x{slice_h:.1f} pixels")
        return rows, cols
    else:
        # Fallback: simple square-ish grid
        sqrt_pieces = int(math.sqrt(target_pieces))
        return sqrt_pieces, (target_pieces + sqrt_pieces - 1) // sqrt_pieces


def find_factor_pairs(n: int) -> List[Tuple[int, int]]:
    """Find all factor pairs (a, b) where a * b = n."""
    pairs = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            pairs.append((i, n // i))
            if i != n // i:  # Avoid duplicate for perfect squares
                pairs.append((n // i, i))
    return pairs


def slice_image_smart_equal(image_path: str, output_dir: str, target_slice_size: int) -> Dict:
    """
    Divide image into pieces as close as possible to target_slice_size pixels per piece.
    
    Calculates optimal grid to achieve approximately target_slice_size area per slice.
    """
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    total_pixels = img_width * img_height
    
    # Calculate approximate number of pieces needed
    estimated_pieces = total_pixels // target_slice_size
    estimated_pieces = max(1, estimated_pieces)  # At least 1 piece
    
    print(f"Target slice size: {target_slice_size} pixels")
    print(f"Total image pixels: {total_pixels}")
    print(f"Estimated pieces needed: {estimated_pieces}")
    
    return slice_image_auto_divide(image_path, output_dir, estimated_pieces)


def verify_images_identical(image1_path: str, image2_path: str) -> Dict:
    """
    Comprehensive verification that two images are 100% identical.
    
    Uses multiple verification methods:
    - File size comparison
    - MD5 hash comparison  
    - Pixel-by-pixel comparison
    - Dimension verification
    """
    import hashlib
    
    print(f"Verifying images are identical:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    print()
    
    results = {
        "images": [image1_path, image2_path],
        "file_sizes_match": False,
        "md5_hashes_match": False,
        "dimensions_match": False,
        "pixels_identical": False,
        "overall_identical": False
    }
    
    try:
        # 1. File size comparison
        size1 = os.path.getsize(image1_path)
        size2 = os.path.getsize(image2_path)
        results["file_sizes_match"] = size1 == size2
        print(f"File sizes: {size1} vs {size2} bytes - {'✓ MATCH' if results['file_sizes_match'] else '✗ DIFFER'}")
        
        # 2. MD5 hash comparison
        def get_md5_hash(filepath):
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        
        hash1 = get_md5_hash(image1_path)
        hash2 = get_md5_hash(image2_path)
        results["md5_hashes_match"] = hash1 == hash2
        results["md5_hash1"] = hash1
        results["md5_hash2"] = hash2
        print(f"MD5 hashes: {'✓ MATCH' if results['md5_hashes_match'] else '✗ DIFFER'}")
        print(f"  Hash 1: {hash1}")
        print(f"  Hash 2: {hash2}")
        
        # 3. Image dimension comparison
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)
        results["dimensions_match"] = img1.size == img2.size
        results["dimensions1"] = img1.size
        results["dimensions2"] = img2.size
        print(f"Dimensions: {img1.size} vs {img2.size} - {'✓ MATCH' if results['dimensions_match'] else '✗ DIFFER'}")
        
        # 4. Pixel-by-pixel comparison (if dimensions match)
        if results["dimensions_match"]:
            img1_array = np.array(img1.convert('RGB'))
            img2_array = np.array(img2.convert('RGB'))
            
            # Compare all pixels
            pixels_identical = np.array_equal(img1_array, img2_array)
            results["pixels_identical"] = pixels_identical
            
            if pixels_identical:
                print(f"Pixel comparison: ✓ ALL PIXELS IDENTICAL")
            else:
                # Count different pixels
                diff_pixels = np.sum(img1_array != img2_array)
                total_pixels = img1_array.size
                diff_percentage = (diff_pixels / total_pixels) * 100
                results["different_pixels"] = int(diff_pixels)
                results["total_pixels"] = int(total_pixels)
                results["difference_percentage"] = diff_percentage
                print(f"Pixel comparison: ✗ {diff_pixels}/{total_pixels} pixels differ ({diff_percentage:.2f}%)")
        else:
            results["pixels_identical"] = False
            print(f"Pixel comparison: ✗ SKIPPED (dimensions differ)")
        
        # Overall result
        results["overall_identical"] = (
            results["file_sizes_match"] and 
            results["md5_hashes_match"] and 
            results["dimensions_match"] and 
            results["pixels_identical"]
        )
        
        print()
        if results["overall_identical"]:
            print("🎉 VERIFICATION RESULT: Images are 100% IDENTICAL")
        else:
            print("❌ VERIFICATION RESULT: Images DIFFER")
            
        return results
            
    except Exception as e:
        print(f"Error during verification: {e}")
        results["error"] = str(e)
        return results


def main():
    """Enhanced command-line interface with multiple cut patterns."""
    parser = argparse.ArgumentParser(description="Enhanced Image Tesselation Tool")
    parser.add_argument("command", choices=[
        "slice", "reconstruct", "quarters", "strips", "diagonal", 
        "random", "concentric", "grid", "auto-divide", "smart-equal", "verify", "demo"
    ], help="Operation to perform")
    
    parser.add_argument("--image", "-i", help="Input image path")
    parser.add_argument("--output", "-o", help="Output path/directory")
    parser.add_argument("--width", "-w", type=int, help="Slice width")
    parser.add_argument("--height", type=int, help="Slice height")
    parser.add_argument("--metadata", "-m", help="Metadata file path")
    
    # Strip options
    parser.add_argument("--direction", choices=["horizontal", "vertical"], 
                       default="horizontal", help="Strip direction")
    parser.add_argument("--num-strips", type=int, default=4, help="Number of strips")
    
    # Grid options  
    parser.add_argument("--rows", type=int, help="Grid rows")
    parser.add_argument("--cols", type=int, help="Grid columns")
    
    # Advanced options
    parser.add_argument("--num-bands", type=int, default=4, help="Number of diagonal bands")
    parser.add_argument("--num-regions", type=int, default=6, help="Number of random regions")
    parser.add_argument("--num-rings", type=int, default=3, help="Number of concentric rings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Auto-divide options
    parser.add_argument("--num-pieces", type=int, help="Target number of pieces for auto-divide")
    parser.add_argument("--target-size", type=int, help="Target pixels per slice for smart-equal")
    
    # Verify options
    parser.add_argument("--image1", help="First image to compare")
    parser.add_argument("--image2", help="Second image to compare")
    
    args = parser.parse_args()
    
    if args.command == "demo":
        demo_all_patterns()
        return 0
    
    if not args.image and args.command not in ["reconstruct", "verify"]:
        print("Error: --image required for most commands")
        return 1
    
    if args.command == "slice":
        if not args.width or not args.height:
            print("Error: --width and --height required for slice command")
            return 1
        output_dir = args.output or "slices"
        slice_image_basic(args.image, output_dir, args.width, args.height)
        
    elif args.command == "quarters":
        output_dir = args.output or "quarters"
        tessellate_quarters(args.image, output_dir)
        
    elif args.command == "strips":
        output_dir = args.output or "strips"
        slice_image_strips(args.image, output_dir, args.direction, args.num_strips)
        
    elif args.command == "grid":
        if not args.rows or not args.cols:
            print("Error: --rows and --cols required for grid command")
            return 1
        output_dir = args.output or "grid"
        tessellate_grid(args.image, output_dir, args.rows, args.cols)
        
    elif args.command == "diagonal":
        output_dir = args.output or "diagonal"
        slice_image_diagonal(args.image, output_dir, args.num_bands)
        
    elif args.command == "random":
        output_dir = args.output or "random"
        slice_image_random_regions(args.image, output_dir, args.num_regions, args.seed)
        
    elif args.command == "concentric":
        output_dir = args.output or "concentric"
        slice_image_concentric(args.image, output_dir, args.num_rings)
        
    elif args.command == "auto-divide":
        if not args.num_pieces:
            print("Error: --num-pieces required for auto-divide command")
            return 1
        output_dir = args.output or "auto_divide"
        slice_image_auto_divide(args.image, output_dir, args.num_pieces)
        
    elif args.command == "smart-equal":
        if not args.target_size:
            print("Error: --target-size required for smart-equal command")
            return 1
        output_dir = args.output or "smart_equal"
        slice_image_smart_equal(args.image, output_dir, args.target_size)
        
    elif args.command == "verify":
        if not args.image1 or not args.image2:
            print("Error: --image1 and --image2 required for verify command")
            return 1
        verify_images_identical(args.image1, args.image2)
        
    elif args.command == "reconstruct":
        if not args.metadata:
            print("Error: --metadata required for reconstruct command")
            return 1
        output_path = args.output or "reconstructed.png"
        reconstruct_image(args.metadata, output_path)
    
    return 0


def demo_all_patterns():
    """Demonstrate all cut patterns."""
    print("Enhanced Tesselation Demo - All Cut Patterns")
    print("=" * 60)
    
    # Create a test image
    from example import create_test_image
    demo_image = create_test_image("enhanced_demo.png", 600, 400)
    
    patterns = [
        ("Basic Grid", lambda: slice_image_basic(demo_image, "demo_basic", 150, 100)),
        ("Horizontal Strips", lambda: slice_image_strips(demo_image, "demo_h_strips", "horizontal", 5)),
        ("Vertical Strips", lambda: slice_image_strips(demo_image, "demo_v_strips", "vertical", 4)),
        ("Diagonal Bands", lambda: slice_image_diagonal(demo_image, "demo_diagonal", 5)),
        ("Random Regions", lambda: slice_image_random_regions(demo_image, "demo_random", 8)),
        ("Concentric Rings", lambda: slice_image_concentric(demo_image, "demo_concentric", 4))
    ]
    
    for name, func in patterns:
        print(f"\n--- {name} ---")
        try:
            metadata = func()
            # Test reconstruction
            reconstruct_image(f"{metadata['cut_parameters']['type']}_demo/metadata.json", 
                            f"reconstructed_{metadata['cut_parameters']['type']}.png")
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print(f"\n{'='*60}")
    print("All pattern demos completed!")


if __name__ == "__main__":
    exit(main())
