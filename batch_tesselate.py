#!/usr/bin/env python3
"""
Batch Image Tesselation for SageMaker
====================================

Large-scale batch processing of TIFF images in nested directory structures.
Optimized for SageMaker Jupyter Lab environments with 12GB+ datasets.

Features:
- Processes only .tif/.tiff files
- Skips folders containing only JPG files
- Parallel processing with progress tracking
- Resume capability for interrupted jobs
- Memory-efficient processing for large files
- Comprehensive error handling and logging
"""

import os
import json
import time
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import multiprocessing as mp

import numpy as np
from PIL import Image
from tqdm import tqdm

# Import our tesselation functions
from enhanced_tesselate import slice_image_auto_divide, verify_images_identical


@dataclass
class BatchJob:
    """Configuration for a batch processing job."""
    input_dir: str
    output_dir: str
    num_pieces: int = 8
    max_workers: int = 4
    min_image_size: int = 1024  # Skip images smaller than this
    skip_existing: bool = True
    resume_mode: bool = True
    log_level: str = "INFO"


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    total_found: int = 0
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    folders_skipped: int = 0
    start_time: float = 0
    end_time: float = 0


class BatchTesselator:
    """Main batch processing class for large-scale image tesselation."""
    
    def __init__(self, job_config: BatchJob):
        self.config = job_config
        self.stats = ProcessingStats()
        self.processed_files: Set[str] = set()
        self.error_log: List[Dict] = []
        
        # Setup logging
        self.setup_logging()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # State files for resume capability
        self.state_file = Path(self.config.output_dir) / "batch_state.json"
        self.report_file = Path(self.config.output_dir) / "batch_report.json"
        
        # Load previous state if resuming
        if self.config.resume_mode:
            self.load_state()
    
    def setup_logging(self):
        """Configure logging for batch processing."""
        # Ensure output directory exists before creating log file
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(self.config.output_dir) / "batch_processing.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def find_tiff_images(self) -> List[Tuple[str, str]]:
        """
        Find all TIFF images in the input directory.
        Returns list of (input_path, relative_path) tuples.
        Skips folders that contain only JPG files.
        """
        self.logger.info(f"Scanning directory: {self.config.input_dir}")
        
        tiff_files = []
        skipped_folders = []
        
        input_path = Path(self.config.input_dir)
        
        for root, dirs, files in os.walk(input_path):
            root_path = Path(root)
            
            # Only check skip logic for folders that actually contain files
            if files and self.should_skip_folder(root_path, files):
                skipped_folders.append(str(root_path.relative_to(input_path)))
                dirs.clear()  # Don't recurse into subdirectories
                continue
            
            # Find TIFF files in this folder
            for file in files:
                if self.is_tiff_file(file):
                    full_path = root_path / file
                    relative_path = full_path.relative_to(input_path)
                    
                    # Check if already processed
                    if self.config.skip_existing and str(relative_path) in self.processed_files:
                        continue
                    
                    # Check minimum size
                    if self.meets_size_requirements(full_path):
                        tiff_files.append((str(full_path), str(relative_path)))
        
        self.stats.folders_skipped = len(skipped_folders)
        self.stats.total_found = len(tiff_files)
        
        self.logger.info(f"Found {len(tiff_files)} TIFF images to process")
        self.logger.info(f"Skipped {len(skipped_folders)} folders (JPG-only or no images)")
        
        if skipped_folders:
            self.logger.debug(f"Skipped folders: {skipped_folders[:10]}...")  # Show first 10
        
        return tiff_files
    
    def should_skip_folder(self, folder_path: Path, files: List[str]) -> bool:
        """
        Determine if a folder should be skipped.
        Skip if:
        - Contains only JPG/JPEG files
        - Contains no image files at all
        - Contains only small/thumbnail images
        """
        image_files = [f for f in files if self.is_image_file(f)]
        
        if not image_files:
            return True  # No image files
        
        tiff_files = [f for f in files if self.is_tiff_file(f)]
        jpg_files = [f for f in files if self.is_jpg_file(f)]
        
        # Skip if only JPG files and no TIFF files
        if jpg_files and not tiff_files:
            return True
        
        return False
    
    def is_tiff_file(self, filename: str) -> bool:
        """Check if file is a TIFF image."""
        return filename.lower().endswith(('.tif', '.tiff'))
    
    def is_jpg_file(self, filename: str) -> bool:
        """Check if file is a JPG image."""
        return filename.lower().endswith(('.jpg', '.jpeg'))
    
    def is_image_file(self, filename: str) -> bool:
        """Check if file is any supported image format."""
        return filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'))
    
    def meets_size_requirements(self, image_path: Path) -> bool:
        """Check if image meets minimum size requirements."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                return width >= self.config.min_image_size and height >= self.config.min_image_size
        except Exception as e:
            self.logger.warning(f"Could not check size for {image_path}: {e}")
            return False
    
    def process_single_image(self, image_info: Tuple[str, str]) -> Dict:
        """
        Process a single image. Designed to run in separate process.
        Returns processing result dictionary.
        """
        input_path, relative_path = image_info
        
        try:
            # Create output directory structure
            rel_path_obj = Path(relative_path)
            image_name = rel_path_obj.stem
            output_subdir = Path(self.config.output_dir) / rel_path_obj.parent / image_name
            
            # Skip if already exists and not in resume mode
            if output_subdir.exists() and self.config.skip_existing:
                return {
                    "status": "skipped",
                    "input_path": input_path,
                    "relative_path": relative_path,
                    "reason": "already_exists"
                }
            
            # Process the image
            start_time = time.time()
            
            # Use our auto-divide function
            metadata = slice_image_auto_divide(
                input_path, 
                str(output_subdir), 
                self.config.num_pieces
            )
            
            # Add batch processing metadata
            metadata["batch_info"] = {
                "original_path": input_path,
                "relative_path": relative_path,
                "processed_at": time.time(),
                "processing_time": time.time() - start_time,
                "batch_config": asdict(self.config)
            }
            
            # Save enhanced metadata
            with open(output_subdir / "batch_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "input_path": input_path,
                "relative_path": relative_path,
                "output_dir": str(output_subdir),
                "slices_created": metadata["total_slices"],
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {
                "status": "error",
                "input_path": input_path,
                "relative_path": relative_path,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def save_state(self):
        """Save current processing state for resume capability."""
        state = {
            "processed_files": list(self.processed_files),
            "stats": asdict(self.stats),
            "config": asdict(self.config),
            "timestamp": time.time()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load previous processing state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.processed_files = set(state.get("processed_files", []))
                self.logger.info(f"Resumed: {len(self.processed_files)} files already processed")
                
            except Exception as e:
                self.logger.warning(f"Could not load previous state: {e}")
    
    def generate_report(self):
        """Generate comprehensive batch processing report."""
        self.stats.end_time = time.time()
        total_time = self.stats.end_time - self.stats.start_time
        
        report = {
            "batch_summary": {
                "total_images_found": self.stats.total_found,
                "successfully_processed": self.stats.processed,
                "skipped": self.stats.skipped,
                "errors": self.stats.errors,
                "folders_skipped": self.stats.folders_skipped,
                "total_processing_time": total_time,
                "average_time_per_image": total_time / max(self.stats.processed, 1),
                "images_per_hour": (self.stats.processed / total_time) * 3600 if total_time > 0 else 0
            },
            "configuration": asdict(self.config),
            "errors": self.error_log,
            "timestamp": time.time()
        }
        
        with open(self.report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Total images found:     {self.stats.total_found}")
        print(f"Successfully processed: {self.stats.processed}")
        print(f"Skipped:               {self.stats.skipped}")
        print(f"Errors:                {self.stats.errors}")
        print(f"Folders skipped:       {self.stats.folders_skipped}")
        print(f"Total time:            {total_time:.1f} seconds")
        print(f"Average per image:     {total_time/max(self.stats.processed,1):.1f} seconds")
        print(f"Processing rate:       {(self.stats.processed/total_time)*3600:.1f} images/hour")
        print(f"Report saved to:       {self.report_file}")
        print("="*60)
        
        return report
    
    def run_batch(self):
        """Execute the batch processing job."""
        self.logger.info("Starting batch tesselation job")
        self.stats.start_time = time.time()
        
        # Find all TIFF images
        tiff_images = self.find_tiff_images()
        
        if not tiff_images:
            self.logger.warning("No TIFF images found to process!")
            return self.generate_report()
        
        # Process images in parallel
        self.logger.info(f"Processing {len(tiff_images)} images with {self.config.max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_image = {
                executor.submit(self.process_single_image, img_info): img_info 
                for img_info in tiff_images
            }
            
            # Process results with progress bar
            with tqdm(total=len(tiff_images), desc="Processing images") as pbar:
                for future in as_completed(future_to_image):
                    result = future.result()
                    
                    # Update statistics
                    if result["status"] == "success":
                        self.stats.processed += 1
                        self.processed_files.add(result["relative_path"])
                        self.logger.debug(f"Processed: {result['relative_path']}")
                        
                    elif result["status"] == "skipped":
                        self.stats.skipped += 1
                        self.logger.debug(f"Skipped: {result['relative_path']}")
                        
                    elif result["status"] == "error":
                        self.stats.errors += 1
                        self.error_log.append(result)
                        self.logger.error(f"Error processing {result['relative_path']}: {result['error']}")
                    
                    pbar.update(1)
                    
                    # Save state periodically
                    if (self.stats.processed + self.stats.skipped + self.stats.errors) % 10 == 0:
                        self.save_state()
        
        # Final state save
        self.save_state()
        
        # Generate final report
        return self.generate_report()


def create_sagemaker_notebook():
    """Create a Jupyter notebook optimized for SageMaker."""
    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Batch Image Tesselation for SageMaker\\n",
    "\\n",
    "Large-scale processing of TIFF images with automatic slicing and metadata generation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required packages if needed\\n",
    "!pip install pillow numpy tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\\n",
    "from batch_tesselate import BatchTesselator, BatchJob\\n",
    "\\n",
    "# Configuration\\n",
    "INPUT_DIR = '/home/ec2-user/SageMaker/12gb_dataset'  # Update this path\\n",
    "OUTPUT_DIR = '/home/ec2-user/SageMaker/processed_output'\\n",
    "\\n",
    "# Create batch job configuration\\n",
    "job_config = BatchJob(\\n",
    "    input_dir=INPUT_DIR,\\n",
    "    output_dir=OUTPUT_DIR,\\n",
    "    num_pieces=8,           # Split each image into 8 pieces\\n",
    "    max_workers=4,          # Adjust based on instance specs\\n",
    "    min_image_size=1024,    # Skip images smaller than 1024px\\n",
    "    skip_existing=True,     # Skip already processed images\\n",
    "    resume_mode=True,       # Enable resume capability\\n",
    "    log_level='INFO'\\n",
    ")\\n",
    "\\n",
    "print(f'Input directory: {INPUT_DIR}')\\n",
    "print(f'Output directory: {OUTPUT_DIR}')\\n",
    "print(f'Configuration: {job_config}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize batch processor\\n",
    "processor = BatchTesselator(job_config)\\n",
    "\\n",
    "# Run a quick scan to see what will be processed\\n",
    "tiff_files = processor.find_tiff_images()\\n",
    "print(f'Found {len(tiff_files)} TIFF images to process')\\n",
    "\\n",
    "# Show first few files as preview\\n",
    "if tiff_files:\\n",
    "    print('\\\\nFirst 10 files to process:')\\n",
    "    for i, (full_path, rel_path) in enumerate(tiff_files[:10]):\\n",
    "        print(f'{i+1:2d}. {rel_path}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Start batch processing\\n",
    "# WARNING: This will process ALL TIFF files found\\n",
    "# Make sure the configuration above is correct before running\\n",
    "\\n",
    "print('Starting batch processing...')\\n",
    "print('This may take several hours for large datasets')\\n",
    "print('You can interrupt and resume later if needed')\\n",
    "\\n",
    "report = processor.run_batch()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check results and create ZIP archive\\n",
    "import shutil\\n",
    "\\n",
    "# Create ZIP of processed results\\n",
    "zip_filename = f'{OUTPUT_DIR}_results.zip'\\n",
    "print(f'Creating ZIP archive: {zip_filename}')\\n",
    "\\n",
    "shutil.make_archive(\\n",
    "    OUTPUT_DIR + '_results',\\n",
    "    'zip',\\n",
    "    OUTPUT_DIR\\n",
    ")\\n",
    "\\n",
    "# Get ZIP file size\\n",
    "zip_size = os.path.getsize(zip_filename) / (1024**3)  # GB\\n",
    "print(f'ZIP file created: {zip_filename} ({zip_size:.2f} GB)')\\n",
    "\\n",
    "print('\\\\nReady for download and S3 upload!')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    with open('SageMaker_Batch_Processing.ipynb', 'w') as f:
        f.write(notebook_content)


def main():
    """Command-line interface for batch processing."""
    parser = argparse.ArgumentParser(description="Batch Image Tesselation")
    parser.add_argument("input_dir", help="Input directory containing TIFF images")
    parser.add_argument("output_dir", help="Output directory for processed images")
    parser.add_argument("--num-pieces", type=int, default=8, help="Number of pieces per image")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--min-size", type=int, default=1024, help="Minimum image size to process")
    parser.add_argument("--resume", action="store_true", help="Resume previous job")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Create job configuration
    job_config = BatchJob(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_pieces=args.num_pieces,
        max_workers=args.max_workers,
        min_image_size=args.min_size,
        resume_mode=args.resume,
        log_level=args.log_level
    )
    
    # Run batch job
    processor = BatchTesselator(job_config)
    report = processor.run_batch()
    
    return report


if __name__ == "__main__":
    main()
