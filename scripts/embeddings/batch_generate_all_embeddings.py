#!/usr/bin/env python3
"""
Convert all BAYC images and generate CLIP embeddings
Uses parallel processing for faster conversion
"""

import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Directories
ORIGINAL_DIR = Path("/Users/qingshen/Desktop/opensea/bayc_images")
CONVERTED_DIR = ORIGINAL_DIR / "converted_temp"

def convert_image(img_path):
    """Convert single AVIF to PNG using ImageMagick"""
    try:
        output_path = CONVERTED_DIR / f"{img_path.stem}_converted.png"
        
        # Skip if already exists
        if output_path.exists():
            return True, img_path.stem
        
        result = subprocess.run(
            ['magick', str(img_path), str(output_path)],
            capture_output=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        )
        return result.returncode == 0, img_path.stem
    except Exception as e:
        return False, img_path.stem

def convert_all_images_parallel():
    """Convert all BAYC images using parallel processing"""
    print("="*60)
    print("Converting BAYC Images (AVIF → PNG) - Parallel Mode")
    print("="*60)
    
    # Create output directory
    CONVERTED_DIR.mkdir(exist_ok=True)
    
    # Get list of all images
    all_images = sorted(ORIGINAL_DIR.glob("*.png"))
    print(f"\n📊 Total images: {len(all_images)}")
    
    # Check existing conversions
    existing = set(f.stem.replace('_converted', '') for f in CONVERTED_DIR.glob("*_converted.png"))
    print(f"✓ Already converted: {len(existing)}")
    
    # Filter to only unconverted images
    to_convert = [img for img in all_images if img.stem not in existing]
    
    if not to_convert:
        print("\n✓ All images already converted!")
        return len(all_images)
    
    print(f"🔄 Need to convert: {len(to_convert)}")
    
    # Use half of CPU cores for conversion
    n_workers = max(1, multiprocessing.cpu_count() // 2)
    print(f"🚀 Using {n_workers} parallel workers")
    print(f"⏱️  Estimated time: ~{len(to_convert) * 0.1 / n_workers:.0f} seconds ({len(to_convert) * 0.1 / n_workers / 60:.1f} minutes)\n")
    
    # Convert in parallel
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(convert_image, img): img for img in to_convert}
        
        # Process results with progress bar
        with tqdm(total=len(to_convert), desc="Converting") as pbar:
            for future in as_completed(futures):
                success, token_id = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                pbar.update(1)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total converted: {len(existing) + success_count}")
    
    return len(existing) + success_count

def main():
    print("="*60)
    print("BAYC Full Collection - CLIP Embeddings")
    print("="*60)
    print()
    
    # Step 1: Convert all images (parallel)
    start_time = time.time()
    total_images = convert_all_images_parallel()
    conversion_time = time.time() - start_time
    
    print(f"\n⏱️  Conversion took {conversion_time:.1f} seconds ({conversion_time/60:.1f} minutes)")
    
    print("\n" + "="*60)
    print("Generating CLIP Embeddings")
    print("="*60)
    print(f"\n🚀 Processing {total_images} images...")
    print(f"⏱️  Estimated time: ~{total_images * 0.06:.0f} seconds ({total_images * 0.06 / 60:.1f} minutes)")
    print("   (This may take 10-15 minutes for ~10k images)\n")
    
    # Step 2: Generate embeddings
    embed_start = time.time()
    cmd = [
        "python", 
        "generate_clip_embeddings.py",
        "--images-dir", str(CONVERTED_DIR)
    ]
    
    result = subprocess.run(cmd)
    embed_time = time.time() - embed_start
    
    total_time = time.time() - start_time
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ Complete!")
        print("="*60)
        print(f"\n📊 Processed {total_images} images")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"   - Conversion: {conversion_time/60:.1f} minutes")
        print(f"   - Embeddings: {embed_time/60:.1f} minutes")
        print(f"\n📁 Output files:")
        print(f"   - embeddings/bayc_embeddings_clip.npz")
        print(f"   - embeddings/bayc_embeddings_clip.json")
        
        # Check file sizes
        npz_file = Path("embeddings/bayc_embeddings_clip.npz")
        if npz_file.exists():
            size_mb = npz_file.stat().st_size / 1024 / 1024
            print(f"\n💾 NPZ file size: {size_mb:.1f} MB")
    else:
        print("\n❌ Error generating embeddings")
        sys.exit(1)

if __name__ == "__main__":
    main()
