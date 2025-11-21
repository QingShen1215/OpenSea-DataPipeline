"""
Batch BAYC Image Analysis with AVIF to PNG Conversion
Automatically converts AVIF images to PNG and analyzes with GPT-4o Vision API
"""

import os
import json
import base64
import subprocess
from pathlib import Path
from openai import OpenAI
import time
from PIL import Image
import io

# Initialize OpenAI client
# Get API key from environment variable for security
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable. Run: export OPENAI_API_KEY='your-key-here'")
client = OpenAI(api_key=OPENAI_API_KEY)

# Paths
IMAGES_DIR = '/Users/qingshen/Desktop/opensea/bayc_images'
TEMP_DIR = '/Users/qingshen/Desktop/opensea/bayc_images/converted_temp'
OUTPUT_FILE = '/Users/qingshen/Desktop/opensea/opensea_pipeline/bayc_metadata.json'
NUM_IMAGES = 200  # Number of images to process

def convert_avif_to_png(avif_path, png_path):
    """Convert AVIF image to PNG using ImageMagick"""
    try:
        result = subprocess.run(
            ['magick', str(avif_path), '-format', 'png', str(png_path)],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Conversion failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("  ✗ ImageMagick not found. Please install with: brew install imagemagick")
        return False

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_bayc_image(image_path, token_id):
    """
    Analyze a BAYC image using GPT-4o Vision API
    """
    print(f"  Analyzing with GPT-4o...")
    
    # Encode image
    base64_image = encode_image(image_path)
    
    # Create prompt for structured extraction
    prompt = """You are an expert at analyzing Bored Ape Yacht Club (BAYC) NFT images.

Analyze this BAYC image and extract the following attributes in JSON format:

{
  "token_id": "<token_id>",
  "background_color_hex": "<hex_code>",
  "background_color_name": "<color_name>",
  "fur_color": "<color_description>",
  "fur_type": "<type_description>",
  "face_expression": "<expression_description>",
  "eye_state": "<open/closed/tired/etc>",
  "headgear_type": "<hat/crown/bandana/none/etc>",
  "eyewear_type": "<sunglasses/3D_glasses/none/etc>"
}

Guidelines:
- background_color_hex: Provide the hex code (e.g., #FF6B6B)
- background_color_name: Common color name (e.g., "Orange", "Blue", "Gray")
- fur_color: Describe the ape's fur color (e.g., "Brown", "Golden Brown", "Gray")
- fur_type: Describe texture/pattern (e.g., "Normal", "Dark Brown", "Cream", "Robot")
- face_expression: Overall emotional expression (e.g., "Bored", "Grin", "Rage")
- eye_state: Eye appearance (e.g., "Open", "Closed", "Sleepy", "Laser Eyes")
- headgear_type: What's on the head (e.g., "Fisherman's Hat", "Crown", "None")
- eyewear_type: Eye accessories (e.g., "Sunglasses", "3D Glasses", "None")

Return ONLY the JSON object, no additional text."""
    
    try:
        # Call GPT-4o Vision API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        # Extract JSON response
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        result = json.loads(content)
        result['token_id'] = token_id
        
        print(f"  ✓ Background: {result.get('background_color_name', 'N/A')} ({result.get('background_color_hex', 'N/A')})")
        print(f"  ✓ Fur: {result.get('fur_color', 'N/A')} - {result.get('fur_type', 'N/A')}")
        print(f"  ✓ Expression: {result.get('face_expression', 'N/A')} (Eyes: {result.get('eye_state', 'N/A')})")
        print(f"  ✓ Headgear: {result.get('headgear_type', 'N/A')}")
        print(f"  ✓ Eyewear: {result.get('eyewear_type', 'N/A')}")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return {
            "token_id": token_id,
            "error": str(e),
            "background_color_hex": None,
            "background_color_name": None,
            "fur_color": None,
            "fur_type": None,
            "face_expression": None,
            "eye_state": None,
            "headgear_type": None,
            "eyewear_type": None
        }

def main():
    """Main function to batch analyze BAYC images"""
    print("="*70)
    print("Batch BAYC Image Analysis with Auto-Conversion")
    print("="*70)
    
    # Create temp directory for converted images
    temp_path = Path(TEMP_DIR)
    temp_path.mkdir(exist_ok=True)
    print(f"\nTemp directory: {TEMP_DIR}")
    
    # Get all image files in the directory
    images_path = Path(IMAGES_DIR)
    if not images_path.exists():
        print(f"\n❌ Error: Directory not found: {IMAGES_DIR}")
        return
    
    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(images_path.glob(ext))
    
    # Sort and limit to NUM_IMAGES
    image_files = sorted(image_files)[:NUM_IMAGES]
    
    if not image_files:
        print(f"\n❌ Error: No image files found in {IMAGES_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Output will be saved to: {OUTPUT_FILE}\n")
    
    # Process each image
    results = []
    for idx, image_path in enumerate(image_files, 1):
        # Extract token ID from filename
        filename = image_path.stem
        token_id = ''.join(filter(str.isdigit, filename))
        if not token_id:
            token_id = f"unknown_{idx}"
        
        print(f"\n[{idx}/{len(image_files)}] Processing BAYC #{token_id} ({image_path.name})")
        
        # Convert AVIF to PNG
        converted_path = temp_path / f"{token_id}_converted.png"
        print(f"  Converting AVIF to PNG...")
        
        if not convert_avif_to_png(image_path, converted_path):
            results.append({
                "token_id": token_id,
                "filename": image_path.name,
                "error": "Conversion failed"
            })
            continue
        
        print(f"  ✓ Converted to {converted_path.name}")
        
        # Analyze with GPT-4o
        result = analyze_bayc_image(str(converted_path), token_id)
        result['filename'] = image_path.name
        results.append(result)
        
        # Rate limiting: wait 1 second between requests
        if idx < len(image_files):
            print(f"  ⏱️  Waiting 1 second...")
            time.sleep(1)
    
    # Save results to JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nProcessed: {len(results)} images")
    print(f"Results saved to: {OUTPUT_FILE}")
    
    # Display summary
    print("\n📊 Summary:")
    successful = sum(1 for r in results if 'error' not in r or not r['error'])
    failed = len(results) - successful
    print(f"  ✓ Successful: {successful}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")
    
    # Show sample results
    if results and 'error' not in results[0]:
        print("\n📝 Sample Results (First 3 Images):")
        for i, sample in enumerate(results[:3], 1):
            if 'error' in sample and sample['error']:
                continue
            print(f"\n  [{i}] BAYC #{sample.get('token_id', 'N/A')}")
            print(f"      Background: {sample.get('background_color_name', 'N/A')} ({sample.get('background_color_hex', 'N/A')})")
            print(f"      Fur: {sample.get('fur_color', 'N/A')} - {sample.get('fur_type', 'N/A')}")
            print(f"      Expression: {sample.get('face_expression', 'N/A')} (Eyes: {sample.get('eye_state', 'N/A')})")
            print(f"      Headgear: {sample.get('headgear_type', 'N/A')}")
            print(f"      Eyewear: {sample.get('eyewear_type', 'N/A')}")
    
    print(f"\n💡 Tip: Converted PNG files are in: {TEMP_DIR}")
    print(f"💰 Estimated cost: ${len(results) * 0.02:.2f} USD (approx $0.02 per image)")

if __name__ == "__main__":
    main()
