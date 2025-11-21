"""
BAYC Image Analysis using GPT-4o Vision API
Extracts visual attributes from BAYC NFT images
"""

import os
import json
import base64
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
OUTPUT_FILE = '/Users/qingshen/Desktop/opensea/opensea_pipeline/bayc_metadata.json'
CONVERTED_IMAGE = '/Users/qingshen/Desktop/opensea/bayc_images/0_converted.png'  # Use converted PNG

def encode_image(image_path):
    """Encode image to base64 string, converting AVIF to PNG if needed"""
    try:
        # Try to open with PIL and convert to PNG
        img = Image.open(image_path)
        
        # Convert to RGB if necessary (remove alpha channel)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to bytes buffer as PNG
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"  ! Warning: Could not convert image, trying direct encoding: {e}")
        # Fallback to direct encoding
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_bayc_image(image_path, token_id):
    """
    Analyze a BAYC image using GPT-4o Vision API
    
    Args:
        image_path: Path to the image file
        token_id: BAYC token ID
        
    Returns:
        Dictionary with extracted attributes
    """
    print(f"Analyzing BAYC #{token_id}...")
    
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
- fur_type: Describe texture/pattern (e.g., "Normal", "Dark Brown", "Cream")
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
        # Sometimes the model wraps JSON in ```json blocks
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        result = json.loads(content)
        result['token_id'] = token_id
        
        print(f"  ✓ Background: {result.get('background_color_name', 'N/A')}")
        print(f"  ✓ Fur: {result.get('fur_color', 'N/A')} - {result.get('fur_type', 'N/A')}")
        print(f"  ✓ Expression: {result.get('face_expression', 'N/A')}")
        print(f"  ✓ Headgear: {result.get('headgear_type', 'N/A')}")
        print(f"  ✓ Eyewear: {result.get('eyewear_type', 'N/A')}")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error analyzing BAYC #{token_id}: {str(e)}")
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
    """Main function to analyze BAYC images"""
    print("="*70)
    print("BAYC Image Analysis using GPT-4o Vision API")
    print("="*70)
    
    # Get all image files in the directory
    images_path = Path(IMAGES_DIR)
    if not images_path.exists():
        print(f"\n❌ Error: Directory not found: {IMAGES_DIR}")
        return
    
    # Find all image files (png, jpg, jpeg)
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(images_path.glob(ext))
    
    image_files = sorted(image_files)[:1]  # Get first 1 image for testing
    
    if not image_files:
        print(f"\n❌ Error: No image files found in {IMAGES_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images to analyze")
    print(f"Output will be saved to: {OUTPUT_FILE}\n")
    
    # Analyze each image
    results = []
    
    # For now, just test with the converted image
    image_path = CONVERTED_IMAGE
    token_id = "0"
    
    print(f"\n[1/1] Processing: {Path(image_path).name}")
    
    result = analyze_bayc_image(str(image_path), token_id)
    result['filename'] = Path(image_path).name
    results.append(result)
    
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
    
    # Show sample result
    if results and 'error' not in results[0]:
        print("\n📝 Sample Result (First Image):")
        sample = results[0]
        print(f"  Token ID: {sample.get('token_id', 'N/A')}")
        print(f"  Background: {sample.get('background_color_name', 'N/A')} ({sample.get('background_color_hex', 'N/A')})")
        print(f"  Fur: {sample.get('fur_color', 'N/A')} - {sample.get('fur_type', 'N/A')}")
        print(f"  Expression: {sample.get('face_expression', 'N/A')} (Eyes: {sample.get('eye_state', 'N/A')})")
        print(f"  Headgear: {sample.get('headgear_type', 'N/A')}")
        print(f"  Eyewear: {sample.get('eyewear_type', 'N/A')}")

if __name__ == "__main__":
    main()
