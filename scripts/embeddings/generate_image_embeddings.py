"""
Generate Image Embeddings for BAYC NFTs
Multiple methods: CLIP, ResNet, OpenAI Vision API
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

# Method 1: CLIP (Recommended - Best for semantic understanding)
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")

# Method 2: OpenAI API Embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not installed. Run: pip install openai")

# Method 3: ResNet (Traditional CNN features)
try:
    import torchvision.models as models
    import torchvision.transforms as transforms
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False
    print("⚠️  torchvision not installed. Run: pip install torchvision")

# Paths
IMAGES_DIR = '/Users/qingshen/Desktop/opensea/bayc_images/converted_temp'
METADATA_FILE = '/Users/qingshen/Desktop/opensea/opensea_pipeline/bayc_metadata.json'
OUTPUT_DIR = '/Users/qingshen/Desktop/opensea/opensea_pipeline/embeddings'

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

class ImageEmbedder:
    def __init__(self, method='clip'):
        """
        Initialize embedder
        
        Args:
            method: 'clip', 'openai', or 'resnet'
        """
        self.method = method
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if method == 'clip':
            self._init_clip()
        elif method == 'openai':
            self._init_openai()
        elif method == 'resnet':
            self._init_resnet()
    
    def _init_clip(self):
        """Initialize CLIP model"""
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not available")
        
        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        print("✓ CLIP model loaded (embedding dim: 512)")
    
    def _init_openai(self):
        """Initialize OpenAI API"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available")
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable. Run: export OPENAI_API_KEY='your-key-here'")
        self.client = OpenAI(api_key=api_key)
        print("✓ OpenAI API initialized")
    
    def _init_resnet(self):
        """Initialize ResNet model"""
        if not RESNET_AVAILABLE:
            raise ImportError("torchvision not available")
        
        print("Loading ResNet50 model...")
        self.model = models.resnet50(pretrained=True)
        # Remove final classification layer to get features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("✓ ResNet50 model loaded (embedding dim: 2048)")
    
    def embed_image_clip(self, image_path):
        """Generate CLIP embedding"""
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    
    def embed_image_openai(self, image_path):
        """Generate OpenAI embedding (requires API call)"""
        import base64
        
        # Encode image
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Note: OpenAI doesn't have a direct image embedding API yet
        # This uses the vision model to generate a description, then embeds the text
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail in one sentence."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            max_tokens=100
        )
        
        description = response.choices[0].message.content
        
        # Get text embedding
        embedding_response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=description
        )
        
        return np.array(embedding_response.data[0].embedding)
    
    def embed_image_resnet(self, image_path):
        """Generate ResNet embedding"""
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image_input)
        
        return features.cpu().numpy().flatten()
    
    def embed_image(self, image_path):
        """Generate embedding based on selected method"""
        if self.method == 'clip':
            return self.embed_image_clip(image_path)
        elif self.method == 'openai':
            return self.embed_image_openai(image_path)
        elif self.method == 'resnet':
            return self.embed_image_resnet(image_path)

def load_metadata():
    """Load BAYC metadata"""
    with open(METADATA_FILE, 'r') as f:
        return json.load(f)

def generate_embeddings(method='clip', limit=None):
    """
    Generate embeddings for all BAYC images
    
    Args:
        method: 'clip' (recommended), 'openai', or 'resnet'
        limit: Maximum number of images to process (None for all)
    """
    print("="*70)
    print(f"Generating Image Embeddings - Method: {method.upper()}")
    print("="*70)
    
    # Load metadata
    metadata = load_metadata()
    print(f"\nLoaded metadata for {len(metadata)} images")
    
    if limit:
        metadata = metadata[:limit]
        print(f"Processing first {limit} images")
    
    # Initialize embedder
    embedder = ImageEmbedder(method=method)
    
    # Generate embeddings
    embeddings_data = []
    images_dir = Path(IMAGES_DIR)
    
    print("\nGenerating embeddings...")
    for item in tqdm(metadata):
        token_id = item['token_id']
        
        # Find converted image
        image_path = images_dir / f"{token_id}_converted.png"
        
        if not image_path.exists():
            print(f"  ⚠️  Image not found: {image_path}")
            continue
        
        try:
            # Generate embedding
            embedding = embedder.embed_image(str(image_path))
            
            embeddings_data.append({
                'token_id': token_id,
                'embedding': embedding.tolist(),
                'embedding_dim': len(embedding),
                'method': method
            })
            
        except Exception as e:
            print(f"  ✗ Error processing {token_id}: {e}")
            continue
    
    # Save embeddings
    output_file = Path(OUTPUT_DIR) / f'bayc_embeddings_{method}.json'
    with open(output_file, 'w') as f:
        json.dump(embeddings_data, f, indent=2)
    
    # Also save as numpy array for faster loading
    embeddings_array = np.array([item['embedding'] for item in embeddings_data])
    token_ids = [item['token_id'] for item in embeddings_data]
    
    np.savez(
        Path(OUTPUT_DIR) / f'bayc_embeddings_{method}.npz',
        embeddings=embeddings_array,
        token_ids=token_ids
    )
    
    print("\n" + "="*70)
    print("Embeddings Generated!")
    print("="*70)
    print(f"\n✓ Processed: {len(embeddings_data)} images")
    print(f"✓ Embedding dimension: {embeddings_data[0]['embedding_dim']}")
    print(f"\nOutput files:")
    print(f"  JSON: {output_file}")
    print(f"  NumPy: {Path(OUTPUT_DIR) / f'bayc_embeddings_{method}.npz'}")
    
    return embeddings_data

def find_similar_images(query_token_id, embeddings_file, top_k=5):
    """
    Find most similar images to a query image
    
    Args:
        query_token_id: Token ID of query image
        embeddings_file: Path to .npz embeddings file
        top_k: Number of similar images to return
    """
    # Load embeddings
    data = np.load(embeddings_file)
    embeddings = data['embeddings']
    token_ids = data['token_ids']
    
    # Find query embedding
    query_idx = np.where(token_ids == str(query_token_id))[0]
    if len(query_idx) == 0:
        print(f"Token {query_token_id} not found in embeddings")
        return []
    
    query_idx = query_idx[0]
    query_embedding = embeddings[query_idx]
    
    # Compute cosine similarities
    similarities = np.dot(embeddings, query_embedding)
    
    # Get top-k (excluding query itself)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    results = []
    for idx in top_indices:
        results.append({
            'token_id': token_ids[idx],
            'similarity': float(similarities[idx])
        })
    
    return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate BAYC image embeddings')
    parser.add_argument('--method', choices=['clip', 'openai', 'resnet'], default='clip',
                        help='Embedding method (default: clip)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process')
    parser.add_argument('--find-similar', type=str, default=None,
                        help='Find similar images to a token ID')
    
    args = parser.parse_args()
    
    if args.find_similar:
        # Find similar images
        embeddings_file = Path(OUTPUT_DIR) / f'bayc_embeddings_{args.method}.npz'
        if not embeddings_file.exists():
            print(f"Embeddings file not found: {embeddings_file}")
            print("Run without --find-similar first to generate embeddings")
            return
        
        print(f"\nFinding images similar to BAYC #{args.find_similar}...")
        similar = find_similar_images(args.find_similar, embeddings_file)
        
        print(f"\nTop {len(similar)} most similar images:")
        for i, result in enumerate(similar, 1):
            print(f"  {i}. BAYC #{result['token_id']} (similarity: {result['similarity']:.4f})")
    else:
        # Generate embeddings
        generate_embeddings(method=args.method, limit=args.limit)

if __name__ == "__main__":
    main()
