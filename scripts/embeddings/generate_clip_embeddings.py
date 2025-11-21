#!/usr/bin/env python3
"""
Generate CLIP embeddings for BAYC NFTs

CLIP provides high-quality 512-dimensional embeddings perfect for 
similarity search and clustering.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import clip
    import torch
except ImportError:
    print("❌ Error: CLIP not installed.")
    print("Please run: bash install_embedding_deps.sh")
    sys.exit(1)


class CLIPEmbedder:
    """Generate embeddings using CLIP model"""
    
    def __init__(self):
        """Initialize CLIP model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        print("📦 Loading CLIP model (ViT-B/32)...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        print("✓ CLIP model loaded successfully\n")
    
    def generate_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate CLIP embedding for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            512-dimensional embedding vector
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # Normalize to unit vector for cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    
    def batch_generate(self, image_paths: List[str]) -> List[Dict]:
        """
        Generate embeddings for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of dicts with token_id and embedding
        """
        embeddings = []
        
        for img_path in tqdm(image_paths, desc="Generating embeddings"):
            try:
                # Extract token ID from filename (e.g., "0_converted.png" -> "0")
                filename = Path(img_path).stem
                token_id = filename.replace("_converted", "")
                
                # Generate embedding
                embedding = self.generate_embedding(img_path)
                
                embeddings.append({
                    "token_id": token_id,
                    "embedding": embedding.tolist(),
                    "embedding_dim": len(embedding),
                    "method": "clip"
                })
            except Exception as e:
                print(f"\n⚠️  Error processing {img_path}: {e}")
                continue
        
        return embeddings


def find_similar_images(
    embeddings: np.ndarray,
    token_ids: List[str],
    query_token_id: str,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find most similar images using cosine similarity
    
    Args:
        embeddings: Array of embeddings (N, embedding_dim)
        token_ids: List of token IDs corresponding to embeddings
        query_token_id: Token ID to find similar images for
        top_k: Number of similar images to return
        
    Returns:
        List of (token_id, similarity_score) tuples
    """
    # Find query embedding
    try:
        query_idx = token_ids.index(str(query_token_id))
    except ValueError:
        print(f"❌ Token {query_token_id} not found in embeddings")
        return []
    
    query_embedding = embeddings[query_idx:query_idx+1]
    
    # Compute cosine similarity (embeddings are already normalized)
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    
    # Get top-k (excluding query itself)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    results = [(token_ids[idx], similarities[idx]) for idx in top_indices]
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP embeddings for BAYC NFTs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images
  python generate_clip_embeddings.py
  
  # Process first 50 images
  python generate_clip_embeddings.py --limit 50
  
  # Find images similar to BAYC #0
  python generate_clip_embeddings.py --find-similar 0
        """
    )
    parser.add_argument("--images-dir", type=str, 
                       default="/Users/qingshen/Desktop/opensea/bayc_images/converted_temp",
                       help="Directory containing converted images")
    parser.add_argument("--output-dir", type=str, 
                       default="embeddings",
                       help="Directory to save embeddings")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of images to process")
    parser.add_argument("--find-similar", type=str, default=None,
                       help="Find images similar to this token ID")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize CLIP embedder
    embedder = CLIPEmbedder()
    
    # Get list of images
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        sys.exit(1)
    
    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        print(f"❌ No PNG images found in {images_dir}")
        sys.exit(1)
    
    if args.limit:
        image_files = image_files[:args.limit]
    
    print(f"📊 Found {len(image_files)} images to process\n")
    
    # Generate embeddings
    embeddings_list = embedder.batch_generate(image_files)
    
    if not embeddings_list:
        print("❌ No embeddings generated")
        sys.exit(1)
    
    # Save as JSON
    output_json = output_dir / "bayc_embeddings_clip.json"
    with open(output_json, 'w') as f:
        json.dump(embeddings_list, f, indent=2)
    print(f"\n💾 Saved JSON: {output_json}")
    
    # Save as NumPy for efficient loading
    embeddings_array = np.array([item['embedding'] for item in embeddings_list])
    token_ids = [item['token_id'] for item in embeddings_list]
    
    output_npz = output_dir / "bayc_embeddings_clip.npz"
    np.savez_compressed(
        output_npz,
        embeddings=embeddings_array,
        token_ids=token_ids,
        method="clip",
        embedding_dim=embeddings_array.shape[1]
    )
    print(f"💾 Saved NumPy: {output_npz}")
    
    # Find similar images if requested
    if args.find_similar is not None:
        print(f"\n🔍 Finding images similar to token {args.find_similar}...")
        similar = find_similar_images(
            embeddings_array, 
            token_ids, 
            args.find_similar,
            top_k=5
        )
        
        if similar:
            print(f"\n✨ Top 5 most similar images to BAYC #{args.find_similar}:")
            for token_id, score in similar:
                print(f"   BAYC #{token_id}: {score:.4f}")
    
    print("\n" + "="*50)
    print("✓ Embedding generation complete!")
    print("="*50)
    print(f"  Method: CLIP (ViT-B/32)")
    print(f"  Images processed: {len(embeddings_list)}")
    print(f"  Embedding dimension: 512")
    print(f"  Output directory: {output_dir}/")
    print(f"\n💡 Next steps:")
    print(f"  • Find similar images:")
    print(f"    python generate_clip_embeddings.py --find-similar 0")
    print(f"  • Load embeddings in Python:")
    print(f"    data = np.load('embeddings/bayc_embeddings_clip.npz')")
    print(f"    embeddings = data['embeddings']  # Shape: ({len(embeddings_list)}, 512)")


if __name__ == "__main__":
    main()
