#!/usr/bin/env python3
"""
Analyze BAYC image embeddings - Examples and use cases
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_embeddings(embeddings_path="embeddings/bayc_embeddings_clip.npz"):
    """Load embeddings from .npz file"""
    print(f"📂 Loading embeddings from {embeddings_path}...")
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    token_ids = data['token_ids']
    print(f"✓ Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]})\n")
    return embeddings, token_ids


def find_similar(embeddings, token_ids, query_id, top_k=10):
    """Find most similar images to a given token ID"""
    try:
        query_idx = list(token_ids).index(str(query_id))
    except ValueError:
        print(f"❌ Token {query_id} not found")
        return []
    
    query_embedding = embeddings[query_idx:query_idx+1]
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k (excluding query itself)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    results = [(token_ids[idx], similarities[idx]) for idx in top_indices]
    return results


def cluster_analysis(embeddings, token_ids, n_clusters=10):
    """Cluster images by visual similarity"""
    print(f"🔬 Clustering {len(embeddings)} images into {n_clusters} groups...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    # Print cluster distribution
    print(f"\n📊 Cluster distribution:")
    unique, counts = np.unique(clusters, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"   Cluster {cluster_id}: {count} images ({count/len(embeddings)*100:.1f}%)")
        # Show sample token IDs from this cluster
        cluster_tokens = [token_ids[i] for i, c in enumerate(clusters) if c == cluster_id]
        print(f"      Samples: {', '.join(['#'+str(t) for t in cluster_tokens[:5]])}")
    
    return clusters


def visualize_embeddings(embeddings, token_ids, clusters=None, output_path="visualization_result/embeddings_tsne.png"):
    """Visualize embeddings using t-SNE"""
    print(f"\n🎨 Creating t-SNE visualization...")
    
    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(14, 10))
    
    if clusters is not None:
        # Color by cluster
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=clusters, cmap='tab10', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        title = f'BAYC Visual Clusters (t-SNE projection)'
    else:
        # Single color
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   alpha=0.6, s=50, color='steelblue')
        title = 'BAYC Embeddings (t-SNE projection)'
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE dimension 1', fontsize=12)
    plt.ylabel('t-SNE dimension 2', fontsize=12)
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


def similarity_heatmap(embeddings, token_ids, sample_size=20, output_path="visualization_result/similarity_heatmap.png"):
    """Create similarity heatmap for sample images"""
    print(f"\n🔥 Creating similarity heatmap (sample of {sample_size} images)...")
    
    # Sample random images
    indices = np.random.choice(len(embeddings), size=min(sample_size, len(embeddings)), replace=False)
    sample_embeddings = embeddings[indices]
    sample_ids = [token_ids[i] for i in indices]
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(sample_embeddings)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, 
                xticklabels=[f"#{tid}" for tid in sample_ids],
                yticklabels=[f"#{tid}" for tid in sample_ids],
                cmap='coolwarm', 
                center=0.8,
                vmin=0.6, 
                vmax=1.0,
                square=True,
                cbar_kws={'label': 'Cosine Similarity'})
    
    plt.title('BAYC Image Similarity Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap: {output_path}")
    plt.close()


def combine_with_metadata(embeddings, token_ids, metadata_path="bayc_metadata.json"):
    """Combine embeddings with metadata for analysis"""
    print(f"\n🔗 Combining embeddings with metadata...")
    
    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Create mapping
    metadata_dict = {str(item['token_id']): item for item in metadata}
    
    # Combine
    combined = []
    for i, token_id in enumerate(token_ids):
        if str(token_id) in metadata_dict:
            item = metadata_dict[str(token_id)].copy()
            item['embedding'] = embeddings[i].tolist()
            item['embedding_norm'] = float(np.linalg.norm(embeddings[i]))
            combined.append(item)
    
    print(f"✓ Combined {len(combined)} images with metadata")
    
    # Analyze by background color
    backgrounds = {}
    for item in combined:
        bg = item.get('background_color_name', 'Unknown')
        if bg not in backgrounds:
            backgrounds[bg] = []
        backgrounds[bg].append(item['embedding'])
    
    print(f"\n🎨 Background color distribution:")
    for bg, embs in sorted(backgrounds.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        avg_embedding = np.mean(embs, axis=0)
        print(f"   {bg}: {len(embs)} images (avg norm: {np.linalg.norm(avg_embedding):.4f})")
    
    return combined


def main():
    print("="*60)
    print("BAYC Image Embeddings - Analysis")
    print("="*60)
    print()
    
    # Load embeddings
    embeddings, token_ids = load_embeddings()
    
    # Example 1: Find similar images
    print("="*60)
    print("Example 1: Find Similar Images")
    print("="*60)
    query_id = 0
    similar = find_similar(embeddings, token_ids, query_id, top_k=5)
    print(f"\n🔍 Top 5 images similar to BAYC #{query_id}:")
    for token_id, score in similar:
        print(f"   BAYC #{token_id}: {score:.4f}")
    
    # Example 2: Clustering
    print("\n" + "="*60)
    print("Example 2: Cluster Analysis")
    print("="*60)
    clusters = cluster_analysis(embeddings, token_ids, n_clusters=8)
    
    # Example 3: Visualizations
    print("\n" + "="*60)
    print("Example 3: Visualizations")
    print("="*60)
    visualize_embeddings(embeddings, token_ids, clusters)
    similarity_heatmap(embeddings, token_ids, sample_size=30)
    
    # Example 4: Combine with metadata
    print("\n" + "="*60)
    print("Example 4: Metadata Integration")
    print("="*60)
    try:
        combined = combine_with_metadata(embeddings, token_ids)
        
        # Save combined data
        output_path = "embeddings/bayc_embeddings_with_metadata.json"
        with open(output_path, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"✓ Saved combined data: {output_path}")
    except FileNotFoundError:
        print("⚠️  Metadata file not found, skipping metadata integration")
    
    print("\n" + "="*60)
    print("✓ Analysis Complete!")
    print("="*60)
    print("\n💡 Next steps:")
    print("  • Use embeddings for recommendation systems")
    print("  • Predict NFT prices based on visual features")
    print("  • Discover rare visual patterns")
    print("  • Build visual search engine")


if __name__ == "__main__":
    main()
