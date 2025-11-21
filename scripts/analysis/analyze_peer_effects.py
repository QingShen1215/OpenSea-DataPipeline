"""
BAYC Peer Effects Analysis: Visual Similarity → Price & Timing Dynamics
======================================================================

Research Question 3: When visually similar apes sell high/fast, does it affect your price/timing?

Analysis:
- Identify K nearest neighbors in embedding space
- Track peer sales in time windows (7d, 30d before/after)
- Regression: peer_price → your_price, peer_sales → your_time_to_sell
- Causal inference: Does peer activity drive your outcomes?
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import json
from datetime import timedelta

# Paths
DATA_DIR = Path("visual_market_analysis")
EMBEDDINGS_PATH = DATA_DIR / "complete_embeddings.npz"
TRANSACTIONS_PATH = DATA_DIR / "transactions_with_cycles.parquet"
NFT_METADATA_PATH = DATA_DIR / "nft_metadata_with_clusters.parquet"
OUTPUT_DIR = DATA_DIR / "peer_effects"
OUTPUT_DIR.mkdir(exist_ok=True)

# Analysis parameters
K_NEIGHBORS = [10, 20, 50]  # Number of nearest neighbors to analyze
TIME_WINDOWS = [7, 30]  # Days before/after to look for peer effects

def load_data():
    """Load embeddings and transaction data"""
    print("📂 Loading Data...")
    print("-" * 70)
    
    # Load embeddings
    embeddings_data = np.load(EMBEDDINGS_PATH)
    embeddings = embeddings_data['embeddings']
    token_ids = embeddings_data['token_ids']
    
    # Load transactions and NFT metadata
    df_txns = pl.read_parquet(TRANSACTIONS_PATH)
    df_nfts = pl.read_parquet(NFT_METADATA_PATH)
    
    print(f"  ✓ Loaded embeddings for {len(embeddings)} NFTs (dim={embeddings.shape[1]})")
    print(f"  ✓ Loaded {len(df_txns)} transactions")
    print(f"  ✓ Loaded {len(df_nfts)} NFT metadata records")
    
    return embeddings, token_ids, df_txns, df_nfts

def compute_similarity_matrix(embeddings, token_ids):
    """
    Compute cosine similarity matrix and find K nearest neighbors for each NFT
    """
    print("\n🔍 Computing Visual Similarity Matrix...")
    print("-" * 70)
    
    # Compute cosine similarity matrix
    print(f"  Computing {len(embeddings)} x {len(embeddings)} similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # For each NFT, find K nearest neighbors
    neighbors_dict = {}
    
    for k in K_NEIGHBORS:
        print(f"  Finding top {k} nearest neighbors for each NFT...")
        neighbors = {}
        
        for i, token_id in enumerate(token_ids):
            # Get similarity scores for this NFT
            similarities = similarity_matrix[i]
            
            # Find indices of top K+1 (including itself)
            top_k_indices = np.argsort(similarities)[::-1][:k+1]
            
            # Exclude itself
            top_k_indices = top_k_indices[top_k_indices != i][:k]
            
            # Store neighbor token_ids and similarity scores
            neighbors[str(token_id)] = {
                'neighbor_ids': [str(token_ids[idx]) for idx in top_k_indices],
                'similarities': [float(similarities[idx]) for idx in top_k_indices]
            }
        
        neighbors_dict[f'k{k}'] = neighbors
    
    # Save neighbors
    with open(OUTPUT_DIR / "visual_neighbors.json", "w") as f:
        json.dump(neighbors_dict, f)
    
    print(f"  ✓ Saved: visual_neighbors.json")
    
    return neighbors_dict

def prepare_sales_data(df_txns):
    """
    Prepare sales data with time-to-next-sale calculation
    """
    print("\n📊 Preparing Sales Data...")
    print("-" * 70)
    
    # Filter to sales only
    df_sales = df_txns.filter(
        (pl.col('event_type') == 'sale') &
        (pl.col('price_each').is_not_null()) &
        (pl.col('price_each') > 0)
    ).sort(['token_id_str', 'datetime'])
    
    # Calculate time to next sale for each NFT
    df_sales = df_sales.with_columns([
        pl.col('datetime').shift(-1).over('token_id_str').alias('next_sale_time'),
    ])
    
    # Calculate days to next sale
    df_sales = df_sales.with_columns([
        ((pl.col('next_sale_time') - pl.col('datetime')).dt.total_seconds() / 86400).alias('days_to_next_sale')
    ])
    
    print(f"  ✓ Prepared {len(df_sales)} sales records")
    print(f"  ✓ {df_sales.filter(pl.col('days_to_next_sale').is_not_null()).height} sales with next-sale timing")
    
    return df_sales

def analyze_peer_effects_on_price(df_sales, neighbors_dict, k, window_days):
    """
    For each sale, look at peer sales in time window and analyze price correlation
    """
    print(f"\n💰 Analyzing Peer Effects on Price (K={k}, Window={window_days}d)...")
    print("-" * 70)
    
    neighbors = neighbors_dict[f'k{k}']
    df_sales_pd = df_sales.to_pandas()
    
    results = []
    
    for idx, row in df_sales_pd.iterrows():
        if idx % 5000 == 0:
            print(f"  Processing sale {idx}/{len(df_sales_pd)}...")
        
        token_id = row['token_id_str']
        sale_time = row['datetime']
        sale_price = row['price_each']
        
        # Skip if no neighbors
        if token_id not in neighbors:
            continue
        
        neighbor_ids = neighbors[token_id]['neighbor_ids']
        
        # Find peer sales in time window BEFORE this sale
        window_start = sale_time - pd.Timedelta(days=window_days)
        
        peer_sales = df_sales_pd[
            (df_sales_pd['token_id_str'].isin(neighbor_ids)) &
            (df_sales_pd['datetime'] >= window_start) &
            (df_sales_pd['datetime'] < sale_time)
        ]
        
        if len(peer_sales) == 0:
            continue
        
        # Aggregate peer metrics
        peer_avg_price = peer_sales['price_each'].mean()
        peer_median_price = peer_sales['price_each'].median()
        peer_max_price = peer_sales['price_each'].max()
        peer_num_sales = len(peer_sales)
        peer_unique_nfts = peer_sales['token_id_str'].nunique()
        
        results.append({
            'token_id': token_id,
            'sale_time': sale_time,
            'sale_price': sale_price,
            'market_cycle': row['market_cycle'],
            'visual_cluster': row.get('visual_cluster', np.nan),
            'peer_avg_price': peer_avg_price,
            'peer_median_price': peer_median_price,
            'peer_max_price': peer_max_price,
            'peer_num_sales': peer_num_sales,
            'peer_unique_nfts': peer_unique_nfts,
            'price_vs_peer_avg': sale_price / peer_avg_price if peer_avg_price > 0 else np.nan,
        })
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        print(f"  ⚠️ No peer effects data found for K={k}, Window={window_days}d")
        return None
    
    # Save raw results
    df_results.to_csv(OUTPUT_DIR / f"peer_price_effects_k{k}_w{window_days}d.csv", index=False)
    print(f"  ✓ Saved: peer_price_effects_k{k}_w{window_days}d.csv ({len(df_results)} sales)")
    
    # Regression analysis: peer_price → sale_price
    # Filter out extreme outliers
    df_reg = df_results[
        (df_results['sale_price'] < df_results['sale_price'].quantile(0.99)) &
        (df_results['peer_avg_price'] < df_results['peer_avg_price'].quantile(0.99)) &
        (df_results['peer_num_sales'] >= 2)  # At least 2 peer sales
    ].copy()
    
    if len(df_reg) < 10:
        print(f"  ⚠️ Not enough data for regression (n={len(df_reg)})")
        return df_results
    
    # Log transform prices for better regression fit
    df_reg['log_sale_price'] = np.log(df_reg['sale_price'])
    df_reg['log_peer_avg_price'] = np.log(df_reg['peer_avg_price'])
    
    # Linear regression
    from sklearn.linear_model import LinearRegression
    
    X = df_reg[['log_peer_avg_price', 'peer_num_sales']].values
    y = df_reg['log_sale_price'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Get statistics
    r_squared = model.score(X, y)
    coef_peer_price = model.coef_[0]
    coef_peer_volume = model.coef_[1]
    
    print(f"\n  📈 Regression Results:")
    print(f"     R² = {r_squared:.4f}")
    print(f"     Peer Price Coefficient = {coef_peer_price:.4f} (elasticity)")
    print(f"     Peer Volume Coefficient = {coef_peer_volume:.4f}")
    print(f"     Interpretation: 1% increase in peer price → {coef_peer_price:.2f}% change in your price")
    
    # Correlation
    corr_price = df_reg['sale_price'].corr(df_reg['peer_avg_price'])
    print(f"     Price Correlation = {corr_price:.4f}")
    
    return df_results

def analyze_peer_effects_on_timing(df_sales, neighbors_dict, k, window_days):
    """
    For each sale, look at peer sales BEFORE and analyze impact on time-to-next-sale
    """
    print(f"\n⏱️  Analyzing Peer Effects on Timing (K={k}, Window={window_days}d)...")
    print("-" * 70)
    
    neighbors = neighbors_dict[f'k{k}']
    df_sales_pd = df_sales.to_pandas()
    
    # Filter to sales with next-sale data
    df_with_next = df_sales_pd[df_sales_pd['days_to_next_sale'].notna()].copy()
    
    if len(df_with_next) == 0:
        print(f"  ⚠️ No timing data available")
        return None
    
    results = []
    
    for idx, row in df_with_next.iterrows():
        if idx % 2000 == 0:
            print(f"  Processing sale {idx}/{len(df_with_next)}...")
        
        token_id = row['token_id_str']
        sale_time = row['datetime']
        days_to_next = row['days_to_next_sale']
        
        # Skip if no neighbors
        if token_id not in neighbors:
            continue
        
        neighbor_ids = neighbors[token_id]['neighbor_ids']
        
        # Find peer sales in time window BEFORE this sale
        window_start = sale_time - pd.Timedelta(days=window_days)
        
        peer_sales = df_sales_pd[
            (df_sales_pd['token_id_str'].isin(neighbor_ids)) &
            (df_sales_pd['datetime'] >= window_start) &
            (df_sales_pd['datetime'] < sale_time)
        ]
        
        if len(peer_sales) == 0:
            continue
        
        # Aggregate peer timing metrics
        peer_avg_time_to_next = peer_sales['days_to_next_sale'].mean()
        peer_num_sales = len(peer_sales)
        peer_avg_price = peer_sales['price_each'].mean()
        
        results.append({
            'token_id': token_id,
            'sale_time': sale_time,
            'days_to_next_sale': days_to_next,
            'market_cycle': row['market_cycle'],
            'peer_num_sales': peer_num_sales,
            'peer_avg_price': peer_avg_price,
            'peer_avg_time_to_next': peer_avg_time_to_next,
        })
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        print(f"  ⚠️ No timing effects data found")
        return None
    
    # Save results
    df_results.to_csv(OUTPUT_DIR / f"peer_timing_effects_k{k}_w{window_days}d.csv", index=False)
    print(f"  ✓ Saved: peer_timing_effects_k{k}_w{window_days}d.csv ({len(df_results)} sales)")
    
    # Regression analysis
    df_reg = df_results[
        (df_results['days_to_next_sale'] < df_results['days_to_next_sale'].quantile(0.95)) &
        (df_results['peer_num_sales'] >= 2)
    ].copy()
    
    if len(df_reg) < 10:
        print(f"  ⚠️ Not enough data for regression")
        return df_results
    
    # Log transform for better fit
    df_reg['log_days_to_next'] = np.log(df_reg['days_to_next_sale'] + 1)
    df_reg['log_peer_num_sales'] = np.log(df_reg['peer_num_sales'])
    
    # Regression
    from sklearn.linear_model import LinearRegression
    
    X = df_reg[['log_peer_num_sales']].values
    y = df_reg['log_days_to_next'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    r_squared = model.score(X, y)
    coef = model.coef_[0]
    
    print(f"\n  📈 Regression Results:")
    print(f"     R² = {r_squared:.4f}")
    print(f"     Peer Sales Volume Coefficient = {coef:.4f}")
    print(f"     Interpretation: More peer sales → {'shorter' if coef < 0 else 'longer'} time to next sale")
    
    # Correlation
    corr = df_reg['days_to_next_sale'].corr(df_reg['peer_num_sales'])
    print(f"     Timing Correlation = {corr:.4f}")
    
    return df_results

def generate_peer_effects_report(neighbors_dict):
    """
    Generate comprehensive peer effects report
    """
    print("\n📝 Generating Peer Effects Report...")
    print("-" * 70)
    
    report = []
    report.append("=" * 80)
    report.append("BAYC Peer Effects Analysis: Visual Similarity → Price & Timing")
    report.append("=" * 80)
    report.append("")
    
    report.append("🔍 Research Question 3:")
    report.append("   When visually similar apes sell high/fast, does it affect your price/timing?")
    report.append("")
    report.append("-" * 80)
    report.append("")
    
    # Methodology
    report.append("📐 Methodology:")
    report.append(f"   • Visual similarity: CLIP embeddings with cosine similarity")
    report.append(f"   • Peer groups: Top K={K_NEIGHBORS} nearest neighbors in embedding space")
    report.append(f"   • Time windows: {TIME_WINDOWS} days before each sale")
    report.append(f"   • Analysis: Regression of peer activity on sale price & timing")
    report.append("")
    
    # Load and summarize results
    price_files = list(OUTPUT_DIR.glob("peer_price_effects_*.csv"))
    timing_files = list(OUTPUT_DIR.glob("peer_timing_effects_*.csv"))
    
    if price_files:
        report.append("💰 Price Effects Summary:")
        report.append("")
        for file in sorted(price_files):
            df = pd.read_csv(file)
            k = file.stem.split('_k')[1].split('_')[0]
            w = file.stem.split('_w')[1].replace('d', '')
            
            avg_price_ratio = df['price_vs_peer_avg'].mean()
            corr = df['sale_price'].corr(df['peer_avg_price'])
            
            report.append(f"   K={k}, Window={w}d:")
            report.append(f"   • {len(df)} sales analyzed")
            report.append(f"   • Avg price vs peer avg: {avg_price_ratio:.2f}x")
            report.append(f"   • Price correlation: {corr:.4f}")
            report.append("")
    
    if timing_files:
        report.append("⏱️  Timing Effects Summary:")
        report.append("")
        for file in sorted(timing_files):
            df = pd.read_csv(file)
            k = file.stem.split('_k')[1].split('_')[0]
            w = file.stem.split('_w')[1].replace('d', '')
            
            avg_days = df['days_to_next_sale'].mean()
            corr = df['days_to_next_sale'].corr(df['peer_num_sales'])
            
            report.append(f"   K={k}, Window={w}d:")
            report.append(f"   • {len(df)} sales analyzed")
            report.append(f"   • Avg days to next sale: {avg_days:.1f}")
            report.append(f"   • Timing correlation with peer activity: {corr:.4f}")
            report.append("")
    
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open(OUTPUT_DIR / "PEER_EFFECTS_REPORT.txt", "w") as f:
        f.write(report_text)
    
    print(f"  ✓ Saved: PEER_EFFECTS_REPORT.txt")
    print("\n" + report_text)

def main():
    """Main peer effects analysis pipeline"""
    print("\n" + "=" * 70)
    print("BAYC Peer Effects Analysis")
    print("=" * 70)
    
    # Load data
    embeddings, token_ids, df_txns, df_nfts = load_data()
    
    # Compute similarity and find neighbors
    neighbors_dict = compute_similarity_matrix(embeddings, token_ids)
    
    # Prepare sales data
    df_sales = prepare_sales_data(df_txns)
    
    # Join with NFT metadata for cluster info
    df_sales = df_sales.join(
        df_nfts.select(['token_id_str', 'visual_cluster']),
        on='token_id_str',
        how='left'
    )
    
    # Analyze peer effects on price
    for k in K_NEIGHBORS:
        for window in TIME_WINDOWS:
            analyze_peer_effects_on_price(df_sales, neighbors_dict, k, window)
    
    # Analyze peer effects on timing
    for k in K_NEIGHBORS:
        for window in TIME_WINDOWS:
            analyze_peer_effects_on_timing(df_sales, neighbors_dict, k, window)
    
    # Generate report
    generate_peer_effects_report(neighbors_dict)
    
    print("\n" + "=" * 70)
    print("✓ Peer Effects Analysis Complete!")
    print("=" * 70)
    print(f"\n📁 Output Directory: {OUTPUT_DIR}/")
    print("   • visual_neighbors.json")
    print("   • peer_price_effects_*.csv")
    print("   • peer_timing_effects_*.csv")
    print("   • PEER_EFFECTS_REPORT.txt")

if __name__ == "__main__":
    main()
