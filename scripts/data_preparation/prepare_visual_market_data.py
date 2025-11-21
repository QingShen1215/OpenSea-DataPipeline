#!/usr/bin/env python3
"""
BAYC Visual Style Market Cycle Analysis

Research Questions:
1. Which visual styles are more popular in bull/bear markets?
2. Is there "cycle-sensitive aesthetics"?
3. Does selling of visually similar NFTs affect your price/timing?
   - Price contagion: Similar apes sell high -> your price increases?
   - Timing effect: Similar apes sell fast -> you sell faster/slower?

Methodology:
- Use CLIP embeddings (8,208 NFTs)
- Match with transaction data
- Identify market cycles (bull/bear)
- Analyze visual cluster performance across cycles
- Study peer effect: similar NFTs' sales impact
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Paths
EMBEDDINGS_PATH = "embeddings/bayc_embeddings_clip.npz"
TRANSACTIONS_PATH = "../raw_data/bayc.csv"
NFT_METADATA_PATH = "../raw_data/boredapeyachtclub.csv"
OUTPUT_DIR = Path("visual_market_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("BAYC Visual Style × Market Cycle Analysis")
print("="*70)
print()

# ============================================================================
# Step 1: Load and Prepare Data
# ============================================================================

print("📂 Step 1: Loading Data...")
print("-" * 70)

# Load embeddings
print("  Loading CLIP embeddings...")
emb_data = np.load(EMBEDDINGS_PATH)
embeddings = emb_data['embeddings']
token_ids_with_emb = set(emb_data['token_ids'])
print(f"  ✓ Loaded {len(embeddings)} embeddings")

# Load NFT metadata
print("  Loading NFT metadata...")
df_nft = pl.read_csv(NFT_METADATA_PATH)
print(f"  ✓ Loaded {len(df_nft)} NFT records")
print(f"  Columns: {df_nft.columns}")

# Load transaction data
print("  Loading transaction data...")
df_txns = pl.read_csv(TRANSACTIONS_PATH)
print(f"  ✓ Loaded {len(df_txns)} transactions")
print(f"  Columns: {df_txns.columns}")

# Filter to BAYC only and convert token_id to string
# Also convert timestamp to datetime if it's integer (unix timestamp)
if df_txns['timestamp'].dtype == pl.Int64:
    df_txns = df_txns.with_columns([
        pl.col('token_id').cast(pl.Utf8).alias('token_id_str'),
        pl.from_epoch('timestamp', time_unit='s').alias('datetime')
    ])
else:
    df_txns = df_txns.with_columns([
        pl.col('token_id').cast(pl.Utf8).alias('token_id_str'),
        pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias('datetime')
    ])

print()

# ============================================================================
# Step 2: Match Data - Keep only NFTs with all three: embedding, metadata, txns
# ============================================================================

print("📊 Step 2: Matching Data Across Sources...")
print("-" * 70)

# Get token IDs from each source
token_ids_metadata = set(df_nft['token_id'].cast(pl.Utf8).to_list())
token_ids_txns = set(df_txns['token_id_str'].unique().to_list())

print(f"  NFTs with embeddings: {len(token_ids_with_emb)}")
print(f"  NFTs with metadata: {len(token_ids_metadata)}")
print(f"  NFTs with transactions: {len(token_ids_txns)}")

# Find intersection - NFTs with ALL data
complete_token_ids = token_ids_with_emb & token_ids_metadata & token_ids_txns
complete_token_ids = sorted(list(complete_token_ids), key=lambda x: int(x))

print(f"\n  ✓ NFTs with complete data: {len(complete_token_ids)}")
print(f"    Sample: {complete_token_ids[:10]}")

# Filter embeddings to complete set
embedding_dict = {tid: emb for tid, emb in zip(emb_data['token_ids'], embeddings)}
complete_embeddings = np.array([embedding_dict[tid] for tid in complete_token_ids])

# Filter transactions to complete set
df_txns_complete = df_txns.filter(pl.col('token_id_str').is_in(complete_token_ids))
print(f"  ✓ Filtered to {len(df_txns_complete)} transactions")

# Filter metadata to complete set
df_nft_complete = df_nft.filter(
    pl.col('token_id').cast(pl.Utf8).is_in(complete_token_ids)
).with_columns([
    pl.col('token_id').cast(pl.Utf8).alias('token_id_str')
])

print()

# ============================================================================
# Step 3: Prepare Transaction Data with Time Features
# ============================================================================

print("⏰ Step 3: Preparing Time Series Data...")
print("-" * 70)

# Datetime already parsed above, just add time features
# Get date range
min_date = df_txns_complete['datetime'].min()
max_date = df_txns_complete['datetime'].max()
print(f"  Date range: {min_date} to {max_date}")

# Add time features
df_txns_complete = df_txns_complete.with_columns([
    pl.col('datetime').dt.date().alias('date'),
    pl.col('datetime').dt.year().alias('year'),
    pl.col('datetime').dt.month().alias('month'),
    pl.col('datetime').dt.quarter().alias('quarter'),
])

# Calculate ETH price stats by month for cycle identification
# Filter out null/zero prices
df_txns_valid_price = df_txns_complete.filter(
    (pl.col('price_each').is_not_null()) & 
    (pl.col('price_each') > 0)
)
monthly_stats = df_txns_valid_price.group_by(['year', 'month']).agg([
    pl.col('price_each').cast(pl.Float64).mean().alias('avg_price_eth'),
    pl.col('price_each').cast(pl.Float64).median().alias('median_price_eth'),
    pl.count().alias('num_sales'),
    pl.col('price_each').cast(pl.Float64).std().alias('std_price_eth'),
]).sort(['year', 'month'])

print(f"\n  Monthly statistics calculated:")
print(monthly_stats.head(10))

print()

# ============================================================================
# Step 4: Identify Market Cycles (Bull/Bear)
# ============================================================================

print("🐂🐻 Step 4: Identifying Market Cycles...")
print("-" * 70)

# Calculate rolling average and momentum
monthly_stats_pd = monthly_stats.to_pandas()
monthly_stats_pd['date'] = pd.to_datetime(
    monthly_stats_pd['year'].astype(str) + '-' + 
    monthly_stats_pd['month'].astype(str).str.zfill(2)
)
monthly_stats_pd = monthly_stats_pd.sort_values('date')

# Rolling metrics
monthly_stats_pd['price_ma3'] = monthly_stats_pd['avg_price_eth'].rolling(3, min_periods=1).mean()
monthly_stats_pd['price_momentum'] = monthly_stats_pd['avg_price_eth'].pct_change()

# ============================================================================
# METHOD 1: Expert-Labeled Market Regimes (Based on BAYC Collection Price)
# ============================================================================

print("\n  📊 METHOD 1: Expert-Labeled Regimes (Based on BAYC Collection Price)")
print("  " + "-" * 65)

# Define market regimes based on BAYC collection-specific price trends
# NOTE: These differ from overall NFT market cycles because BAYC minted in April 2021
# The collection's own bull/bear doesn't align with broader NFT market timing
MARKET_REGIMES = [
    # Launch & Discovery (2021.04 - 2021.07): Low floor, mint phase
    ('2021-04-01', '2021-07-31', 'discovery', 'BAYC mint & initial discovery phase'),
    
    # Bull Market (2021.08 - 2022.05): Rising floor, peak euphoria
    ('2021-08-01', '2022-05-31', 'bull', 'BAYC bull run: 27 ETH → 800+ ETH peak'),
    
    # Crash (2022.06 - 2022.09): Rapid correction
    ('2022-06-01', '2022-09-30', 'crash', 'Sharp decline: 1400 ETH → 80 ETH'),
    
    # Bear Market (2022.10 - 2024.12): Long-term bear with brief bounces
    ('2022-10-01', '2024-12-31', 'bear', 'Extended bear market with occasional pumps'),
    
    # Current Phase (2025.01+): Bottom/recovery unclear
    ('2025-01-01', '2025-12-31', 'uncertain', 'Current phase: low liquidity'),
]

def assign_expert_cycle(date_val):
    """Assign market cycle based on BAYC-specific price regimes"""
    if pd.isna(date_val):
        return 'unknown'
    date_val = pd.to_datetime(date_val)
    for start, end, cycle_type, _ in MARKET_REGIMES:
        if pd.to_datetime(start) <= date_val <= pd.to_datetime(end):
            return cycle_type
    return 'unknown'

monthly_stats_pd['cycle_expert'] = monthly_stats_pd['date'].apply(assign_expert_cycle)

print("  Expert-labeled cycle distribution:")
print(monthly_stats_pd['cycle_expert'].value_counts().sort_index())
print()

# ============================================================================
# METHOD 2: Algorithmic Momentum-Based Quantile Method (Robustness Check)
# ============================================================================

print("  🔬 METHOD 2: Algorithmic Quantile Method (Robustness Check)")
print("  " + "-" * 65)

# More sophisticated: use quantiles
price_change_25 = monthly_stats_pd['price_momentum'].quantile(0.25)
price_change_75 = monthly_stats_pd['price_momentum'].quantile(0.75)

monthly_stats_pd['cycle_algo'] = monthly_stats_pd['price_momentum'].apply(
    lambda x: 'strong_bull' if x > price_change_75 else
              'bull' if x > 0 else
              'bear' if x > price_change_25 else
              'strong_bear'
)

print("  Algorithmic cycle distribution:")
print(monthly_stats_pd['cycle_algo'].value_counts())
print()

# ============================================================================
# PRIMARY METHOD: Use Expert Labels for Main Analysis
# ============================================================================

print("  ✅ Using EXPERT-LABELED cycles as primary method")
print("  ℹ️  Algorithmic method saved for robustness comparison\n")

# Add cycle info back to transactions (using expert labels)
cycle_map = dict(zip(
    zip(monthly_stats_pd['year'], monthly_stats_pd['month']),
    monthly_stats_pd['cycle_expert']
))

cycle_map_algo = dict(zip(
    zip(monthly_stats_pd['year'], monthly_stats_pd['month']),
    monthly_stats_pd['cycle_algo']
))

df_txns_complete = df_txns_complete.with_columns([
    pl.struct(['year', 'month'])
    .map_elements(lambda x: cycle_map.get((x['year'], x['month']), 'unknown'))
    .alias('market_cycle'),
    pl.struct(['year', 'month'])
    .map_elements(lambda x: cycle_map_algo.get((x['year'], x['month']), 'unknown'))
    .alias('market_cycle_algo')
])

print("  Transaction distribution by cycle (Expert Method):")
cycle_dist = df_txns_complete.group_by('market_cycle').agg(pl.count().alias('count')).sort('market_cycle')
print(cycle_dist)

print("\n  Transaction distribution by cycle (Algorithmic Method):")
cycle_dist_algo = df_txns_complete.group_by('market_cycle_algo').agg(pl.count().alias('count')).sort('market_cycle_algo')
print(cycle_dist_algo)

print()

# ============================================================================
# Step 5: Visual Clustering
# ============================================================================

print("🎨 Step 5: Clustering NFTs by Visual Style...")
print("-" * 70)

# Cluster NFTs into visual groups
N_CLUSTERS = 20
print(f"  Clustering {len(complete_embeddings)} NFTs into {N_CLUSTERS} visual clusters...")

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
visual_clusters = kmeans.fit_predict(complete_embeddings)

# Create cluster mapping
cluster_map = dict(zip(complete_token_ids, visual_clusters))

# Add cluster to NFT metadata
df_nft_complete = df_nft_complete.with_columns([
    pl.col('token_id_str').map_elements(lambda x: cluster_map.get(x, -1)).alias('visual_cluster')
])

# Add cluster to transactions
df_txns_complete = df_txns_complete.with_columns([
    pl.col('token_id_str').map_elements(lambda x: cluster_map.get(x, -1)).alias('visual_cluster')
])

print("  Cluster distribution:")
cluster_dist = df_nft_complete.group_by('visual_cluster').agg(pl.count().alias('count'))
print(cluster_dist.sort('visual_cluster'))

print()

# ============================================================================
# Step 6: Save Prepared Data
# ============================================================================

print("💾 Step 6: Saving Prepared Data...")
print("-" * 70)

# Save embeddings with complete token IDs
np.savez_compressed(
    OUTPUT_DIR / 'complete_embeddings.npz',
    embeddings=complete_embeddings,
    token_ids=np.array(complete_token_ids),
    visual_clusters=visual_clusters
)
print(f"  ✓ Saved embeddings: {OUTPUT_DIR / 'complete_embeddings.npz'}")

# Save processed data
df_txns_complete.write_parquet(OUTPUT_DIR / 'transactions_with_cycles.parquet')
print(f"  ✓ Saved transactions: {OUTPUT_DIR / 'transactions_with_cycles.parquet'}")

df_nft_complete.write_parquet(OUTPUT_DIR / 'nft_metadata_with_clusters.parquet')
print(f"  ✓ Saved NFT metadata: {OUTPUT_DIR / 'nft_metadata_with_clusters.parquet'}")

monthly_stats_pd.to_parquet(OUTPUT_DIR / 'monthly_market_stats.parquet')
print(f"  ✓ Saved monthly stats: {OUTPUT_DIR / 'monthly_market_stats.parquet'}")

# Save cycle comparison for robustness analysis
cycle_comparison = monthly_stats_pd[['date', 'year', 'month', 'avg_price_eth', 
                                       'price_momentum', 'cycle_expert', 'cycle_algo']]
cycle_comparison.to_csv(OUTPUT_DIR / 'cycle_methods_comparison.csv', index=False)
print(f"  ✓ Saved cycle comparison: {OUTPUT_DIR / 'cycle_methods_comparison.csv'}")

# Save summary
summary = {
    'total_nfts': len(complete_token_ids),
    'total_transactions': len(df_txns_complete),
    'date_range': {
        'start': str(min_date),
        'end': str(max_date)
    },
    'visual_clusters': N_CLUSTERS,
    'market_cycles_expert': dict(cycle_dist.to_pandas().values.tolist()),
    'market_cycles_algo': dict(cycle_dist_algo.to_pandas().values.tolist()),
    'cycle_identification_method': {
        'primary': 'expert_labeled',
        'description': 'Based on NFT Index 2020-2025 comprehensive analysis',
        'robustness_check': 'algorithmic_momentum_quantile'
    },
    'token_ids_sample': complete_token_ids[:20]
}

with open(OUTPUT_DIR / 'data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Saved summary: {OUTPUT_DIR / 'data_summary.json'}")

print()
print("="*70)
print("✓ Data Preparation Complete!")
print("="*70)
print()
print(f"📊 Summary:")
print(f"  • Complete NFTs: {len(complete_token_ids)}")
print(f"  • Transactions: {len(df_txns_complete)}")
print(f"  • Visual Clusters: {N_CLUSTERS}")
print(f"  • Date Range: {min_date} to {max_date}")
print()
print("📁 Output Directory: visual_market_analysis/")
print("  • complete_embeddings.npz")
print("  • transactions_with_cycles.parquet")
print("  • nft_metadata_with_clusters.parquet")
print("  • monthly_market_stats.parquet")
print("  • data_summary.json")
print()
print("🚀 Next Steps:")
print("  Run: python analyze_visual_cycles.py")
print("       python analyze_peer_effects.py")
