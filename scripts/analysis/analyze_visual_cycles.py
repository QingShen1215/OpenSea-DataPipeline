"""
BAYC Visual Aesthetics × Market Cycles Analysis
==============================================

Research Question 1: Which visual styles are more popular in bull vs bear markets?
Research Question 2: Is there "cycle-sensitive aesthetics"?

Analysis:
- Compare visual cluster performance across market cycles
- Identify "bull-market darlings" and "bear-market refuges"
- Statistical significance tests for cluster×cycle interactions
"""

import polars as pl
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Paths
DATA_DIR = Path("visual_market_analysis")
TRANSACTIONS_PATH = DATA_DIR / "transactions_with_cycles.parquet"
NFT_METADATA_PATH = DATA_DIR / "nft_metadata_with_clusters.parquet"
OUTPUT_DIR = DATA_DIR / "cycle_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load prepared data"""
    print("📂 Loading Data...")
    print("-" * 70)
    
    df_txns = pl.read_parquet(TRANSACTIONS_PATH)
    df_nfts = pl.read_parquet(NFT_METADATA_PATH)
    
    print(f"  ✓ Loaded {len(df_txns)} transactions")
    print(f"  ✓ Loaded {len(df_nfts)} NFTs with cluster assignments")
    
    return df_txns, df_nfts

def analyze_cluster_performance_by_cycle(df_txns, df_nfts):
    """
    For each visual cluster, calculate performance metrics in each market cycle
    """
    print("\n🎨 Analyzing Visual Cluster Performance by Market Cycle...")
    print("-" * 70)
    
    # Join transactions with NFT cluster info
    df_analysis = df_txns.join(
        df_nfts.select(['token_id_str', 'visual_cluster']),
        on='token_id_str',
        how='inner'
    )
    
    # Filter to sales only (exclude listings, etc.)
    df_sales = df_analysis.filter(
        (pl.col('event_type') == 'sale') &
        (pl.col('price_each').is_not_null()) &
        (pl.col('price_each') > 0)
    )
    
    print(f"  Analyzing {len(df_sales)} sales across clusters and cycles...")
    
    # Group by cluster and cycle
    cluster_cycle_stats = df_sales.group_by(['visual_cluster', 'market_cycle']).agg([
        pl.col('price_each').cast(pl.Float64).mean().alias('avg_price'),
        pl.col('price_each').cast(pl.Float64).median().alias('median_price'),
        pl.col('price_each').cast(pl.Float64).std().alias('std_price'),
        pl.col('price_each').cast(pl.Float64).min().alias('min_price'),
        pl.col('price_each').cast(pl.Float64).max().alias('max_price'),
        pl.count().alias('num_sales'),
        pl.col('token_id_str').n_unique().alias('unique_nfts_sold'),
    ]).sort(['visual_cluster', 'market_cycle'])
    
    # Save detailed stats
    cluster_cycle_stats.write_parquet(OUTPUT_DIR / "cluster_cycle_performance.parquet")
    print(f"  ✓ Saved: cluster_cycle_performance.parquet")
    
    return cluster_cycle_stats

def identify_cycle_sensitive_aesthetics(cluster_cycle_stats):
    """
    Identify clusters with significantly different performance in bull vs bear markets
    Uses BAYC-specific price regimes: discovery, bull, crash, bear, uncertain
    """
    print("\n🔍 Identifying Cycle-Sensitive Aesthetics...")
    print("-" * 70)
    
    # Convert to pandas for easier manipulation
    df = cluster_cycle_stats.to_pandas()
    
    # Calculate performance ratio: bull / bear
    results = []
    
    for cluster_id in sorted(df['visual_cluster'].unique()):
        cluster_data = df[df['visual_cluster'] == cluster_id]
        
        # Get cycle stats based on BAYC-specific regimes
        discovery_data = cluster_data[cluster_data['market_cycle'] == 'discovery']
        bull_data = cluster_data[cluster_data['market_cycle'] == 'bull']
        crash_data = cluster_data[cluster_data['market_cycle'] == 'crash']
        bear_data = cluster_data[cluster_data['market_cycle'] == 'bear']
        
        # Focus on bull vs bear comparison (core periods)
        bull_price = bull_data['avg_price'].mean() if len(bull_data) > 0 else np.nan
        bear_price = bear_data['avg_price'].mean() if len(bear_data) > 0 else np.nan
        
        bull_sales = bull_data['num_sales'].sum() if len(bull_data) > 0 else 0
        bear_sales = bear_data['num_sales'].sum() if len(bear_data) > 0 else 0
        
        # Also track other phases for reference
        discovery_price = discovery_data['avg_price'].mean() if len(discovery_data) > 0 else np.nan
        crash_price = crash_data['avg_price'].mean() if len(crash_data) > 0 else np.nan
        
        # Calculate ratios
        price_ratio = bull_price / bear_price if (bear_price > 0 and not np.isnan(bear_price) and not np.isnan(bull_price)) else np.nan
        sales_ratio = bull_sales / bear_sales if bear_sales > 0 else np.nan
        
        # Cycle sensitivity score: how much does this cluster's performance vary by cycle?
        # High score = very different in bull vs bear
        cycle_sensitivity = abs(np.log(price_ratio)) if not np.isnan(price_ratio) else 0
        
        results.append({
            'visual_cluster': cluster_id,
            'discovery_avg_price': discovery_price,
            'bull_avg_price': bull_price,
            'crash_avg_price': crash_price,
            'bear_avg_price': bear_price,
            'price_bull_bear_ratio': price_ratio,
            'bull_sales': int(bull_sales),
            'bear_sales': int(bear_sales),
            'sales_bull_bear_ratio': sales_ratio,
            'cycle_sensitivity_score': cycle_sensitivity,
            'cluster_size': cluster_data['unique_nfts_sold'].sum(),
        })
    
    df_sensitivity = pd.DataFrame(results).sort_values('cycle_sensitivity_score', ascending=False)
    
    # Identify types
    df_sensitivity['aesthetic_type'] = df_sensitivity.apply(lambda row:
        'bull_darling' if row['price_bull_bear_ratio'] > 1.2 else
        'bear_refuge' if row['price_bull_bear_ratio'] < 0.8 else
        'cycle_neutral',
        axis=1
    )
    
    # Save results
    df_sensitivity.to_csv(OUTPUT_DIR / "cycle_sensitive_aesthetics.csv", index=False)
    print(f"  ✓ Saved: cycle_sensitive_aesthetics.csv")
    
    # Print top findings
    print("\n  📊 Top Cycle-Sensitive Visual Clusters:")
    print("  " + "=" * 66)
    for idx, row in df_sensitivity.head(10).iterrows():
        print(f"  Cluster {row['visual_cluster']:2d} ({row['aesthetic_type']:13s}): "
              f"Bull/Bear Price Ratio = {row['price_bull_bear_ratio']:.2f}, "
              f"Sensitivity = {row['cycle_sensitivity_score']:.3f}")
    
    return df_sensitivity

def statistical_tests(df_txns, df_nfts):
    """
    Statistical significance tests for cluster×cycle interaction effects
    Uses BAYC-specific regimes: discovery, bull, crash, bear
    """
    print("\n📈 Running Statistical Tests...")
    print("-" * 70)
    
    # Prepare data
    df_analysis = df_txns.join(
        df_nfts.select(['token_id_str', 'visual_cluster']),
        on='token_id_str',
        how='inner'
    ).filter(
        (pl.col('event_type') == 'sale') &
        (pl.col('price_each').is_not_null()) &
        (pl.col('price_each') > 0) &
        (pl.col('market_cycle').is_in(['discovery', 'bull', 'crash', 'bear']))  # Exclude 'unknown' and 'uncertain'
    ).to_pandas()
    
    print(f"  Analyzing {len(df_analysis)} sales across {df_analysis['visual_cluster'].nunique()} clusters")
    print(f"  Market cycles: {sorted(df_analysis['market_cycle'].unique())}")
    
    # For each cluster, test if prices differ significantly across cycles
    test_results = []
    
    for cluster_id in sorted(df_analysis['visual_cluster'].unique()):
        cluster_data = df_analysis[df_analysis['visual_cluster'] == cluster_id]
        
        # Get prices for each cycle
        available_cycles = []
        groups = []
        for cycle in ['discovery', 'bull', 'crash', 'bear']:
            cycle_data = cluster_data[cluster_data['market_cycle'] == cycle]['price_each'].values
            if len(cycle_data) > 0:
                available_cycles.append(cycle)
                groups.append(cycle_data)
        
        # Skip if not enough data
        if len(groups) < 2 or any(len(g) < 3 for g in groups):
            continue
        
        # One-way ANOVA: test if means differ across cycles
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        grand_mean = cluster_data['price_each'].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = sum((cluster_data['price_each'] - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        test_results.append({
            'visual_cluster': cluster_id,
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < 0.05,
            'num_samples': len(cluster_data),
            'cycles_tested': ', '.join(available_cycles),
        })
    
    df_tests = pd.DataFrame(test_results).sort_values('p_value')
    df_tests.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
    print(f"  ✓ Saved: statistical_tests.csv")
    
    # Print summary
    num_significant = df_tests['significant'].sum()
    print(f"\n  📊 {num_significant}/{len(df_tests)} clusters show significant cycle effects (p < 0.05)")
    
    return df_tests

def generate_summary_report(df_sensitivity, df_tests):
    """
    Generate human-readable summary report
    """
    print("\n📝 Generating Summary Report...")
    print("-" * 70)
    
    report = []
    report.append("=" * 80)
    report.append("BAYC Visual Aesthetics × Market Cycles Analysis")
    report.append("=" * 80)
    report.append("")
    
    # Research Question 1
    report.append("🔍 Research Question 1: Which visual styles perform better in bull vs bear markets?")
    report.append("-" * 80)
    report.append("")
    
    bull_darlings = df_sensitivity[df_sensitivity['aesthetic_type'] == 'bull_darling'].head(5)
    bear_refuges = df_sensitivity[df_sensitivity['aesthetic_type'] == 'bear_refuge'].head(5)
    
    report.append("📈 Bull Market Darlings (Top 5):")
    report.append("   Visual styles that command higher prices in bull markets:")
    report.append("")
    for idx, row in bull_darlings.iterrows():
        report.append(f"   • Cluster {row['visual_cluster']:2d}: "
                     f"{row['price_bull_bear_ratio']:.2f}x higher price in bull markets "
                     f"({row['bull_sales']} bull sales vs {row['bear_sales']} bear sales)")
    report.append("")
    
    report.append("🐻 Bear Market Refuges (Top 5):")
    report.append("   Visual styles that hold value better in bear markets:")
    report.append("")
    for idx, row in bear_refuges.iterrows():
        report.append(f"   • Cluster {row['visual_cluster']:2d}: "
                     f"{1/row['price_bull_bear_ratio']:.2f}x better relative price in bear markets "
                     f"({row['bear_sales']} bear sales vs {row['bull_sales']} bull sales)")
    report.append("")
    
    # Research Question 2
    report.append("🎨 Research Question 2: Is there 'cycle-sensitive aesthetics'?")
    report.append("-" * 80)
    report.append("")
    
    most_sensitive = df_sensitivity.head(3)
    report.append("🔥 Most Cycle-Sensitive Aesthetics:")
    report.append("   Visual styles whose value fluctuates most with market cycles:")
    report.append("")
    for idx, row in most_sensitive.iterrows():
        report.append(f"   • Cluster {row['visual_cluster']:2d}: "
                     f"Sensitivity Score = {row['cycle_sensitivity_score']:.3f}, "
                     f"Price Ratio = {row['price_bull_bear_ratio']:.2f}")
    report.append("")
    
    least_sensitive = df_sensitivity.tail(3)
    report.append("🧘 Most Cycle-Neutral Aesthetics:")
    report.append("   Visual styles that maintain consistent value across cycles:")
    report.append("")
    for idx, row in least_sensitive.iterrows():
        report.append(f"   • Cluster {row['visual_cluster']:2d}: "
                     f"Sensitivity Score = {row['cycle_sensitivity_score']:.3f}, "
                     f"Price Ratio = {row['price_bull_bear_ratio']:.2f}")
    report.append("")
    
    # Statistical significance
    report.append("📊 Statistical Significance:")
    report.append("-" * 80)
    num_significant = df_tests['significant'].sum()
    report.append(f"   • {num_significant}/{len(df_tests)} visual clusters show statistically significant")
    report.append(f"     cycle effects (ANOVA p < 0.05)")
    report.append("")
    
    if len(df_tests) > 0:
        most_significant = df_tests.head(3)
        report.append("   Top 3 Most Significant:")
        for idx, row in most_significant.iterrows():
            report.append(f"   • Cluster {row['visual_cluster']:2d}: "
                         f"p = {row['p_value']:.6f}, "
                         f"effect size η² = {row['eta_squared']:.3f}")
    report.append("")
    
    # Key findings
    report.append("💡 Key Findings:")
    report.append("-" * 80)
    report.append(f"   1. {len(df_sensitivity[df_sensitivity['aesthetic_type'] == 'bull_darling'])} visual clusters are 'Bull Darlings' (>20% premium in bulls)")
    report.append(f"   2. {len(df_sensitivity[df_sensitivity['aesthetic_type'] == 'bear_refuge'])} visual clusters are 'Bear Refuges' (>20% discount avoided)")
    report.append(f"   3. {len(df_sensitivity[df_sensitivity['aesthetic_type'] == 'cycle_neutral'])} visual clusters are 'Cycle Neutral' (±20% range)")
    report.append(f"   4. Cycle-sensitive aesthetics exist: price variation ranges from "
                 f"{df_sensitivity['price_bull_bear_ratio'].min():.2f}x to "
                 f"{df_sensitivity['price_bull_bear_ratio'].max():.2f}x")
    report.append("")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open(OUTPUT_DIR / "CYCLE_ANALYSIS_REPORT.txt", "w") as f:
        f.write(report_text)
    
    print(f"  ✓ Saved: CYCLE_ANALYSIS_REPORT.txt")
    print("\n" + report_text)

def main():
    """Main analysis pipeline"""
    print("\n" + "=" * 70)
    print("BAYC Visual Aesthetics × Market Cycles Analysis")
    print("=" * 70)
    
    # Load data
    df_txns, df_nfts = load_data()
    
    # Analysis 1: Cluster performance by cycle
    cluster_cycle_stats = analyze_cluster_performance_by_cycle(df_txns, df_nfts)
    
    # Analysis 2: Identify cycle-sensitive aesthetics
    df_sensitivity = identify_cycle_sensitive_aesthetics(cluster_cycle_stats)
    
    # Analysis 3: Statistical tests
    df_tests = statistical_tests(df_txns, df_nfts)
    
    # Generate summary report
    generate_summary_report(df_sensitivity, df_tests)
    
    print("\n" + "=" * 70)
    print("✓ Analysis Complete!")
    print("=" * 70)
    print(f"\n📁 Output Directory: {OUTPUT_DIR}/")
    print("   • cluster_cycle_performance.parquet")
    print("   • cycle_sensitive_aesthetics.csv")
    print("   • statistical_tests.csv")
    print("   • CYCLE_ANALYSIS_REPORT.txt")
    print("\n🚀 Next: Run python analyze_peer_effects.py")

if __name__ == "__main__":
    main()
