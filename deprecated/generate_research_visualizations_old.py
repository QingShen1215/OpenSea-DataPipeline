"""
Visualizations for BAYC Visual Aesthetics × Market Cycles Research
================================================================

Generate comprehensive visualizations for:
1. Cycle-sensitive aesthetics
2. Peer effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Paths
DATA_DIR = Path("visual_market_analysis")
CYCLE_DIR = DATA_DIR / "cycle_analysis"
PEER_DIR = DATA_DIR / "peer_effects"
VIZ_DIR = DATA_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

def plot_cycle_sensitivity_ranking():
    """
    Plot: Cycle sensitivity ranking for all visual clusters
    """
    df = pd.read_csv(CYCLE_DIR / "cycle_sensitive_aesthetics.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Cycle sensitivity score
    ax1 = axes[0]
    colors = ['red' if x == 'bull_darling' else 'blue' if x == 'bear_refuge' else 'gray' 
              for x in df['aesthetic_type']]
    
    ax1.barh(range(len(df)), df['cycle_sensitivity_score'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels([f"Cluster {x}" for x in df['visual_cluster']])
    ax1.set_xlabel('Cycle Sensitivity Score', fontsize=13, fontweight='bold')
    ax1.set_title('Visual Clusters Ranked by Cycle Sensitivity', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Bull Darling'),
        Patch(facecolor='blue', alpha=0.7, label='Bear Refuge'),
        Patch(facecolor='gray', alpha=0.7, label='Cycle Neutral')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Bull/Bear price ratio
    ax2 = axes[1]
    ax2.barh(range(len(df)), df['price_bull_bear_ratio'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels([f"Cluster {x}" for x in df['visual_cluster']])
    ax2.set_xlabel('Bull/Bear Price Ratio', fontsize=13, fontweight='bold')
    ax2.set_title('Price Premium in Bull vs Bear Markets', fontsize=14, fontweight='bold')
    ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "01_cycle_sensitivity_ranking.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 01_cycle_sensitivity_ranking.png")
    plt.close()

def plot_cluster_performance_by_cycle():
    """
    Plot: Average price by cluster and cycle
    """
    df = pd.read_parquet(CYCLE_DIR / "cluster_cycle_performance.parquet")
    
    # Pivot for heatmap
    pivot = df.pivot_table(
        index='visual_cluster',
        columns='market_cycle',
        values='avg_price',
        aggfunc='mean'
    )
    
    # Reorder columns
    cycle_order = ['strong_bull', 'bull', 'bear', 'strong_bear']
    pivot = pivot[[c for c in cycle_order if c in pivot.columns]]
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                cbar_kws={'label': 'Avg Price (ETH)'}, ax=ax, linewidths=0.5)
    ax.set_title('Average NFT Price by Visual Cluster and Market Cycle', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Market Cycle', fontsize=12, fontweight='bold')
    ax.set_ylabel('Visual Cluster', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "02_cluster_cycle_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 02_cluster_cycle_heatmap.png")
    plt.close()

def plot_bull_darlings_vs_bear_refuges():
    """
    Plot: Direct comparison of bull darlings vs bear refuges
    """
    df = pd.read_csv(CYCLE_DIR / "cycle_sensitive_aesthetics.csv")
    
    bull_darlings = df[df['aesthetic_type'] == 'bull_darling'].nlargest(5, 'price_bull_bear_ratio')
    bear_refuges = df[df['aesthetic_type'] == 'bear_refuge'].nsmallest(5, 'price_bull_bear_ratio')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Bull Darlings
    ax1 = axes[0]
    x1 = range(len(bull_darlings))
    ax1.bar([i-0.2 for i in x1], bull_darlings['bull_avg_price'], width=0.4, 
            label='Bull Market Price', color='green', alpha=0.7)
    ax1.bar([i+0.2 for i in x1], bull_darlings['bear_avg_price'], width=0.4, 
            label='Bear Market Price', color='red', alpha=0.7)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([f"Cluster {x}" for x in bull_darlings['visual_cluster']])
    ax1.set_ylabel('Average Price (ETH)', fontsize=12, fontweight='bold')
    ax1.set_title('Bull Market Darlings: Visual Styles That Thrive in Bulls', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add ratio labels
    for i, (idx, row) in enumerate(bull_darlings.iterrows()):
        ax1.text(i, max(row['bull_avg_price'], row['bear_avg_price']) * 1.05, 
                f"{row['price_bull_bear_ratio']:.1f}x", 
                ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Bear Refuges
    ax2 = axes[1]
    x2 = range(len(bear_refuges))
    ax2.bar([i-0.2 for i in x2], bear_refuges['bull_avg_price'], width=0.4, 
            label='Bull Market Price', color='green', alpha=0.7)
    ax2.bar([i+0.2 for i in x2], bear_refuges['bear_avg_price'], width=0.4, 
            label='Bear Market Price', color='red', alpha=0.7)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"Cluster {x}" for x in bear_refuges['visual_cluster']])
    ax2.set_ylabel('Average Price (ETH)', fontsize=12, fontweight='bold')
    ax2.set_title('Bear Market Refuges: Visual Styles That Hold Value in Bears', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add ratio labels
    for i, (idx, row) in enumerate(bear_refuges.iterrows()):
        ax2.text(i, max(row['bull_avg_price'], row['bear_avg_price']) * 1.05, 
                f"{1/row['price_bull_bear_ratio']:.1f}x", 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "03_bull_darlings_vs_bear_refuges.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 03_bull_darlings_vs_bear_refuges.png")
    plt.close()

def plot_sales_volume_by_cluster_cycle():
    """
    Plot: Sales volume distribution across clusters and cycles
    """
    df = pd.read_parquet(CYCLE_DIR / "cluster_cycle_performance.parquet")
    
    # Pivot for stacked bar chart
    pivot = df.pivot_table(
        index='visual_cluster',
        columns='market_cycle',
        values='num_sales',
        aggfunc='sum',
        fill_value=0
    )
    
    # Reorder columns
    cycle_order = ['strong_bull', 'bull', 'bear', 'strong_bear']
    pivot = pivot[[c for c in cycle_order if c in pivot.columns]]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    pivot.plot(kind='barh', stacked=True, ax=ax, 
               color=['darkgreen', 'lightgreen', 'lightcoral', 'darkred'], alpha=0.8)
    ax.set_xlabel('Number of Sales', fontsize=12, fontweight='bold')
    ax.set_ylabel('Visual Cluster', fontsize=12, fontweight='bold')
    ax.set_title('Sales Volume Distribution by Visual Cluster and Market Cycle', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Market Cycle', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "04_sales_volume_by_cluster_cycle.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 04_sales_volume_by_cluster_cycle.png")
    plt.close()

def plot_peer_price_effects():
    """
    Plot: Peer effects on price
    """
    # Load one peer effect file
    file = PEER_DIR / "peer_price_effects_k20_w30d.csv"
    if not file.exists():
        print(f"  ⚠️ Peer price effects file not found")
        return
    
    df = pd.read_csv(file)
    
    # Filter outliers
    df = df[
        (df['sale_price'] < df['sale_price'].quantile(0.95)) &
        (df['peer_avg_price'] < df['peer_avg_price'].quantile(0.95)) &
        (df['peer_num_sales'] >= 2)
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Sale price vs Peer avg price
    ax1 = axes[0]
    ax1.scatter(df['peer_avg_price'], df['sale_price'], alpha=0.3, s=20)
    ax1.set_xlabel('Peer Average Price (30d window, ETH)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sale Price (ETH)', fontsize=12, fontweight='bold')
    ax1.set_title('Your Price vs Peer Average Price\n(K=20 nearest visual neighbors)', 
                  fontsize=13, fontweight='bold')
    
    # Add regression line
    from scipy.stats import linregress
    mask = df['peer_avg_price'].notna() & df['sale_price'].notna()
    slope, intercept, r_value, _, _ = linregress(df['peer_avg_price'][mask], df['sale_price'][mask])
    x_line = np.linspace(df['peer_avg_price'].min(), df['peer_avg_price'].max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, 
             label=f'R² = {r_value**2:.4f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Distribution of price ratio
    ax2 = axes[1]
    price_ratio = df['sale_price'] / df['peer_avg_price']
    price_ratio = price_ratio[price_ratio < price_ratio.quantile(0.99)]
    
    ax2.hist(price_ratio, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Equal Price')
    ax2.set_xlabel('Your Price / Peer Average Price', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Price Relative to Peers', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "05_peer_price_effects.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 05_peer_price_effects.png")
    plt.close()

def plot_peer_timing_effects():
    """
    Plot: Peer effects on timing
    """
    file = PEER_DIR / "peer_timing_effects_k20_w30d.csv"
    if not file.exists():
        print(f"  ⚠️ Peer timing effects file not found")
        return
    
    df = pd.read_csv(file)
    
    # Filter outliers
    df = df[
        (df['days_to_next_sale'] < df['days_to_next_sale'].quantile(0.95)) &
        (df['peer_num_sales'] >= 2)
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Days to next sale vs peer sales volume
    ax1 = axes[0]
    
    # Bin peer_num_sales for clearer visualization
    df['peer_sales_bin'] = pd.cut(df['peer_num_sales'], bins=[0, 5, 10, 20, 50, 100])
    grouped = df.groupby('peer_sales_bin')['days_to_next_sale'].agg(['mean', 'std', 'count'])
    
    x_pos = range(len(grouped))
    ax1.bar(x_pos, grouped['mean'], yerr=grouped['std'], alpha=0.7, capsize=5, color='steelblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(grouped.index.astype(str), rotation=45, ha='right')
    ax1.set_xlabel('Number of Peer Sales (30d window)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Days to Next Sale', fontsize=12, fontweight='bold')
    ax1.set_title('Time to Resale vs Peer Activity\n(K=20 nearest visual neighbors)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Scatter plot
    ax2 = axes[1]
    sample = df.sample(min(5000, len(df)))
    ax2.scatter(sample['peer_num_sales'], sample['days_to_next_sale'], alpha=0.2, s=20)
    ax2.set_xlabel('Number of Peer Sales (30d window)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Days to Next Sale', fontsize=12, fontweight='bold')
    ax2.set_title('Peer Activity vs Resale Timing', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Add correlation
    corr = df['peer_num_sales'].corr(df['days_to_next_sale'])
    ax2.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "06_peer_timing_effects.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 06_peer_timing_effects.png")
    plt.close()

def plot_statistical_significance():
    """
    Plot: Statistical significance of cycle effects
    """
    df = pd.read_csv(CYCLE_DIR / "statistical_tests.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: P-values
    ax1 = axes[0]
    colors = ['green' if p < 0.05 else 'gray' for p in df['p_value']]
    ax1.barh(range(len(df)), -np.log10(df['p_value']), color=colors, alpha=0.7)
    ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p = 0.05')
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels([f"Cluster {x}" for x in df['visual_cluster']])
    ax1.set_xlabel('-log10(p-value)', fontsize=12, fontweight='bold')
    ax1.set_title('Statistical Significance of Cycle Effects\n(ANOVA test)', 
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Effect size
    ax2 = axes[1]
    ax2.barh(range(len(df)), df['eta_squared'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels([f"Cluster {x}" for x in df['visual_cluster']])
    ax2.set_xlabel('Effect Size (η²)', fontsize=12, fontweight='bold')
    ax2.set_title('Effect Size of Market Cycles on Price', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "07_statistical_significance.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 07_statistical_significance.png")
    plt.close()

def main():
    """Generate all visualizations"""
    print("\n" + "=" * 70)
    print("Generating Visualizations for Visual Aesthetics Research")
    print("=" * 70)
    
    print("\n📊 Cycle-Sensitive Aesthetics Visualizations...")
    plot_cycle_sensitivity_ranking()
    plot_cluster_performance_by_cycle()
    plot_bull_darlings_vs_bear_refuges()
    plot_sales_volume_by_cluster_cycle()
    plot_statistical_significance()
    
    print("\n🔗 Peer Effects Visualizations...")
    plot_peer_price_effects()
    plot_peer_timing_effects()
    
    print("\n" + "=" * 70)
    print("✓ All Visualizations Complete!")
    print("=" * 70)
    print(f"\n📁 Output Directory: {VIZ_DIR}/")
    print("   • 01_cycle_sensitivity_ranking.png")
    print("   • 02_cluster_cycle_heatmap.png")
    print("   • 03_bull_darlings_vs_bear_refuges.png")
    print("   • 04_sales_volume_by_cluster_cycle.png")
    print("   • 05_peer_price_effects.png")
    print("   • 06_peer_timing_effects.png")
    print("   • 07_statistical_significance.png")

if __name__ == "__main__":
    main()
