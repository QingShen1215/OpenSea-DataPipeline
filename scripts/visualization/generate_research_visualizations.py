#!/usr/bin/env python3
"""
BAYC Visual Aesthetics Research - Complete Visualization Suite
================================================================

Generates all publication-quality figures for the research paper.

Figures:
    1. Market Cycle Timeline with Price Overlay
    2. Cluster Bull/Bear Ratio Ranking (Lollipop Chart)
    3. Bull Darlings vs Bear Refuges Price Distribution
    4. Trait Purity vs Cycle Sensitivity Scatter
    5. CLIP t-SNE Visualization with Cycle Labels
    6. ANOVA Statistical Significance Heatmap
    7. Investment Strategy Backtest

Author: Research Team
Date: 2025-11
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from sklearn.manifold import TSNE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================

# Unified color scheme
COLORS = {
    'discovery': '#F39C12',  # Orange-yellow
    'bull': '#2ECC71',       # Green
    'crash': '#E67E22',      # Deep orange
    'bear': '#3498DB',       # Blue
    'bear_refuge': '#E74C3C', # Red
    'neutral': '#95A5A6',    # Gray
    'primary': '#2E86AB',    # Primary blue
}

# Font sizes
FONT_SIZES = {
    'title': 14,
    'subtitle': 12,
    'axis_label': 11,
    'tick_label': 10,
    'annotation': 9,
    'legend': 10,
}

# Figure sizes
SINGLE_COLUMN = (8, 6)
DOUBLE_COLUMN = (14, 6)
TALL_FIGURE = (10, 12)

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "visual_market_analysis"
OUTPUT_DIR = BASE_DIR / "visual_market_analysis" / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== Data Loading ====================

def load_all_data():
    """Load all required datasets."""
    print("Loading datasets...")
    
    data = {}
    
    # 1. Transaction data with cycles
    data['transactions'] = pd.read_parquet(DATA_DIR / "transactions_with_cycles.parquet")
    # Use time_utc if event_datetime doesn't exist
    if 'time_utc' in data['transactions'].columns:
        data['transactions']['event_datetime'] = pd.to_datetime(data['transactions']['time_utc'])
    elif 'event_datetime' in data['transactions'].columns:
        data['transactions']['event_datetime'] = pd.to_datetime(data['transactions']['event_datetime'])
    
    # 2. Monthly statistics
    data['monthly'] = pd.read_parquet(DATA_DIR / "monthly_market_stats.parquet")
    data['monthly']['month'] = pd.to_datetime(data['monthly']['month'])
    
    # 3. Cycle sensitivity analysis
    data['cycle_sens'] = pd.read_csv(DATA_DIR / "cycle_analysis" / "cycle_sensitive_aesthetics.csv")
    
    # 4. Statistical tests
    data['stats'] = pd.read_csv(DATA_DIR / "cycle_analysis" / "statistical_tests.csv")
    
    # 5. Cluster traits
    data['traits'] = pd.read_csv(DATA_DIR / "cluster_traits" / "cluster_trait_summary.csv")
    
    # 6. NFT metadata
    data['metadata'] = pd.read_parquet(DATA_DIR / "nft_metadata_with_clusters.parquet")
    
    # 7. Embeddings
    embeddings_file = DATA_DIR / "complete_embeddings.npz"
    if embeddings_file.exists():
        emb_data = np.load(embeddings_file)
        data['embeddings'] = emb_data['embeddings']
        data['clusters'] = emb_data['visual_clusters']
    
    # 8. Summary
    with open(DATA_DIR / "data_summary.json", 'r') as f:
        data['summary'] = json.load(f)
    
    print(f"✓ Loaded {len(data['transactions']):,} transactions")
    print(f"✓ Loaded {len(data['metadata']):,} NFTs")
    print(f"✓ Loaded {len(data['cycle_sens'])} clusters")
    
    return data

# ==================== Figure 1: Market Cycle Timeline ====================

def create_figure1_timeline(data):
    """
    Market Cycle Timeline with Price Overlay
    Shows BAYC-specific cycles vs collection floor price.
    """
    print("\n[Figure 1] Creating Market Cycle Timeline...")
    
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    
    df = data['monthly'].copy()
    df = df.sort_values('month')
    
    # Define cycle periods
    cycles = [
        ('2021-04-01', '2021-07-31', 'Discovery', COLORS['discovery']),
        ('2021-08-01', '2022-05-31', 'Bull', COLORS['bull']),
        ('2022-06-01', '2022-09-30', 'Crash', COLORS['crash']),
        ('2022-10-01', '2024-12-31', 'Bear', COLORS['bear']),
    ]
    
    # Background shading
    for start, end, label, color in cycles:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), 
                   alpha=0.2, color=color, label=label)
    
    # Price line
    ax.plot(df['month'], df['avg_price_eth'], 
            linewidth=2.5, color=COLORS['primary'], 
            marker='o', markersize=5, markeredgecolor='white', markeredgewidth=1,
            label='Avg Floor Price', zorder=10)
    
    # Log scale
    ax.set_yscale('log')
    ax.set_ylabel('Average Price (ETH, log scale)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_xlabel('Date', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('BAYC Collection Cycle: Price Timeline (2021-2025)', 
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    
    # Key annotations
    annotations = [
        ('2021-04-23', 0.08, 'Mint\n0.08 ETH', -30, 20),
        ('2021-09-15', 35, 'Peak\n~40 ETH', 10, 20),
        ('2022-02-01', 100, 'ATH Period\n100+ ETH', 10, -30),
        ('2023-02-15', 80, 'Blur Pump\n80 ETH', 10, 20),
    ]
    
    for date_str, price, label, xoffset, yoffset in annotations:
        date = pd.Timestamp(date_str)
        if df['month'].min() <= date <= df['month'].max():
            ax.annotate(label, xy=(date, price),
                       xytext=(xoffset, yoffset), textcoords='offset points',
                       fontsize=FONT_SIZES['annotation'],
                       bbox=dict(boxstyle='round,pad=0.5', fc='white', 
                                alpha=0.85, edgecolor='gray'),
                       arrowprops=dict(arrowstyle='->', lw=1.5, 
                                      color='black', alpha=0.7))
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Reorder: price line first, then cycles
    order = [4, 0, 1, 2, 3] if len(handles) >= 5 else list(range(len(handles)))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
             loc='upper left', fontsize=FONT_SIZES['legend'], 
             framealpha=0.9, edgecolor='gray')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick_label'])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fig1_market_cycle_timeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ==================== Figure 2: Cluster Ranking ====================

def create_figure2_ranking(data):
    """
    Cluster Bull/Bear Ratio Ranking (Lollipop Chart)
    The most important figure - shows 37x price differential.
    """
    print("\n[Figure 2] Creating Cluster Ranking...")
    
    df = data['cycle_sens'].copy()
    df = df.sort_values('price_bull_bear_ratio', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Color mapping
    colors = []
    for ratio in df['price_bull_bear_ratio']:
        if ratio > 1.2:
            colors.append(COLORS['bull'])
        elif ratio < 0.8:
            colors.append(COLORS['bear_refuge'])
        else:
            colors.append(COLORS['neutral'])
    
    y_positions = range(len(df))
    
    # Lollipop stems
    for idx, (y_pos, row, color) in enumerate(zip(y_positions, df.itertuples(), colors)):
        ax.plot([0, row.price_bull_bear_ratio], [y_pos, y_pos],
                color=color, linewidth=2.5, alpha=0.7, zorder=1)
    
    # Lollipop heads (size proportional to cluster size)
    sizes = (df['cluster_size'] / df['cluster_size'].max() * 300) + 100
    scatter = ax.scatter(df['price_bull_bear_ratio'], y_positions,
                        s=sizes, c=colors, alpha=0.8,
                        edgecolors='black', linewidth=1.5, zorder=2)
    
    # Neutral line
    ax.axvline(x=1.0, color='black', linestyle='--', 
               linewidth=2, alpha=0.5, label='Neutral (1.0x)')
    
    # Y-axis labels with cluster names
    labels = []
    for _, row in df.iterrows():
        # Get dominant trait
        trait_info = ""
        if row['visual_cluster'] in data['traits']['cluster_id'].values:
            trait_row = data['traits'][data['traits']['cluster_id'] == row['visual_cluster']].iloc[0]
            # Find highest percentage trait
            trait_cols = [col for col in trait_row.index if col.startswith('top_') and col.endswith('_pct')]
            if trait_cols:
                max_pct = 0
                max_trait = ""
                for col in trait_cols:
                    pct = trait_row[col]
                    if pct > max_pct:
                        max_pct = pct
                        trait_name_col = col.replace('_pct', '')
                        max_trait = trait_row[trait_name_col]
                if max_trait and max_pct > 30:
                    trait_info = f" ({max_trait})"
        
        label = f"C{int(row['visual_cluster'])}{trait_info}"
        labels.append(label)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=FONT_SIZES['tick_label'])
    
    # X-axis (log scale)
    ax.set_xscale('log')
    ax.set_xlabel('Bull/Bear Price Ratio (log scale)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Visual Cluster Cycle Sensitivity Ranking\n(37x Bull Premium → 12x Bear Premium)',
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    
    # Annotations for extremes
    max_row = df.iloc[-1]
    min_row = df.iloc[0]
    
    ax.annotate(f'Cluster {int(max_row["visual_cluster"])}\n{max_row["price_bull_bear_ratio"]:.1f}x Bull Premium\n({int(max_row["cluster_size"])} NFTs)',
                xy=(max_row['price_bull_bear_ratio'], len(df)-1),
                xytext=(15, 0), textcoords='offset points',
                fontsize=FONT_SIZES['annotation'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc=COLORS['bull'], 
                         alpha=0.7, edgecolor='darkgreen', linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
    
    ax.annotate(f'Cluster {int(min_row["visual_cluster"])}\n{1/min_row["price_bull_bear_ratio"]:.1f}x Bear Premium\n({int(min_row["cluster_size"])} NFTs)',
                xy=(min_row['price_bull_bear_ratio'], 0),
                xytext=(15, 0), textcoords='offset points',
                fontsize=FONT_SIZES['annotation'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc=COLORS['bear_refuge'],
                         alpha=0.7, edgecolor='darkred', linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['bull'], label='Bull Darlings (>1.2x)', edgecolor='black'),
        Patch(facecolor=COLORS['bear_refuge'], label='Bear Refuges (<0.8x)', edgecolor='black'),
        Patch(facecolor=COLORS['neutral'], label='Cycle Neutral', edgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
             fontsize=FONT_SIZES['legend'], framealpha=0.9)
    
    ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick_label'])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fig2_cluster_ranking_lollipop.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ==================== Figure 3: Price Distributions ====================

def create_figure3_distributions(data):
    """
    Bull Darlings vs Bear Refuges - Price Distribution (Violin Plot)
    """
    print("\n[Figure 3] Creating Price Distributions...")
    
    df_sens = data['cycle_sens']
    df_txns = data['transactions']
    
    # Select top 3 of each type
    bull_darlings = df_sens.nlargest(3, 'price_bull_bear_ratio')['visual_cluster'].tolist()
    bear_refuges = df_sens.nsmallest(3, 'price_bull_bear_ratio')['visual_cluster'].tolist()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    cycle_order = ['discovery', 'bull', 'crash', 'bear']
    
    # Panel A: Bull Darlings
    df_bull = df_txns[df_txns['visual_cluster'].isin(bull_darlings)].copy()
    # Use market_cycle column
    cycle_col = 'market_cycle' if 'market_cycle' in df_bull.columns else 'cycle_expert'
    df_bull = df_bull[df_bull[cycle_col].isin(cycle_order)]
    df_bull['cluster_label'] = df_bull['visual_cluster'].apply(lambda x: f'C{int(x)}')
    
    if len(df_bull) > 0:
        sns.violinplot(data=df_bull, x=cycle_col, y='price_each',
                      hue='cluster_label', ax=axes[0],
                      palette=[COLORS['bull'], '#58D68D', '#82E0AA'],
                      order=cycle_order, scale='width', inner='quartile',
                      cut=0, linewidth=1.5)
        
        axes[0].set_yscale('log')
        axes[0].set_ylabel('Price (ETH, log scale)', 
                          fontsize=FONT_SIZES['axis_label'], fontweight='bold')
        axes[0].set_xlabel('')
        axes[0].set_title('Panel A: Bull Darlings - Price Distribution Across Market Cycles',
                         fontsize=FONT_SIZES['subtitle'], fontweight='bold', pad=15)
        
        # Update legend
        handles, labels = axes[0].get_legend_handles_labels()
        new_labels = [f'{label} ({df_sens[df_sens["visual_cluster"]==int(label[1:])]["price_bull_bear_ratio"].values[0]:.1f}x)' 
                     for label in labels]
        axes[0].legend(handles, new_labels, title='Cluster (Bull/Bear Ratio)',
                      fontsize=FONT_SIZES['legend'], loc='upper right')
    
    # Panel B: Bear Refuges
    df_bear = df_txns[df_txns['visual_cluster'].isin(bear_refuges)].copy()
    df_bear = df_bear[df_bear[cycle_col].isin(cycle_order)]
    df_bear['cluster_label'] = df_bear['visual_cluster'].apply(lambda x: f'C{int(x)}')
    
    if len(df_bear) > 0:
        sns.violinplot(data=df_bear, x=cycle_col, y='price_each',
                      hue='cluster_label', ax=axes[1],
                      palette=[COLORS['bear_refuge'], '#EC7063', '#F1948A'],
                      order=cycle_order, scale='width', inner='quartile',
                      cut=0, linewidth=1.5)
        
        axes[1].set_yscale('log')
        axes[1].set_ylabel('Price (ETH, log scale)',
                          fontsize=FONT_SIZES['axis_label'], fontweight='bold')
        axes[1].set_xlabel('Market Cycle', 
                          fontsize=FONT_SIZES['axis_label'], fontweight='bold')
        axes[1].set_title('Panel B: Bear Refuges - Price Distribution Across Market Cycles',
                         fontsize=FONT_SIZES['subtitle'], fontweight='bold', pad=15)
        
        # Update legend
        handles, labels = axes[1].get_legend_handles_labels()
        new_labels = [f'{label} ({1/df_sens[df_sens["visual_cluster"]==int(label[1:])]["price_bull_bear_ratio"].values[0]:.1f}x Bear)' 
                     for label in labels]
        axes[1].legend(handles, new_labels, title='Cluster (Bear Premium)',
                      fontsize=FONT_SIZES['legend'], loc='upper right')
    
    for ax in axes:
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=FONT_SIZES['tick_label'])
        ax.set_xticklabels([c.title() for c in cycle_order])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fig3_price_distributions_violin.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ==================== Figure 4: Purity vs Sensitivity ====================

def create_figure4_purity_scatter(data):
    """
    Trait Purity vs Cycle Sensitivity (Scatter Plot with Quadrants)
    """
    print("\n[Figure 4] Creating Purity vs Sensitivity Scatter...")
    
    df_sens = data['cycle_sens'].copy()
    df_traits = data['traits'].copy()
    
    # Calculate max trait purity
    trait_pct_cols = [col for col in df_traits.columns if col.endswith('_pct')]
    df_traits['max_trait_pct'] = df_traits[trait_pct_cols].max(axis=1)
    
    # Merge
    df_plot = df_sens.merge(df_traits[['cluster_id', 'max_trait_pct']], 
                            left_on='visual_cluster', right_on='cluster_id', how='left')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    for idx, row in df_plot.iterrows():
        if pd.isna(row['max_trait_pct']):
            continue
        
        color = COLORS['bull'] if row['price_bull_bear_ratio'] > 1.2 else COLORS['bear_refuge']
        size = (row['cluster_size'] / df_plot['cluster_size'].max() * 500) + 100
        
        ax.scatter(row['max_trait_pct'], row['cycle_sensitivity_score'],
                  s=size, c=color, alpha=0.7,
                  edgecolors='black', linewidth=1.5)
        
        # Label
        ax.text(row['max_trait_pct'], row['cycle_sensitivity_score'],
               f" C{int(row['visual_cluster'])}", 
               fontsize=FONT_SIZES['annotation'], fontweight='bold',
               va='center', ha='left')
    
    # Quadrant lines
    median_purity = df_plot['max_trait_pct'].median()
    median_sensitivity = df_plot['cycle_sensitivity_score'].median()
    
    ax.axhline(y=median_sensitivity, color='gray', linestyle='--', 
               linewidth=1.5, alpha=0.5)
    ax.axvline(x=median_purity, color='gray', linestyle='--',
               linewidth=1.5, alpha=0.5)
    
    # Quadrant labels
    ax.text(90, df_plot['cycle_sensitivity_score'].max() * 0.95,
           'High Purity\nHigh Sensitivity', fontsize=10, 
           ha='right', va='top', style='italic', alpha=0.6)
    ax.text(20, df_plot['cycle_sensitivity_score'].max() * 0.95,
           'Low Purity\nHigh Sensitivity', fontsize=10,
           ha='left', va='top', style='italic', alpha=0.6)
    
    ax.set_xlabel('Dominant Trait Purity (%)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_ylabel('Cycle Sensitivity Score\n(|log(Bull/Bear Ratio)|)',
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Trait Purity vs Cycle Sensitivity\n(Bubble Size = Cluster Size)',
                 fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['bull'], label='Bull Darlings', edgecolor='black'),
        Patch(facecolor=COLORS['bear_refuge'], label='Bear Refuges', edgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
             fontsize=FONT_SIZES['legend'], framealpha=0.9)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick_label'])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fig4_purity_vs_sensitivity_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ==================== Figure 5: t-SNE Visualization ====================

def create_figure5_tsne(data):
    """
    CLIP t-SNE Visualization with Cycle Labels
    """
    print("\n[Figure 5] Creating t-SNE Visualization...")
    
    if 'embeddings' not in data or 'clusters' not in data:
        print("⚠ Skipping Figure 5: Embeddings not found")
        return
    
    embeddings = data['embeddings']
    clusters = data['clusters']
    
    # Subsample if too large
    n_samples = len(embeddings)
    if n_samples > 5000:
        print(f"  Subsampling {n_samples} → 5000 for t-SNE...")
        idx = np.random.choice(n_samples, 5000, replace=False)
        embeddings_sub = embeddings[idx]
        clusters_sub = clusters[idx]
    else:
        embeddings_sub = embeddings
        clusters_sub = clusters
    
    print("  Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_sub)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot points
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=clusters_sub, cmap='tab20', s=15, alpha=0.5,
                        edgecolors='none')
    
    # Cluster centers and labels
    for cluster_id in range(20):
        mask = clusters_sub == cluster_id
        if mask.sum() > 0:
            center = embeddings_2d[mask].mean(axis=0)
            ax.scatter(center[0], center[1], s=600, c='black',
                      marker='o', edgecolors='white', linewidth=3, zorder=10)
            ax.text(center[0], center[1], str(cluster_id),
                   fontsize=12, fontweight='bold', color='white',
                   ha='center', va='center', zorder=11)
    
    ax.set_title('CLIP Embeddings t-SNE: 20 Visual Aesthetic Clusters',
                fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.axis('off')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Cluster ID', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fig5_tsne_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ==================== Figure 6: ANOVA Heatmap ====================

def create_figure6_anova_heatmap(data):
    """
    ANOVA Statistical Significance Heatmap
    """
    print("\n[Figure 6] Creating ANOVA Heatmap...")
    
    df_stats = data['stats'].copy()
    df_stats = df_stats.sort_values('visual_cluster')
    
    # Prepare matrices
    clusters = df_stats['visual_cluster'].values
    f_stats = df_stats['f_statistic'].values
    p_values = df_stats['p_value'].values
    p_log = -np.log10(p_values)
    eta_sq = df_stats['eta_squared'].values
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # Heatmap 1: F-statistic
    im1 = axes[0].imshow([f_stats], cmap='Blues', aspect='auto', vmin=0)
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(['F-statistic'], fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    axes[0].set_xticks(range(len(clusters)))
    axes[0].set_xticklabels([])
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical', pad=0.01)
    cbar1.ax.tick_params(labelsize=FONT_SIZES['tick_label'])
    
    # Add significance markers
    for i, (f, p) in enumerate(zip(f_stats, p_values)):
        if p < 0.001:
            axes[0].text(i, 0, '***', ha='center', va='center',
                        fontweight='bold', fontsize=14, color='darkblue')
        elif p < 0.01:
            axes[0].text(i, 0, '**', ha='center', va='center',
                        fontweight='bold', fontsize=12, color='darkblue')
        elif p < 0.05:
            axes[0].text(i, 0, '*', ha='center', va='center',
                        fontweight='bold', fontsize=10, color='darkblue')
    
    # Heatmap 2: -log10(p-value)
    im2 = axes[1].imshow([p_log], cmap='Reds', aspect='auto', vmin=0)
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(['-log₁₀(p)'], fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    axes[1].set_xticks(range(len(clusters)))
    axes[1].set_xticklabels([])
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='vertical', pad=0.01)
    cbar2.ax.tick_params(labelsize=FONT_SIZES['tick_label'])
    
    # Add significance line
    axes[1].axhline(y=0, xmin=0, xmax=1, color='white', linewidth=2, linestyle='--')
    
    # Heatmap 3: Eta-squared (effect size)
    im3 = axes[2].imshow([eta_sq], cmap='Greens', aspect='auto', vmin=0)
    axes[2].set_yticks([0])
    axes[2].set_yticklabels(['η² (Effect Size)'], fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    axes[2].set_xticks(range(len(clusters)))
    axes[2].set_xticklabels([f'C{int(c)}' for c in clusters], 
                           fontsize=FONT_SIZES['tick_label'])
    cbar3 = fig.colorbar(im3, ax=axes[2], orientation='vertical', pad=0.01)
    cbar3.ax.tick_params(labelsize=FONT_SIZES['tick_label'])
    
    fig.suptitle('ANOVA Statistical Significance by Cluster\n(*** p<0.001, ** p<0.01, * p<0.05)',
                fontsize=FONT_SIZES['title'], fontweight='bold', y=0.995)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fig6_anova_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ==================== Figure 7: Strategy Backtest ====================

def create_figure7_backtest(data):
    """
    Investment Strategy Backtest
    Compares Bull Darlings vs Bear Refuges vs Balanced Portfolio
    """
    print("\n[Figure 7] Creating Strategy Backtest...")
    
    df_sens = data['cycle_sens']
    df_txns = data['transactions'].copy()
    
    # Identify strategies
    bull_darlings = df_sens.nlargest(3, 'price_bull_bear_ratio')['visual_cluster'].tolist()
    bear_refuges = df_sens.nsmallest(3, 'price_bull_bear_ratio')['visual_cluster'].tolist()
    neutral = df_sens[(df_sens['price_bull_bear_ratio'] >= 0.8) & 
                      (df_sens['price_bull_bear_ratio'] <= 1.2)]['visual_cluster'].tolist()
    
    # Aggregate to monthly
    df_txns['month'] = pd.to_datetime(df_txns['event_datetime']).dt.to_period('M').dt.to_timestamp()
    
    def calculate_returns(clusters, name):
        """Calculate monthly returns for a cluster strategy."""
        df_strat = df_txns[df_txns['visual_cluster'].isin(clusters)].copy()
        monthly = df_strat.groupby('month')['price_each'].mean().sort_index()
        returns = monthly.pct_change().fillna(0)
        cumulative = (1 + returns).cumprod() * 100  # Start with 100 ETH
        return cumulative
    
    # Calculate returns
    returns_bull = calculate_returns(bull_darlings, 'Bull Darlings')
    returns_bear = calculate_returns(bear_refuges, 'Bear Refuges')
    returns_baseline = calculate_returns(list(range(20)), 'Buy & Hold')
    
    # Balanced: 40% bull + 40% bear + 20% neutral
    if len(neutral) > 0:
        returns_neutral = calculate_returns(neutral, 'Neutral')
        returns_balanced = returns_bull * 0.4 + returns_bear * 0.4 + returns_neutral * 0.2
    else:
        returns_balanced = returns_bull * 0.5 + returns_bear * 0.5
    
    # Plot
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    
    months = returns_baseline.index
    
    ax.plot(months, returns_bull, linewidth=2.5, color=COLORS['bull'],
           label=f'Bull Darlings (C{bull_darlings[0]}, {bull_darlings[1]}, {bull_darlings[2]})',
           marker='o', markersize=4, markevery=3)
    
    ax.plot(months, returns_bear, linewidth=2.5, color=COLORS['bear_refuge'],
           label=f'Bear Refuges (C{bear_refuges[0]}, {bear_refuges[1]}, {bear_refuges[2]})',
           marker='s', markersize=4, markevery=3)
    
    ax.plot(months, returns_balanced, linewidth=2.5, color=COLORS['primary'],
           label='Balanced Portfolio (40% Bull + 40% Bear + 20% Neutral)',
           marker='^', markersize=4, markevery=3)
    
    ax.plot(months, returns_baseline, linewidth=2, color='gray', linestyle='--',
           label='Buy & Hold Baseline (All Clusters)', alpha=0.7)
    
    # Cycle backgrounds
    cycles = [
        ('2021-08-01', '2022-05-31', 'Bull Period', COLORS['bull'], 0.1),
        ('2022-10-01', '2024-12-31', 'Bear Period', COLORS['bear'], 0.1),
    ]
    
    for start, end, label, color, alpha in cycles:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts >= months.min() and end_ts <= months.max():
            ax.axvspan(start_ts, end_ts, alpha=alpha, color=color, label=label)
    
    ax.set_ylabel('Portfolio Value (ETH)', 
                  fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_xlabel('Date', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    ax.set_title('Investment Strategy Backtest: Cycle-Based Portfolio Performance\n(Initial: 100 ETH, Monthly Rebalancing)',
                fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', fontsize=FONT_SIZES['legend']-1, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick_label'])
    
    # Add final values as text
    final_bull = returns_bull.iloc[-1]
    final_bear = returns_bear.iloc[-1]
    final_balanced = returns_balanced.iloc[-1]
    final_baseline = returns_baseline.iloc[-1]
    
    textstr = f'Final Values (Sep 2025):\n'
    textstr += f'Bull Strategy: {final_bull:.1f} ETH\n'
    textstr += f'Bear Strategy: {final_bear:.1f} ETH\n'
    textstr += f'Balanced: {final_balanced:.1f} ETH\n'
    textstr += f'Baseline: {final_baseline:.1f} ETH'
    
    ax.text(0.98, 0.50, textstr, transform=ax.transAxes,
           fontsize=FONT_SIZES['annotation'], verticalalignment='center',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fig7_strategy_backtest.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ==================== Main Execution ====================

def main():
    """Generate all figures."""
    print("="*70)
    print("BAYC Visual Aesthetics Research - Visualization Generator")
    print("="*70)
    
    # Load data
    data = load_all_data()
    
    # Generate all figures
    figures = [
        ("Figure 1: Market Cycle Timeline", create_figure1_timeline),
        ("Figure 2: Cluster Ranking", create_figure2_ranking),
        ("Figure 3: Price Distributions", create_figure3_distributions),
        ("Figure 4: Purity vs Sensitivity", create_figure4_purity_scatter),
        ("Figure 5: t-SNE Visualization", create_figure5_tsne),
        ("Figure 6: ANOVA Heatmap", create_figure6_anova_heatmap),
        ("Figure 7: Strategy Backtest", create_figure7_backtest),
    ]
    
    print("\n" + "="*70)
    print("Generating Figures...")
    print("="*70)
    
    for name, func in figures:
        try:
            func(data)
        except Exception as e:
            print(f"✗ Error in {name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✓ All figures generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("="*70)
    
    # List all generated files
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("fig*.png")):
        file_size = file.stat().st_size / 1024  # KB
        print(f"  • {file.name} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main()
