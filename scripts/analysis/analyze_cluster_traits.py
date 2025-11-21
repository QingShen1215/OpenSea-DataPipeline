"""
Analyze Trait Composition of Visual Clusters
===========================================

Combine visual clusters with actual NFT traits to understand:
- What specific traits define "Bull Darlings"?
- What traits characterize "Bear Refuges"?
- Are there trait patterns that explain cycle sensitivity?
"""

import polars as pl
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict

# Paths
DATA_DIR = Path("visual_market_analysis")
NFT_METADATA_PATH = DATA_DIR / "nft_metadata_with_clusters.parquet"
CYCLE_SENSITIVITY_PATH = DATA_DIR / "cycle_analysis" / "cycle_sensitive_aesthetics.csv"
COLLECTION_METADATA_PATH = Path("../raw_data/boredapeyachtclub.csv")
OUTPUT_DIR = DATA_DIR / "cluster_traits"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load NFT metadata with clusters and traits"""
    print("📂 Loading Data...")
    print("-" * 70)
    
    # Load NFTs with cluster assignments
    df_nfts = pl.read_parquet(NFT_METADATA_PATH)
    print(f"  ✓ Loaded {len(df_nfts)} NFTs with cluster assignments")
    
    # Load collection metadata with traits
    df_collection = pl.read_csv(COLLECTION_METADATA_PATH)
    print(f"  ✓ Loaded {len(df_collection)} NFT metadata records")
    
    # Load cycle sensitivity scores
    df_sensitivity = pd.read_csv(CYCLE_SENSITIVITY_PATH)
    print(f"  ✓ Loaded cycle sensitivity scores for {len(df_sensitivity)} clusters")
    
    return df_nfts, df_collection, df_sensitivity

def parse_traits(df_collection):
    """
    Parse attributes JSON to extract individual traits
    """
    print("\n🔍 Parsing NFT Traits...")
    print("-" * 70)
    
    # Convert to pandas for easier JSON handling
    df = df_collection.to_pandas()
    
    # Parse attributes JSON
    traits_data = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Parsing {idx}/{len(df)}...")
        
        token_id = str(row['token_id'])
        attributes = row['attributes']
        
        try:
            # Parse JSON - attributes is a JSON array of objects
            if isinstance(attributes, str):
                attrs = json.loads(attributes)
            else:
                attrs = attributes
            
            # Extract individual traits from array format
            trait_dict = {'token_id': token_id}
            
            if isinstance(attrs, list):
                # Format: [{"trait_type": "Fur", "value": "Red"}, ...]
                for trait_obj in attrs:
                    if isinstance(trait_obj, dict) and 'trait_type' in trait_obj and 'value' in trait_obj:
                        trait_type = trait_obj['trait_type']
                        trait_value = trait_obj['value']
                        trait_dict[trait_type] = trait_value
            elif isinstance(attrs, dict):
                # Fallback: if it's already a dict
                for key, value in attrs.items():
                    trait_dict[key] = value
            
            traits_data.append(trait_dict)
            
        except Exception as e:
            # Skip errors silently, just add token_id
            traits_data.append({'token_id': token_id})
    
    df_traits = pd.DataFrame(traits_data)
    print(f"\n  ✓ Parsed traits for {len(df_traits)} NFTs")
    print(f"  Trait categories: {[col for col in df_traits.columns if col != 'token_id']}")
    
    return df_traits

def analyze_cluster_trait_composition(df_nfts, df_traits, df_sensitivity):
    """
    For each cluster, analyze the distribution of traits
    """
    print("\n🎨 Analyzing Trait Composition by Cluster...")
    print("-" * 70)
    
    # Merge cluster assignments with traits
    df_merged = df_nfts.to_pandas().merge(
        df_traits,
        left_on='token_id_str',
        right_on='token_id',
        how='inner'
    )
    
    print(f"  ✓ Merged data: {len(df_merged)} NFTs with clusters and traits")
    
    # Get trait columns (exclude metadata columns)
    trait_columns = [col for col in df_traits.columns 
                    if col not in ['token_id', 'token_id_str', 'nft_name']]
    
    print(f"  Analyzing {len(trait_columns)} trait categories: {trait_columns}")
    
    # For each cluster, calculate trait distributions
    cluster_traits = {}
    
    for cluster_id in sorted(df_merged['visual_cluster'].unique()):
        cluster_data = df_merged[df_merged['visual_cluster'] == cluster_id]
        
        cluster_info = {
            'cluster_id': int(cluster_id),
            'size': len(cluster_data),
            'traits': {}
        }
        
        # For each trait category, get distribution
        for trait_col in trait_columns:
            if trait_col in cluster_data.columns:
                trait_counts = cluster_data[trait_col].value_counts().to_dict()
                trait_percentages = {
                    str(k): round(v / len(cluster_data) * 100, 1)
                    for k, v in trait_counts.items()
                    if pd.notna(k)
                }
                cluster_info['traits'][trait_col] = trait_percentages
        
        cluster_traits[int(cluster_id)] = cluster_info
    
    # Save raw data
    with open(OUTPUT_DIR / "cluster_trait_distributions.json", "w") as f:
        json.dump(cluster_traits, f, indent=2)
    
    print(f"\n  ✓ Saved: cluster_trait_distributions.json")
    
    return cluster_traits, df_merged, trait_columns

def identify_cluster_signatures(cluster_traits, df_sensitivity, trait_columns):
    """
    Identify the most distinctive traits for each cluster
    """
    print("\n🔬 Identifying Cluster Trait Signatures...")
    print("-" * 70)
    
    # Calculate overall trait distributions (baseline)
    all_trait_distributions = {}
    total_nfts = sum(info['size'] for info in cluster_traits.values())
    
    for trait_col in trait_columns:
        trait_counts = Counter()
        for cluster_info in cluster_traits.values():
            if trait_col in cluster_info['traits']:
                for trait_value, pct in cluster_info['traits'][trait_col].items():
                    # Convert percentage back to count
                    count = int(pct * cluster_info['size'] / 100)
                    trait_counts[trait_value] += count
        
        all_trait_distributions[trait_col] = {
            k: round(v / total_nfts * 100, 1)
            for k, v in trait_counts.items()
        }
    
    # For each cluster, find traits that are over-represented
    cluster_signatures = []
    
    for cluster_id, cluster_info in cluster_traits.items():
        # Get cycle info
        cycle_info = df_sensitivity[df_sensitivity['visual_cluster'] == cluster_id].iloc[0]
        
        signature = {
            'cluster_id': cluster_id,
            'size': cluster_info['size'],
            'aesthetic_type': cycle_info['aesthetic_type'],
            'price_bull_bear_ratio': float(cycle_info['price_bull_bear_ratio']),
            'cycle_sensitivity_score': float(cycle_info['cycle_sensitivity_score']),
            'distinctive_traits': []
        }
        
        # Find over-represented traits
        for trait_col in trait_columns:
            if trait_col not in cluster_info['traits']:
                continue
            
            for trait_value, cluster_pct in cluster_info['traits'][trait_col].items():
                overall_pct = all_trait_distributions[trait_col].get(trait_value, 0)
                
                # Consider trait distinctive if it appears at least 2x more than overall
                if cluster_pct > 20 and cluster_pct > overall_pct * 1.5:
                    over_representation = cluster_pct / overall_pct if overall_pct > 0 else float('inf')
                    
                    signature['distinctive_traits'].append({
                        'category': trait_col,
                        'value': trait_value,
                        'cluster_pct': cluster_pct,
                        'overall_pct': overall_pct,
                        'over_representation': round(over_representation, 2)
                    })
        
        # Sort by over-representation
        signature['distinctive_traits'].sort(
            key=lambda x: x['over_representation'], 
            reverse=True
        )
        
        cluster_signatures.append(signature)
    
    # Save signatures
    with open(OUTPUT_DIR / "cluster_trait_signatures.json", "w") as f:
        json.dump(cluster_signatures, f, indent=2)
    
    print(f"  ✓ Saved: cluster_trait_signatures.json")
    
    return cluster_signatures

def generate_bull_darling_report(cluster_signatures):
    """
    Generate report for Bull Darling clusters
    """
    print("\n📈 Analyzing Bull Darling Traits...")
    print("-" * 70)
    
    bull_darlings = [
        sig for sig in cluster_signatures
        if sig['aesthetic_type'] == 'bull_darling'
    ]
    
    # Sort by price ratio
    bull_darlings.sort(key=lambda x: x['price_bull_bear_ratio'], reverse=True)
    
    report = []
    report.append("=" * 80)
    report.append("BULL MARKET DARLINGS: Trait Analysis")
    report.append("=" * 80)
    report.append("")
    
    for sig in bull_darlings[:10]:  # Top 10
        report.append(f"📊 Cluster {sig['cluster_id']}: {sig['price_bull_bear_ratio']:.2f}x Bull/Bear Ratio")
        report.append(f"   Size: {sig['size']} NFTs")
        report.append(f"   Cycle Sensitivity: {sig['cycle_sensitivity_score']:.3f}")
        report.append("")
        report.append("   🔥 Distinctive Traits:")
        
        if sig['distinctive_traits']:
            for trait in sig['distinctive_traits'][:5]:  # Top 5 traits
                report.append(f"      • {trait['category']}: {trait['value']}")
                report.append(f"        └─ {trait['cluster_pct']:.1f}% in cluster vs {trait['overall_pct']:.1f}% overall ({trait['over_representation']:.1f}x)")
        else:
            report.append("      • No strongly distinctive traits found")
        
        report.append("")
        report.append("-" * 80)
        report.append("")
    
    report_text = "\n".join(report)
    
    with open(OUTPUT_DIR / "BULL_DARLINGS_TRAITS.txt", "w") as f:
        f.write(report_text)
    
    print(f"  ✓ Saved: BULL_DARLINGS_TRAITS.txt")
    print("\n" + report_text)
    
    return bull_darlings

def generate_bear_refuge_report(cluster_signatures):
    """
    Generate report for Bear Refuge clusters
    """
    print("\n🐻 Analyzing Bear Refuge Traits...")
    print("-" * 70)
    
    bear_refuges = [
        sig for sig in cluster_signatures
        if sig['aesthetic_type'] == 'bear_refuge'
    ]
    
    # Sort by price ratio (ascending for bear refuges)
    bear_refuges.sort(key=lambda x: x['price_bull_bear_ratio'])
    
    report = []
    report.append("=" * 80)
    report.append("BEAR MARKET REFUGES: Trait Analysis")
    report.append("=" * 80)
    report.append("")
    
    for sig in bear_refuges:
        relative_performance = 1 / sig['price_bull_bear_ratio']
        report.append(f"📊 Cluster {sig['cluster_id']}: {relative_performance:.2f}x Better in Bears")
        report.append(f"   Size: {sig['size']} NFTs")
        report.append(f"   Bull/Bear Ratio: {sig['price_bull_bear_ratio']:.2f}")
        report.append("")
        report.append("   🛡️ Distinctive Traits:")
        
        if sig['distinctive_traits']:
            for trait in sig['distinctive_traits'][:5]:
                report.append(f"      • {trait['category']}: {trait['value']}")
                report.append(f"        └─ {trait['cluster_pct']:.1f}% in cluster vs {trait['overall_pct']:.1f}% overall ({trait['over_representation']:.1f}x)")
        else:
            report.append("      • No strongly distinctive traits found")
        
        report.append("")
        report.append("-" * 80)
        report.append("")
    
    report_text = "\n".join(report)
    
    with open(OUTPUT_DIR / "BEAR_REFUGES_TRAITS.txt", "w") as f:
        f.write(report_text)
    
    print(f"  ✓ Saved: BEAR_REFUGES_TRAITS.txt")
    print("\n" + report_text)
    
    return bear_refuges

def create_trait_summary_table(cluster_signatures, df_merged, trait_columns):
    """
    Create a summary table of top traits per cluster
    """
    print("\n📋 Creating Trait Summary Table...")
    print("-" * 70)
    
    summary_data = []
    
    for sig in cluster_signatures:
        cluster_id = sig['cluster_id']
        cluster_data = df_merged[df_merged['visual_cluster'] == cluster_id]
        
        row = {
            'cluster_id': cluster_id,
            'size': sig['size'],
            'aesthetic_type': sig['aesthetic_type'],
            'bull_bear_ratio': sig['price_bull_bear_ratio'],
            'sensitivity_score': sig['cycle_sensitivity_score']
        }
        
        # Get most common trait in each category
        for trait_col in trait_columns:
            if trait_col in cluster_data.columns:
                most_common = cluster_data[trait_col].mode()
                if len(most_common) > 0:
                    row[f'top_{trait_col}'] = str(most_common.iloc[0])
                    # Get percentage
                    count = (cluster_data[trait_col] == most_common.iloc[0]).sum()
                    row[f'top_{trait_col}_pct'] = round(count / len(cluster_data) * 100, 1)
        
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data).sort_values('bull_bear_ratio', ascending=False)
    
    # Save
    df_summary.to_csv(OUTPUT_DIR / "cluster_trait_summary.csv", index=False)
    print(f"  ✓ Saved: cluster_trait_summary.csv")
    
    return df_summary

def main():
    """Main analysis pipeline"""
    print("\n" + "=" * 70)
    print("BAYC Cluster × Trait Analysis")
    print("=" * 70)
    
    # Load data
    df_nfts, df_collection, df_sensitivity = load_data()
    
    # Parse traits from JSON
    df_traits = parse_traits(df_collection)
    
    # Analyze trait composition by cluster
    cluster_traits, df_merged, trait_columns = analyze_cluster_trait_composition(
        df_nfts, df_traits, df_sensitivity
    )
    
    # Identify distinctive traits for each cluster
    cluster_signatures = identify_cluster_signatures(
        cluster_traits, df_sensitivity, trait_columns
    )
    
    # Generate reports
    bull_darlings = generate_bull_darling_report(cluster_signatures)
    bear_refuges = generate_bear_refuge_report(cluster_signatures)
    
    # Create summary table
    df_summary = create_trait_summary_table(cluster_signatures, df_merged, trait_columns)
    
    print("\n" + "=" * 70)
    print("✓ Cluster × Trait Analysis Complete!")
    print("=" * 70)
    print(f"\n📁 Output Directory: {OUTPUT_DIR}/")
    print("   • cluster_trait_distributions.json")
    print("   • cluster_trait_signatures.json")
    print("   • cluster_trait_summary.csv")
    print("   • BULL_DARLINGS_TRAITS.txt")
    print("   • BEAR_REFUGES_TRAITS.txt")
    
    print("\n💡 Key Insights:")
    print(f"   • {len(bull_darlings)} Bull Darling clusters analyzed")
    print(f"   • {len(bear_refuges)} Bear Refuge clusters analyzed")
    print(f"   • {len(trait_columns)} trait categories examined")

if __name__ == "__main__":
    main()
