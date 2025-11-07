"""
OpenSea NFT Data Visualization Analysis
Generate insightful charts to analyze NFT transaction data
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set font and style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

print("="*60)
print("OpenSea NFT Data Visualization Analysis")
print("="*60)

# Load processed minimal analytical dataset
DATA_PATH = '/Users/qingshen/Desktop/opensea/opensea_pipeline/clean/2025-10-31_02-04-53/minimal_events.parquet'
df = pl.read_parquet(DATA_PATH)

print(f"\nLoaded data: {len(df):,} transactions")
print(f"Collections: {df['collection'].n_unique()}")
print(f"Time span: {df['event_date'].min()} to {df['event_date'].max()}")

# Convert to pandas for plotting
df_pd = df.to_pandas()

# Create output directory
import os
output_dir = '/Users/qingshen/Desktop/opensea/opensea_pipeline/visualizations'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Question 1: Transaction Volume and Total Value by Collection
# ============================================================
print("\n\n[1/7] Generating: Collection Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Transaction count comparison
collection_counts = df.group_by('collection').agg([
    pl.count('event_id').alias('transactions')
]).sort('transactions', descending=True).to_pandas()

ax1.barh(collection_counts['collection'], collection_counts['transactions'], color=sns.color_palette("viridis", len(collection_counts)))
ax1.set_xlabel('Transaction Count', fontsize=12)
ax1.set_title('Total Transactions by Collection', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(collection_counts['transactions']):
    ax1.text(v, i, f' {v:,}', va='center', fontsize=10)

# Trading value comparison (only transactions with prices)
collection_volume = df.filter(pl.col('price_total_eth') > 0).group_by('collection').agg([
    pl.sum('price_total_eth').alias('total_volume_eth')
]).sort('total_volume_eth', descending=True).to_pandas()

ax2.barh(collection_volume['collection'], collection_volume['total_volume_eth'], color=sns.color_palette("rocket", len(collection_volume)))
ax2.set_xlabel('Total Volume (ETH)', fontsize=12)
ax2.set_title('Total Trading Volume by Collection', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(collection_volume['total_volume_eth']):
    ax2.text(v, i, f' {v:,.0f} ETH', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_collection_comparison.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 01_collection_comparison.png")

# ============================================================
# Question 2: Time Series - Daily Trading Trends (All Collections)
# ============================================================
print("\n[2/7] Generating: Daily Trends...")

daily_trend = df.group_by('event_date').agg([
    pl.count('event_id').alias('transactions'),
    pl.sum('price_total_eth').alias('volume_eth')
]).sort('event_date').to_pandas()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Daily transaction count
ax1.plot(daily_trend['event_date'], daily_trend['transactions'], linewidth=2, color='#2E86AB')
ax1.fill_between(daily_trend['event_date'], daily_trend['transactions'], alpha=0.3, color='#2E86AB')
ax1.set_ylabel('Daily Transactions', fontsize=12)
ax1.set_title('Daily Transaction Volume Over Time', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

# Daily trading volume
ax2.plot(daily_trend['event_date'], daily_trend['volume_eth'], linewidth=2, color='#A23B72')
ax2.fill_between(daily_trend['event_date'], daily_trend['volume_eth'], alpha=0.3, color='#A23B72')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Daily Volume (ETH)', fontsize=12)
ax2.set_title('Daily Trading Volume (ETH) Over Time', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_daily_trend.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 02_daily_trend.png")

# ============================================================
# Question 3: Price Distribution Box Plot by Collection
# ============================================================
print("\n[3/7] Generating: Price Distribution...")

priced_df = df.filter(
    (pl.col('price_each_eth') > 0) & 
    (pl.col('price_each_eth') < 100)  # Filter extreme outliers
).to_pandas()

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=priced_df, y='collection', x='price_each_eth', ax=ax, palette='Set2')
ax.set_xlabel('Price (ETH)', fontsize=12)
ax.set_ylabel('Collection', fontsize=12)
ax.set_title('Price Distribution by Collection (0-100 ETH)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_price_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 03_price_distribution.png")

# ============================================================
# Question 4: Event Type Distribution (Pie Chart + Bar Chart)
# ============================================================
print("\n[4/7] Generating: Event Type Distribution...")

event_distribution = df.group_by('event_type').agg([
    pl.count('event_id').alias('count')
]).sort('count', descending=True).to_pandas()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
colors = sns.color_palette("pastel", len(event_distribution))
ax1.pie(event_distribution['count'], labels=event_distribution['event_type'], 
        autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
ax1.set_title('Event Type Distribution', fontsize=14, fontweight='bold')

# Bar chart (by collection and event type)
event_by_collection = df.group_by(['collection', 'event_type']).agg([
    pl.count('event_id').alias('count')
]).to_pandas()
event_pivot = event_by_collection.pivot(index='collection', columns='event_type', values='count').fillna(0)
event_pivot.plot(kind='bar', stacked=True, ax=ax2, colormap='tab10')
ax2.set_ylabel('Transaction Count', fontsize=12)
ax2.set_xlabel('Collection', fontsize=12)
ax2.set_title('Event Type Distribution by Collection', fontsize=14, fontweight='bold')
ax2.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(axis='y', alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{output_dir}/04_event_type_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 04_event_type_distribution.png")

# ============================================================
# Question 5: Monthly Transaction Heatmap (Collection x Month)
# ============================================================
print("\n[5/7] Generating: Monthly Heatmap...")

# Add year-month column
df_monthly = df.with_columns([
    pl.col('event_date').dt.strftime('%Y-%m').alias('year_month')
])

monthly_heatmap = df_monthly.group_by(['collection', 'year_month']).agg([
    pl.count('event_id').alias('transactions')
]).to_pandas()

# Create pivot table
heatmap_pivot = monthly_heatmap.pivot(index='collection', columns='year_month', values='transactions').fillna(0)

# Show only last 24 months
if len(heatmap_pivot.columns) > 24:
    heatmap_pivot = heatmap_pivot[heatmap_pivot.columns[-24:]]

fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(heatmap_pivot, annot=False, fmt='g', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Transactions'})
ax.set_title('Monthly Transaction Heatmap by Collection (Last 24 Months)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Collection', fontsize=12)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{output_dir}/05_monthly_heatmap.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 05_monthly_heatmap.png")

# ============================================================
# Question 6: Average Price Trend (By Collection)
# ============================================================
print("\n[6/7] Generating: Price Trend Analysis...")

# Calculate monthly average price
monthly_price = df.filter(pl.col('price_each_eth') > 0).with_columns([
    pl.col('event_date').dt.strftime('%Y-%m').alias('year_month')
]).group_by(['collection', 'year_month']).agg([
    pl.mean('price_each_eth').alias('avg_price'),
    pl.median('price_each_eth').alias('median_price')
]).sort(['collection', 'year_month']).to_pandas()

fig, ax = plt.subplots(figsize=(16, 8))

for collection in monthly_price['collection'].unique():
    collection_data = monthly_price[monthly_price['collection'] == collection]
    ax.plot(collection_data['year_month'], collection_data['avg_price'], 
            marker='o', linewidth=2, label=collection, markersize=4)

ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Average Price (ETH)', fontsize=12)
ax.set_title('Average NFT Price Trend by Collection', fontsize=14, fontweight='bold')
ax.legend(title='Collection', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{output_dir}/06_price_trend.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 06_price_trend.png")

# ============================================================
# Question 7: Buyer/Seller Activity Analysis
# ============================================================
print("\n[7/7] Generating: Buyer/Seller Activity...")

# Most active buyers
top_buyers = df.filter(
    (pl.col('buyer').is_not_null()) & (pl.col('buyer') != '')
).group_by('buyer').agg([
    pl.count('event_id').alias('purchases'),
    pl.sum('price_total_eth').alias('total_spent')
]).sort('purchases', descending=True).head(15).to_pandas()

# Most active sellers
top_sellers = df.filter(
    (pl.col('seller').is_not_null()) & (pl.col('seller') != '')
).group_by('seller').agg([
    pl.count('event_id').alias('sales'),
    pl.sum('price_total_eth').alias('total_earned')
]).sort('sales', descending=True).head(15).to_pandas()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Top 15 buyers
buyer_labels = [f"{addr[:6]}...{addr[-4:]}" for addr in top_buyers['buyer']]
ax1.barh(buyer_labels, top_buyers['purchases'], color=sns.color_palette("Blues_r", 15))
ax1.set_xlabel('Number of Purchases', fontsize=12)
ax1.set_title('Top 15 Most Active Buyers', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_buyers['purchases']):
    ax1.text(v, i, f' {v:,}', va='center', fontsize=9)

# Top 15 sellers
seller_labels = [f"{addr[:6]}...{addr[-4:]}" for addr in top_sellers['seller']]
ax2.barh(seller_labels, top_sellers['sales'], color=sns.color_palette("Reds_r", 15))
ax2.set_xlabel('Number of Sales', fontsize=12)
ax2.set_title('Top 15 Most Active Sellers', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_sellers['sales']):
    ax2.text(v, i, f' {v:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/07_buyer_seller_activity.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 07_buyer_seller_activity.png")

# ============================================================
# Generate Statistical Summary
# ============================================================
print("\n\n" + "="*60)
print("Data Statistical Summary")
print("="*60)

print(f"\nTotal transactions: {len(df):,}")
print(f"Total collections: {df['collection'].n_unique()}")
print(f"Unique NFTs: {df['token_id'].n_unique():,}")
print(f"Time span: {df['event_date'].min()} to {df['event_date'].max()}")

priced = df.filter(pl.col('price_total_eth') > 0)
print(f"\nPriced transactions: {len(priced):,} ({len(priced)/len(df)*100:.1f}%)")
print(f"Total volume: {priced['price_total_eth'].sum():,.2f} ETH")
print(f"Average price: {priced['price_each_eth'].mean():.2f} ETH")
print(f"Median price: {priced['price_each_eth'].median():.2f} ETH")

print("\nTop 5 Collections by Transaction Volume:")
top5 = df.group_by('collection').agg([
    pl.count('event_id').alias('transactions')
]).sort('transactions', descending=True).head(5)
for row in top5.iter_rows(named=True):
    print(f"  {row['collection']:20s} {row['transactions']:>8,} txns")

print("\n" + "="*60)
print(f"All charts saved to: {output_dir}/")
print("="*60)
print("\nGenerated charts:")
print("  1. 01_collection_comparison.png    - Collection volume & value comparison")
print("  2. 02_daily_trend.png              - Daily trading trends")
print("  3. 03_price_distribution.png       - Price distribution box plots")
print("  4. 04_event_type_distribution.png  - Event type distribution")
print("  5. 05_monthly_heatmap.png          - Monthly transaction heatmap")
print("  6. 06_price_trend.png              - Price trend analysis")
print("  7. 07_buyer_seller_activity.png    - Buyer/seller activity")
print("\n✓ Visualization analysis complete!")
