

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("="*70)
print("Advanced Time Series Analysis - Pattern Recognition")
print("="*70)

# Load data
DATA_PATH = '/Users/qingshen/Desktop/opensea/opensea_pipeline/clean/2025-10-31_02-04-53/minimal_events.parquet'
df = pl.read_parquet(DATA_PATH)

output_dir = '/Users/qingshen/Desktop/opensea/opensea_pipeline/visualizations'

# ============================================================
# Pattern 1: Weekly Trading Pattern (Weekday vs Weekend)
# ============================================================
print("\n[Pattern 1] Analyzing weekly trading patterns...")

df_weekly = df.with_columns([
    pl.col('event_date').dt.weekday().alias('weekday'),
    pl.col('event_date').dt.strftime('%A').alias('day_name')
])

weekday_stats = df_weekly.group_by('weekday').agg([
    pl.count('event_id').alias('transactions'),
    pl.sum('price_total_eth').alias('volume_eth'),
    pl.mean('price_each_eth').alias('avg_price')
]).sort('weekday').to_pandas()

# Add day names
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_stats['day_name'] = day_names

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1.1 Weekly transaction volume
ax = axes[0, 0]
bars = ax.bar(weekday_stats['day_name'], weekday_stats['transactions'], 
              color=['#FF6B6B' if d in ['Saturday', 'Sunday'] else '#4ECDC4' for d in day_names])
ax.set_ylabel('Total Transactions', fontsize=12)
ax.set_title('Trading Volume by Day of Week', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(weekday_stats['transactions']):
    ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 1.2 Weekly trading volume
ax = axes[0, 1]
ax.bar(weekday_stats['day_name'], weekday_stats['volume_eth'], 
       color=['#FF6B6B' if d in ['Saturday', 'Sunday'] else '#95E1D3' for d in day_names])
ax.set_ylabel('Trading Volume (ETH)', fontsize=12)
ax.set_title('Trading Volume (ETH) by Day of Week', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 1.3 Weekly average price
ax = axes[1, 0]
ax.plot(weekday_stats['day_name'], weekday_stats['avg_price'], 
        marker='o', linewidth=2, markersize=10, color='#F38181')
ax.set_ylabel('Average Price (ETH)', fontsize=12)
ax.set_title('Average NFT Price by Day of Week', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 1.4 Weekly pattern by collection
ax = axes[1, 1]
for collection in df['collection'].unique().sort():
    coll_weekly = df_weekly.filter(pl.col('collection') == collection).group_by('weekday').agg([
        pl.count('event_id').alias('transactions')
    ]).sort('weekday').to_pandas()
    ax.plot(day_names, coll_weekly['transactions'], marker='o', linewidth=2, label=collection, alpha=0.7)
ax.set_ylabel('Transactions', fontsize=12)
ax.set_title('Weekly Pattern by Collection', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{output_dir}/08_weekly_pattern.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 08_weekly_pattern.png")

# ============================================================
# Pattern 2: Hourly Trading Pattern (24-hour heatmap)
# ============================================================
print("\n[Pattern 2] Analyzing 24-hour trading patterns...")

df_hourly = df.with_columns([
    pl.col('event_timestamp').dt.hour().alias('hour'),
    pl.col('event_date').dt.weekday().alias('weekday')
])

hourly_heatmap = df_hourly.group_by(['hour', 'weekday']).agg([
    pl.count('event_id').alias('transactions')
]).to_pandas()

# Create pivot table
heatmap_pivot = hourly_heatmap.pivot(index='hour', columns='weekday', values='transactions').fillna(0)
heatmap_pivot.columns = day_names

fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(heatmap_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, 
            cbar_kws={'label': 'Transactions'}, linewidths=0.5)
ax.set_title('24-Hour Trading Heatmap (UTC) by Day of Week', fontsize=16, fontweight='bold')
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Hour (UTC)', fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/09_hourly_heatmap.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 09_hourly_heatmap.png")

# ============================================================
# Pattern 3: 7-day/30-day Moving Average + Volatility Analysis
# ============================================================
print("\n[Pattern 3] Calculating moving averages and volatility...")

daily_stats = df.group_by('event_date').agg([
    pl.count('event_id').alias('transactions'),
    pl.sum('price_total_eth').alias('volume_eth'),
    pl.mean('price_each_eth').alias('avg_price')
]).sort('event_date').to_pandas()

# Calculate moving averages
daily_stats['ma_7'] = daily_stats['transactions'].rolling(window=7, min_periods=1).mean()
daily_stats['ma_30'] = daily_stats['transactions'].rolling(window=30, min_periods=1).mean()
daily_stats['volatility_7'] = daily_stats['transactions'].rolling(window=7, min_periods=1).std()

fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# 3.1 Transaction volume + moving averages
ax = axes[0]
ax.plot(daily_stats['event_date'], daily_stats['transactions'], 
        alpha=0.3, linewidth=1, color='gray', label='Daily')
ax.plot(daily_stats['event_date'], daily_stats['ma_7'], 
        linewidth=2, color='#FF6B6B', label='7-Day MA')
ax.plot(daily_stats['event_date'], daily_stats['ma_30'], 
        linewidth=2, color='#4ECDC4', label='30-Day MA')
ax.set_ylabel('Transactions', fontsize=12)
ax.set_title('Daily Transactions with Moving Averages', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(alpha=0.3)

# 3.2 Trading volume + moving averages
daily_stats['volume_ma_7'] = daily_stats['volume_eth'].rolling(window=7, min_periods=1).mean()
daily_stats['volume_ma_30'] = daily_stats['volume_eth'].rolling(window=30, min_periods=1).mean()

ax = axes[1]
ax.plot(daily_stats['event_date'], daily_stats['volume_eth'], 
        alpha=0.3, linewidth=1, color='gray', label='Daily')
ax.plot(daily_stats['event_date'], daily_stats['volume_ma_7'], 
        linewidth=2, color='#FFD93D', label='7-Day MA')
ax.plot(daily_stats['event_date'], daily_stats['volume_ma_30'], 
        linewidth=2, color='#6BCB77', label='30-Day MA')
ax.set_ylabel('Volume (ETH)', fontsize=12)
ax.set_title('Daily Trading Volume (ETH) with Moving Averages', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(alpha=0.3)

# 3.3 Volatility
ax = axes[2]
ax.fill_between(daily_stats['event_date'], daily_stats['volatility_7'], 
                alpha=0.5, color='#C44569', label='7-Day Volatility')
ax.plot(daily_stats['event_date'], daily_stats['volatility_7'], 
        linewidth=2, color='#8B0000')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Transaction Volatility (Std Dev)', fontsize=12)
ax.set_title('Market Volatility (7-Day Rolling Std Dev)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/10_moving_averages.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 10_moving_averages.png")

# ============================================================
# Pattern 4: Year-over-Year Monthly Analysis
# ============================================================
print("\n[Pattern 4] Year-over-year monthly analysis...")

df_monthly = df.with_columns([
    pl.col('event_date').dt.year().alias('year'),
    pl.col('event_date').dt.month().alias('month')
])

monthly_yoy = df_monthly.group_by(['year', 'month']).agg([
    pl.count('event_id').alias('transactions'),
    pl.sum('price_total_eth').alias('volume_eth')
]).sort(['year', 'month']).to_pandas()

monthly_yoy['year_month'] = monthly_yoy.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1)

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# 4.1 Monthly transaction volume grouped by year
ax = axes[0]
for year in sorted(monthly_yoy['year'].unique()):
    year_data = monthly_yoy[monthly_yoy['year'] == year]
    ax.plot(year_data['month'], year_data['transactions'], 
            marker='o', linewidth=2.5, markersize=8, label=f'{int(year)}', alpha=0.8)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Transactions', fontsize=12)
ax.set_title('Year-over-Year Monthly Transaction Comparison', fontsize=14, fontweight='bold')
ax.legend(title='Year', fontsize=11)
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# 4.2 Monthly trading volume grouped by year
ax = axes[1]
for year in sorted(monthly_yoy['year'].unique()):
    year_data = monthly_yoy[monthly_yoy['year'] == year]
    ax.plot(year_data['month'], year_data['volume_eth'], 
            marker='s', linewidth=2.5, markersize=8, label=f'{int(year)}', alpha=0.8)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Volume (ETH)', fontsize=12)
ax.set_title('Year-over-Year Monthly Volume Comparison', fontsize=14, fontweight='bold')
ax.legend(title='Year', fontsize=11)
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.tight_layout()
plt.savefig(f'{output_dir}/11_year_over_year.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 11_year_over_year.png")

# ============================================================
# Pattern 5: Collection Lifecycle Analysis
# ============================================================
print("\n[Pattern 5] Collection lifecycle analysis...")

# ============================================================
# Pattern 5: 各集合的生命周期分析
# ============================================================
print("\n[Pattern 5] 集合生命周期分析...")

collection_lifecycle = df.group_by(['collection', 'event_date']).agg([
    pl.count('event_id').alias('transactions'),
    pl.mean('price_each_eth').alias('avg_price')
]).sort(['collection', 'event_date']).to_pandas()

# Calculate cumulative transaction volume for each collection
collection_lifecycle['cumulative_transactions'] = collection_lifecycle.groupby('collection')['transactions'].cumsum()

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# 5.1 Daily transaction volume comparison by collection
ax = axes[0]
for collection in sorted(df['collection'].unique()):
    coll_data = collection_lifecycle[collection_lifecycle['collection'] == collection]
    ax.plot(coll_data['event_date'], coll_data['transactions'], 
            linewidth=1.5, label=collection, alpha=0.7)
ax.set_ylabel('Daily Transactions', fontsize=12)
ax.set_title('Collection Lifecycle - Daily Transaction Activity', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax.grid(alpha=0.3)

# 5.2 Cumulative transaction volume (growth curves)
ax = axes[1]
for collection in sorted(df['collection'].unique()):
    coll_data = collection_lifecycle[collection_lifecycle['collection'] == collection]
    ax.plot(coll_data['event_date'], coll_data['cumulative_transactions'], 
            linewidth=2.5, label=collection, alpha=0.8)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Transactions', fontsize=12)
ax.set_title('Collection Growth Curves - Cumulative Transactions', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/12_lifecycle_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 12_lifecycle_analysis.png")

# ============================================================
# Pattern 6: Anomaly Detection - Identify Abnormal Trading Days
# ============================================================
print("\n[Pattern 6] Anomaly detection...")

# Calculate Z-score to identify anomalies
from scipy import stats

daily_stats['volume_zscore'] = np.abs(stats.zscore(daily_stats['transactions']))
daily_stats['price_zscore'] = np.abs(stats.zscore(daily_stats['avg_price'].fillna(0)))

# Mark anomalies (Z-score > 3)
anomalies = daily_stats[(daily_stats['volume_zscore'] > 3) | (daily_stats['price_zscore'] > 3)].copy()
anomalies = anomalies.sort_values('event_date', ascending=False).head(20)

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# 6.1 Transaction volume anomalies
ax = axes[0]
ax.plot(daily_stats['event_date'], daily_stats['transactions'], 
        linewidth=1.5, color='#3498db', alpha=0.7, label='Daily Transactions')
ax.scatter(anomalies['event_date'], anomalies['transactions'], 
           color='red', s=200, zorder=5, marker='*', label='Anomalies (Z>3)', edgecolors='black', linewidths=2)
ax.set_ylabel('Transactions', fontsize=12)
ax.set_title('Transaction Volume Anomaly Detection', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# 6.2 Price anomalies
ax = axes[1]
ax.plot(daily_stats['event_date'], daily_stats['avg_price'], 
        linewidth=1.5, color='#2ecc71', alpha=0.7, label='Daily Avg Price')
price_anomalies = anomalies[anomalies['price_zscore'] > 3]
ax.scatter(price_anomalies['event_date'], price_anomalies['avg_price'], 
           color='orange', s=200, zorder=5, marker='*', label='Price Anomalies (Z>3)', edgecolors='black', linewidths=2)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Average Price (ETH)', fontsize=12)
ax.set_title('Price Anomaly Detection', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/13_anomaly_detection.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 13_anomaly_detection.png")

# ============================================================
# Pattern 7: Collection Price Correlation Analysis
# ============================================================
print("\n[Pattern 7] Collection price correlation...")

# Calculate daily average price for each collection
price_correlation = df.filter(pl.col('price_each_eth') > 0).group_by(['collection', 'event_date']).agg([
    pl.mean('price_each_eth').alias('avg_price')
]).to_pandas()

# Pivot to wide format
price_pivot = price_correlation.pivot(index='event_date', columns='collection', values='avg_price')

# Calculate correlation matrix
correlation_matrix = price_pivot.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, ax=ax,
            cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('NFT Collection Price Correlation Matrix', fontsize=16, fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(f'{output_dir}/14_price_correlation.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: 14_price_correlation.png")

# ============================================================
# Generate Pattern Analysis Report
# ============================================================
print("\n\n" + "="*70)
print("Pattern Analysis Summary")
print("="*70)

print("\n🔍 Key Patterns Identified:")

print("\n1️⃣  Weekly Pattern:")
weekday_peak = weekday_stats.loc[weekday_stats['transactions'].idxmax()]
weekday_low = weekday_stats.loc[weekday_stats['transactions'].idxmin()]
print(f"   - Peak trading day: {weekday_peak['day_name']} ({weekday_peak['transactions']:,} txns)")
print(f"   - Low trading day: {weekday_low['day_name']} ({weekday_low['transactions']:,} txns)")
print(f"   - Fluctuation range: {(weekday_peak['transactions']/weekday_low['transactions']-1)*100:.1f}%")

print("\n2️⃣  Monthly Trend:")
recent_months = monthly_yoy.tail(6)
print(f"   - Last 6 months avg volume: {recent_months['transactions'].mean():,.0f} txns/month")
print(f"   - Most active month: {monthly_yoy.loc[monthly_yoy['transactions'].idxmax(), 'year_month']}")

print("\n3️⃣  Volatility:")
avg_volatility = daily_stats['volatility_7'].mean()
recent_volatility = daily_stats['volatility_7'].tail(30).mean()
print(f"   - Historical avg volatility: {avg_volatility:,.0f}")
print(f"   - Recent 30-day volatility: {recent_volatility:,.0f}")
print(f"   - Market status: {'Stable' if recent_volatility < avg_volatility else 'Increased volatility'}")

print("\n4️⃣  Anomalous Events:")
print(f"   - Detected {len(anomalies)} anomalous trading days")
if len(anomalies) > 0:
    top_anomaly = anomalies.iloc[0]
    print(f"   - Largest anomaly: {top_anomaly['event_date']} ({top_anomaly['transactions']:,} txns)")

print("\n5️⃣  Collection Correlation:")
# Find collection pairs with highest correlation
corr_values = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_values.append({
            'pair': f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}",
            'correlation': correlation_matrix.iloc[i, j]
        })
if corr_values:
    import pandas as pd
    corr_df = pd.DataFrame(corr_values).sort_values('correlation', ascending=False)
    print(f"   - Highest positive correlation: {corr_df.iloc[0]['pair']} ({corr_df.iloc[0]['correlation']:.2f})")
    print(f"   - Lowest correlation: {corr_df.iloc[-1]['pair']} ({corr_df.iloc[-1]['correlation']:.2f})")

print("\n" + "="*70)
print(f"All advanced time series charts saved to: {output_dir}/")
print("="*70)
print("\nNew charts (08-14):")
print("  08. weekly_pattern.png         - Weekly trading patterns (Mon-Sun)")
print("  09. hourly_heatmap.png         - 24-hour trading heatmap")
print("  10. moving_averages.png        - 7-day/30-day moving averages + volatility")
print("  11. year_over_year.png         - Year-over-year monthly comparison")
print("  12. lifecycle_analysis.png     - Collection lifecycle & growth curves")
print("  13. anomaly_detection.png      - Anomaly detection")
print("  14. price_correlation.png      - Price correlation matrix")
print("\n✓ Advanced time series analysis complete!")
