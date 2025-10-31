"""
Aggregation module for creating analytical datasets.
Generates daily statistics, token metrics, and dimension tables.
"""

from typing import List
import polars as pl


class EventAggregator:
    """Aggregates event data into analytical datasets."""
    
    def __init__(self):
        pass
    
    def create_daily_collection_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create daily collection-level statistics.
        
        Args:
            df: Cleaned event DataFrame
            
        Returns:
            DataFrame with daily collection statistics
        """
        print("Creating daily collection statistics...")
        
        # Filter to only sales/transfers with prices for volume calculations
        priced_events = df.filter(pl.col('price_total_eth') > 0)
        
        # Count different event types
        event_counts = df.pivot(
            index=['collection', 'event_date'],
            columns='event_type',
            values='event_id',
            aggregate_function='count'
        )
        
        # Calculate statistics per collection per day
        stats = df.group_by(['collection', 'event_date']).agg([
            pl.count('event_id').alias('total_transactions'),
            pl.col('buyer').n_unique().alias('unique_buyers'),
            pl.col('seller').n_unique().alias('unique_sellers'),
            pl.col('token_id').n_unique().alias('unique_tokens_traded'),
        ])
        
        # Calculate price statistics from priced events
        price_stats = priced_events.group_by(['collection', 'event_date']).agg([
            pl.sum('price_total_eth').alias('total_volume_eth'),
            pl.mean('price_each_eth').alias('avg_price_eth'),
            pl.median('price_each_eth').alias('median_price_eth'),
            pl.min('price_each_eth').alias('min_price_eth'),
            pl.max('price_each_eth').alias('max_price_eth'),
        ])
        
        # Join stats with price stats
        stats = stats.join(price_stats, on=['collection', 'event_date'], how='left')
        
        # Add event type counts if available
        if 'mint' in event_counts.columns:
            stats = stats.join(
                event_counts.select(['collection', 'event_date', 'mint']).rename({'mint': 'mint_count'}),
                on=['collection', 'event_date'],
                how='left'
            )
        else:
            stats = stats.with_columns([pl.lit(0).alias('mint_count')])
        
        if 'sale' in event_counts.columns:
            stats = stats.join(
                event_counts.select(['collection', 'event_date', 'sale']).rename({'sale': 'sale_count'}),
                on=['collection', 'event_date'],
                how='left'
            )
        else:
            stats = stats.with_columns([pl.lit(0).alias('sale_count')])
        
        if 'transfer' in event_counts.columns:
            stats = stats.join(
                event_counts.select(['collection', 'event_date', 'transfer']).rename({'transfer': 'transfer_count'}),
                on=['collection', 'event_date'],
                how='left'
            )
        else:
            stats = stats.with_columns([pl.lit(0).alias('transfer_count')])
        
        # Fill nulls with 0
        stats = stats.fill_null(0)
        
        # Sort by collection and date
        stats = stats.sort(['collection', 'event_date'])
        
        print(f"  Created {len(stats)} daily collection records")
        return stats
    
    def create_token_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create per-token statistics.
        
        Args:
            df: Cleaned event DataFrame
            
        Returns:
            DataFrame with token-level statistics
        """
        print("Creating token statistics...")
        
        # Get first mint date
        mint_dates = df.filter(pl.col('event_type') == 'mint').group_by(['collection', 'token_id']).agg([
            pl.col('event_date').min().alias('first_mint_date')
        ])
        
        # Get last trade date and price
        last_trade = df.filter(
            (pl.col('event_type').is_in(['sale', 'transfer'])) & 
            (pl.col('price_total_eth') > 0)
        ).group_by(['collection', 'token_id']).agg([
            pl.col('event_date').max().alias('last_trade_date'),
            pl.col('price_each_eth').last().alias('last_price_eth')
        ])
        
        # Calculate overall token statistics
        token_stats = df.group_by(['collection', 'token_id', 'contract_address']).agg([
            pl.count('event_id').alias('total_trades'),
            pl.sum('price_total_eth').alias('total_volume_eth'),
            pl.mean('price_each_eth').alias('avg_price_eth'),
            pl.col('to_address').n_unique().alias('unique_owners'),
            pl.col('rarity_rank').first().alias('rarity_rank'),
            pl.col('rarity_score').first().alias('rarity_score'),
        ])
        
        # Join all statistics
        token_stats = token_stats.join(mint_dates, on=['collection', 'token_id'], how='left')
        token_stats = token_stats.join(last_trade, on=['collection', 'token_id'], how='left')
        
        # Sort by collection and token_id
        token_stats = token_stats.sort(['collection', 'token_id'])
        
        print(f"  Created statistics for {len(token_stats)} tokens")
        return token_stats
    
    def create_collection_dimension(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create collection dimension table.
        
        Args:
            df: Cleaned event DataFrame
            
        Returns:
            DataFrame with collection dimensions
        """
        print("Creating collection dimension...")
        
        collection_dim = df.group_by(['collection']).agg([
            pl.col('contract_address').first().alias('contract_address'),
            pl.col('event_date').min().alias('first_event_date'),
            pl.col('event_date').max().alias('last_event_date'),
            pl.col('token_id').n_unique().alias('total_tokens'),
            pl.count('event_id').alias('total_transactions'),
        ])
        
        collection_dim = collection_dim.sort('collection')
        
        print(f"  Created dimension for {len(collection_dim)} collections")
        return collection_dim
    
    def create_wallet_dimension(self, df: pl.DataFrame, min_transactions: int = 1) -> pl.DataFrame:
        """
        Create wallet dimension table.
        
        Args:
            df: Cleaned event DataFrame
            min_transactions: Minimum transactions to include a wallet
            
        Returns:
            DataFrame with wallet dimensions
        """
        print(f"Creating wallet dimension (min {min_transactions} transactions)...")
        
        # Get buyer statistics
        buyer_stats = df.filter(pl.col('buyer').is_not_null() & (pl.col('buyer') != '')).group_by('buyer').agg([
            pl.col('event_date').min().alias('first_buy_date'),
            pl.col('event_date').max().alias('last_buy_date'),
            pl.count('event_id').alias('total_purchases'),
            pl.sum('price_total_eth').alias('total_volume_bought_eth'),
            pl.col('collection').n_unique().alias('collections_bought'),
        ])
        
        # Get seller statistics
        seller_stats = df.filter(pl.col('seller').is_not_null() & (pl.col('seller') != '')).group_by('seller').agg([
            pl.col('event_date').min().alias('first_sell_date'),
            pl.col('event_date').max().alias('last_sell_date'),
            pl.count('event_id').alias('total_sales'),
            pl.sum('price_total_eth').alias('total_volume_sold_eth'),
            pl.col('collection').n_unique().alias('collections_sold'),
        ])
        
        # Combine buyer and seller data
        buyer_stats = buyer_stats.rename({'buyer': 'wallet_address'})
        seller_stats = seller_stats.rename({'seller': 'wallet_address'})
        
        wallet_dim = buyer_stats.join(seller_stats, on='wallet_address', how='outer')
        
        # Calculate overall statistics
        wallet_dim = wallet_dim.with_columns([
            pl.min_horizontal(['first_buy_date', 'first_sell_date']).alias('first_activity_date'),
            pl.max_horizontal(['last_buy_date', 'last_sell_date']).alias('last_activity_date'),
            pl.col('total_purchases').fill_null(0),
            pl.col('total_sales').fill_null(0),
            pl.col('total_volume_bought_eth').fill_null(0.0),
            pl.col('total_volume_sold_eth').fill_null(0.0),
            pl.max_horizontal(['collections_bought', 'collections_sold']).alias('unique_collections'),
        ])
        
        # Filter by minimum transactions
        wallet_dim = wallet_dim.filter(
            (pl.col('total_purchases') + pl.col('total_sales')) >= min_transactions
        )
        
        # Select final columns
        wallet_dim = wallet_dim.select([
            'wallet_address',
            'first_activity_date',
            'last_activity_date',
            'total_purchases',
            'total_sales',
            'total_volume_bought_eth',
            'total_volume_sold_eth',
            'unique_collections',
        ])
        
        wallet_dim = wallet_dim.sort('wallet_address')
        
        print(f"  Created dimension for {len(wallet_dim)} wallets")
        return wallet_dim
    
    def create_collection_summary(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create high-level collection summary statistics.
        
        Args:
            df: Cleaned event DataFrame
            
        Returns:
            DataFrame with collection summaries
        """
        print("Creating collection summary...")
        
        priced_events = df.filter(pl.col('price_total_eth') > 0)
        
        summary = df.group_by('collection').agg([
            pl.count('event_id').alias('total_events'),
            pl.col('token_id').n_unique().alias('unique_tokens'),
            pl.col('buyer').n_unique().alias('unique_buyers'),
            pl.col('seller').n_unique().alias('unique_sellers'),
            pl.col('event_date').min().alias('first_event_date'),
            pl.col('event_date').max().alias('last_event_date'),
        ])
        
        price_summary = priced_events.group_by('collection').agg([
            pl.sum('price_total_eth').alias('total_volume_eth'),
            pl.mean('price_each_eth').alias('avg_price_eth'),
            pl.median('price_each_eth').alias('median_price_eth'),
            pl.col('price_each_eth').quantile(0.25).alias('p25_price_eth'),
            pl.col('price_each_eth').quantile(0.75).alias('p75_price_eth'),
            pl.min('price_each_eth').alias('min_price_eth'),
            pl.max('price_each_eth').alias('max_price_eth'),
        ])
        
        summary = summary.join(price_summary, on='collection', how='left')
        summary = summary.sort('collection')
        
        print(f"  Created summary for {len(summary)} collections")
        return summary


def calculate_time_series_metrics(df: pl.DataFrame, collection: str = None) -> pl.DataFrame:
    """
    Calculate time series metrics for trend analysis.
    
    Args:
        df: Cleaned event DataFrame
        collection: Optional collection filter
        
    Returns:
        DataFrame with time series metrics
    """
    if collection:
        df = df.filter(pl.col('collection') == collection)
    
    # Daily time series
    ts = df.group_by('event_date').agg([
        pl.count('event_id').alias('daily_transactions'),
        pl.sum('price_total_eth').alias('daily_volume_eth'),
        pl.mean('price_each_eth').alias('daily_avg_price_eth'),
        pl.col('token_id').n_unique().alias('daily_unique_tokens'),
    ])
    
    # Sort by date
    ts = ts.sort('event_date')
    
    # Calculate rolling averages (7-day)
    ts = ts.with_columns([
        pl.col('daily_transactions').rolling_mean(window_size=7).alias('ma7_transactions'),
        pl.col('daily_volume_eth').rolling_mean(window_size=7).alias('ma7_volume_eth'),
        pl.col('daily_avg_price_eth').rolling_mean(window_size=7).alias('ma7_avg_price_eth'),
    ])
    
    return ts
