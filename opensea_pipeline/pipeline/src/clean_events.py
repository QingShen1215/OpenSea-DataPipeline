"""
Event cleaning and transformation module.
Handles field normalization, deduplication, and price calculations.
"""

from typing import Optional
import polars as pl
from schemas import EventSchema, NULL_ADDRESS


class EventCleaner:
    """Cleans and transforms raw event data into analytical format."""
    
    def __init__(self):
        self.schema = EventSchema()
        
    def clean_events(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Main cleaning pipeline for event data.
        
        Args:
            df: Raw event DataFrame
            
        Returns:
            Cleaned DataFrame ready for analysis
        """
        print("Starting event cleaning pipeline...")
        print(f"Input rows: {len(df):,}")
        
        # Step 1: Normalize field types
        df = self._normalize_types(df)
        
        # Step 2: Parse and standardize timestamps
        df = self._clean_timestamps(df)
        
        # Step 3: Clean and validate addresses
        df = self._clean_addresses(df)
        
        # Step 4: Fix and derive prices
        df = self._clean_prices(df)
        
        # Step 5: Standardize event types
        df = self._clean_event_types(df)
        
        # Step 6: Remove duplicates
        df = self._deduplicate(df)
        
        # Step 7: Create composite keys
        df = self._create_keys(df)
        
        # Step 8: Select and rename to minimal schema
        df = self._to_minimal_schema(df)
        
        print(f"Output rows: {len(df):,}")
        print("Event cleaning completed!")
        
        return df
    
    def _normalize_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize data types for key columns."""
        print("  → Normalizing data types...")
        
        return df.with_columns([
            # Ensure strings
            pl.col('chain').cast(pl.Utf8).fill_null('ethereum'),
            pl.col('collection').cast(pl.Utf8),
            pl.col('event_type').cast(pl.Utf8),
            pl.col('tx').cast(pl.Utf8),
            pl.col('contract').cast(pl.Utf8),
            pl.col('token_id').cast(pl.Utf8),
            pl.col('currency_symbol').cast(pl.Utf8).fill_null('ETH'),
            
            # Ensure numeric types
            pl.col('timestamp').cast(pl.Int64),
            pl.col('quantity').cast(pl.Int64).fill_null(1),
            pl.col('price_total').cast(pl.Float64).fill_null(0.0),
        ])
    
    def _clean_timestamps(self, df: pl.DataFrame) -> pl.DataFrame:
        """Parse and validate timestamp fields."""
        print("  → Cleaning timestamps...")
        
        # Check if time_utc is already a datetime
        if df['time_utc'].dtype in [pl.Datetime, pl.Datetime('us'), pl.Datetime('ms'), pl.Datetime('ns')]:
            # Convert to UTC and remove timezone info for consistency
            parsed_datetime = pl.col('time_utc').dt.convert_time_zone('UTC').dt.replace_time_zone(None)
        else:
            # Parse UTC timestamp string to datetime
            parsed_datetime = pl.col('time_utc').str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S%z', strict=False)
        
        df = df.with_columns([
            parsed_datetime.alias('parsed_datetime'),
            # Convert unix timestamp to datetime
            pl.from_epoch('timestamp', time_unit='s').alias('datetime_from_unix')
        ])
        
        # Use parsed_datetime if available, otherwise use datetime_from_unix
        df = df.with_columns([
            pl.coalesce([pl.col('parsed_datetime'), pl.col('datetime_from_unix')])
                .alias('event_timestamp')
        ])
        
        # Extract date
        df = df.with_columns([
            pl.col('event_timestamp').cast(pl.Date).alias('event_date')
        ])
        
        # Drop intermediate columns
        df = df.drop(['parsed_datetime', 'datetime_from_unix'])
        
        return df
    
    def _clean_addresses(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean and standardize Ethereum addresses."""
        print("  → Cleaning addresses...")
        
        address_cols = ['seller', 'buyer', 'from_address', 'to_address', 'contract']
        
        for col in address_cols:
            if col not in df.columns:
                continue
            
            df = df.with_columns([
                # Lowercase all addresses for consistency
                pl.col(col)
                    .str.to_lowercase()
                    .str.strip_chars()
                    .fill_null('')
                    .alias(col)
            ])
        
        # Identify mints (from null address)
        df = df.with_columns([
            (pl.col('from_address') == NULL_ADDRESS).alias('is_mint')
        ])
        
        return df
    
    def _clean_prices(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean prices and derive price_each if missing."""
        print("  → Cleaning and deriving prices...")
        
        # Convert price_each to float, handling empty strings and various types
        if 'price_each' in df.columns:
            df = df.with_columns([
                pl.when(
                    (pl.col('price_each').is_null()) | 
                    (pl.col('price_each').cast(pl.Utf8) == '') |
                    (pl.col('price_each').cast(pl.Utf8).str.strip_chars() == '')
                )
                .then(None)
                .otherwise(pl.col('price_each').cast(pl.Float64, strict=False))
                .alias('price_each_parsed')
            ])
        else:
            df = df.with_columns([
                pl.lit(None).cast(pl.Float64).alias('price_each_parsed')
            ])
        
        # Derive price_each from price_total / quantity if missing
        df = df.with_columns([
            pl.coalesce([
                pl.col('price_each_parsed'),
                pl.when(pl.col('quantity') > 0)
                    .then(pl.col('price_total') / pl.col('quantity'))
                    .otherwise(pl.col('price_total'))
            ]).alias('price_each_eth')
        ])
        
        # Rename price_total for clarity
        df = df.with_columns([
            pl.col('price_total').alias('price_total_eth')
        ])
        
        # Filter out negative prices (likely data errors)
        original_len = len(df)
        df = df.filter(
            (pl.col('price_total_eth') >= 0) & 
            (pl.col('price_each_eth') >= 0)
        )
        removed = original_len - len(df)
        if removed > 0:
            print(f"    Removed {removed} rows with negative prices")
        
        return df
    
    def _clean_event_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize event type values."""
        print("  → Standardizing event types...")
        
        # Lowercase and strip whitespace
        df = df.with_columns([
            pl.col('event_type').str.to_lowercase().str.strip_chars()
        ])
        
        # Map common variations to standard types
        event_mapping = {
            'transfer': 'transfer',
            'sale': 'sale',
            'mint': 'mint',
            'listing': 'list',
            'list': 'list',
            'cancel_listing': 'cancel_list',
            'cancel_list': 'cancel_list',
            'offer': 'offer',
            'cancel_offer': 'cancel_offer',
        }
        
        # Apply mapping
        mapping_expr = pl.col('event_type')
        for old, new in event_mapping.items():
            mapping_expr = pl.when(pl.col('event_type') == old).then(pl.lit(new)).otherwise(mapping_expr)
        
        df = df.with_columns([mapping_expr.alias('event_type')])
        
        return df
    
    def _deduplicate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove duplicate events."""
        print("  → Removing duplicates...")
        
        original_len = len(df)
        
        # Define uniqueness: same tx, token_id, event_type, and timestamp
        df = df.unique(subset=['tx', 'token_id', 'event_type', 'timestamp'], keep='first')
        
        removed = original_len - len(df)
        if removed > 0:
            print(f"    Removed {removed} duplicate rows")
        
        return df
    
    def _create_keys(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create composite keys for event identification."""
        print("  → Creating composite keys...")
        
        df = df.with_columns([
            # Event ID: combination of tx hash and token_id
            (pl.col('tx') + '_' + pl.col('token_id')).alias('event_id'),
            
            # Rename tx to tx_hash for clarity
            pl.col('tx').alias('tx_hash'),
            
            # Rename contract to contract_address
            pl.col('contract').alias('contract_address')
        ])
        
        return df
    
    def _to_minimal_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Select and rename columns to match minimal analytical schema."""
        print("  → Transforming to minimal schema...")
        
        # Add rarity fields if they exist
        if 'rarity_rank' in df.columns:
            rarity_rank = pl.col('rarity_rank').cast(pl.Int64, strict=False)
        else:
            rarity_rank = pl.lit(None).cast(pl.Int64)
        
        if 'rarity_score' in df.columns:
            rarity_score = pl.col('rarity_score').cast(pl.Float64, strict=False)
        else:
            rarity_score = pl.lit(None).cast(pl.Float64)
        
        # Select columns in the minimal schema order
        df = df.select([
            pl.col('event_id'),
            pl.col('chain'),
            pl.col('collection'),
            pl.col('token_id'),
            pl.col('event_type'),
            pl.col('event_date'),
            pl.col('event_timestamp'),
            pl.col('timestamp').alias('unix_timestamp'),
            pl.col('tx_hash'),
            pl.col('seller'),
            pl.col('buyer'),
            pl.col('from_address'),
            pl.col('to_address'),
            pl.col('quantity'),
            pl.col('price_total_eth'),
            pl.col('price_each_eth'),
            pl.col('currency_symbol'),
            pl.col('contract_address'),
            rarity_rank.alias('rarity_rank'),
            rarity_score.alias('rarity_score'),
        ])
        
        return df


def get_data_quality_metrics(df: pl.DataFrame) -> dict:
    """
    Calculate data quality metrics after cleaning.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {
        'total_rows': len(df),
        'total_collections': df['collection'].n_unique(),
        'total_tokens': df['token_id'].n_unique(),
        'date_range': {
            'min': df['event_date'].min(),
            'max': df['event_date'].max()
        },
        'event_types': df.group_by('event_type').count().sort('count', descending=True).to_dicts(),
        'collections': df.group_by('collection').count().sort('count', descending=True).to_dicts(),
        'null_prices': (df['price_total_eth'] == 0).sum(),
        'transactions_with_price': (df['price_total_eth'] > 0).sum(),
    }
    
    return metrics
