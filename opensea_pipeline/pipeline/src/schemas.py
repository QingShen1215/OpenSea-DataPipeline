"""
Schema definitions for OpenSea NFT transaction data pipeline.
Defines expected columns, types, and validation rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import polars as pl


@dataclass
class EventSchema:
    """Schema definition for raw NFT transaction events."""
    
    # Core event fields
    EXPECTED_COLUMNS: List[str] = field(default_factory=lambda: [
        'chain',
        'collection',
        'identifier',
        'event_type',
        'time_utc',
        'timestamp',
        'tx',
        'seller',
        'buyer',
        'from_address',
        'to_address',
        'quantity',
        'price_total',
        'currency_symbol',
        'contract',
        'token_id',
        'price_each'
    ])
    
    # Optional NFT metadata fields
    OPTIONAL_META_COLUMNS: List[str] = field(default_factory=lambda: [
        'nft_description',
        'nft_image_url',
        'nft_display_image_url',
        'nft_animation_url',
        'metadata_url',
        'rarity_rank',
        'rarity_score'
    ])
    
    @staticmethod
    def get_polars_schema() -> Dict[str, pl.DataType]:
        """Return Polars schema with explicit data types."""
        return {
            'chain': pl.Utf8,
            'collection': pl.Utf8,
            'identifier': pl.Utf8,
            'event_type': pl.Utf8,
            'time_utc': pl.Utf8,  # Will parse to datetime later
            'timestamp': pl.Int64,
            'tx': pl.Utf8,
            'seller': pl.Utf8,
            'buyer': pl.Utf8,
            'from_address': pl.Utf8,
            'to_address': pl.Utf8,
            'quantity': pl.Int64,
            'price_total': pl.Float64,
            'currency_symbol': pl.Utf8,
            'contract': pl.Utf8,
            'token_id': pl.Utf8,
            'price_each': pl.Utf8,  # Can be empty, will convert to float
        }
    
    @staticmethod
    def get_minimal_event_schema() -> Dict[str, pl.DataType]:
        """Schema for minimal analytical event dataset."""
        return {
            'event_id': pl.Utf8,  # tx + token_id
            'chain': pl.Utf8,
            'collection': pl.Utf8,
            'token_id': pl.Utf8,
            'event_type': pl.Utf8,
            'event_date': pl.Date,
            'event_timestamp': pl.Datetime,
            'unix_timestamp': pl.Int64,
            'tx_hash': pl.Utf8,
            'seller': pl.Utf8,
            'buyer': pl.Utf8,
            'from_address': pl.Utf8,
            'to_address': pl.Utf8,
            'quantity': pl.Int64,
            'price_total_eth': pl.Float64,
            'price_each_eth': pl.Float64,
            'currency_symbol': pl.Utf8,
            'contract_address': pl.Utf8,
            # Optional metadata
            'rarity_rank': pl.Int64,
            'rarity_score': pl.Float64,
        }


@dataclass
class AggregateSchema:
    """Schema for aggregated analytical datasets."""
    
    @staticmethod
    def get_daily_collection_stats_schema() -> Dict[str, pl.DataType]:
        """Schema for daily collection-level statistics."""
        return {
            'collection': pl.Utf8,
            'event_date': pl.Date,
            'total_transactions': pl.Int64,
            'unique_buyers': pl.Int64,
            'unique_sellers': pl.Int64,
            'unique_tokens_traded': pl.Int64,
            'total_volume_eth': pl.Float64,
            'avg_price_eth': pl.Float64,
            'median_price_eth': pl.Float64,
            'min_price_eth': pl.Float64,
            'max_price_eth': pl.Float64,
            'mint_count': pl.Int64,
            'sale_count': pl.Int64,
            'transfer_count': pl.Int64,
        }
    
    @staticmethod
    def get_token_stats_schema() -> Dict[str, pl.DataType]:
        """Schema for individual token trading statistics."""
        return {
            'collection': pl.Utf8,
            'token_id': pl.Utf8,
            'contract_address': pl.Utf8,
            'first_mint_date': pl.Date,
            'last_trade_date': pl.Date,
            'total_trades': pl.Int64,
            'total_volume_eth': pl.Float64,
            'avg_price_eth': pl.Float64,
            'last_price_eth': pl.Float64,
            'unique_owners': pl.Int64,
            'rarity_rank': pl.Int64,
            'rarity_score': pl.Float64,
        }


@dataclass
class DimensionSchema:
    """Schema for dimension tables."""
    
    @staticmethod
    def get_collection_dim_schema() -> Dict[str, pl.DataType]:
        """Schema for collection dimension table."""
        return {
            'collection': pl.Utf8,
            'contract_address': pl.Utf8,
            'first_event_date': pl.Date,
            'last_event_date': pl.Date,
            'total_tokens': pl.Int64,
            'total_transactions': pl.Int64,
        }
    
    @staticmethod
    def get_wallet_dim_schema() -> Dict[str, pl.DataType]:
        """Schema for wallet dimension table."""
        return {
            'wallet_address': pl.Utf8,
            'first_activity_date': pl.Date,
            'last_activity_date': pl.Date,
            'total_purchases': pl.Int64,
            'total_sales': pl.Int64,
            'total_volume_bought_eth': pl.Float64,
            'total_volume_sold_eth': pl.Float64,
            'unique_collections': pl.Int64,
        }


# Event type constants
VALID_EVENT_TYPES = ['mint', 'sale', 'transfer', 'list', 'cancel_list', 'offer', 'cancel_offer']

# Known collections (from raw_data folder)
KNOWN_COLLECTIONS = ['azuki', 'bayc', 'clonex', 'coolcats', 'milady', 'pudgypenguins']

# Null address constant
NULL_ADDRESS = '0x0000000000000000000000000000000000000000'
