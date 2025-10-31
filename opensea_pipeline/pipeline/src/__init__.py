"""
OpenSea NFT Transaction Data Pipeline
A modular ETL system for processing NFT transaction data.
"""

__version__ = '1.0.0'
__author__ = 'OpenSea Analytics Team'

from .schemas import EventSchema, AggregateSchema, DimensionSchema
from .io_utils import DataLoader, DataWriter, VersionedOutput
from .validate import DataValidator
from .clean_events import EventCleaner, get_data_quality_metrics
from .aggregate import EventAggregator

__all__ = [
    'EventSchema',
    'AggregateSchema',
    'DimensionSchema',
    'DataLoader',
    'DataWriter',
    'VersionedOutput',
    'DataValidator',
    'EventCleaner',
    'EventAggregator',
    'get_data_quality_metrics',
]
