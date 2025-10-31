"""
Validation utilities for data quality checks.
Validates schemas, detects duplicates, and identifies data issues.
"""

from typing import List, Dict, Tuple, Any
import polars as pl
from schemas import EventSchema, VALID_EVENT_TYPES, NULL_ADDRESS


class DataValidator:
    """Validates data quality and schema compliance."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.validation_results = []
        
    def validate_schema(self, df: pl.DataFrame, expected_cols: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if DataFrame has expected columns.
        
        Args:
            df: DataFrame to validate
            expected_cols: List of expected column names
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        actual_cols = set(df.columns)
        expected_cols_set = set(expected_cols)
        missing = expected_cols_set - actual_cols
        
        if missing:
            self._log(f"❌ Missing columns: {missing}")
            return False, list(missing)
        
        self._log(f"✓ All required columns present")
        return True, []
    
    def check_nulls(self, df: pl.DataFrame, critical_cols: List[str]) -> Dict[str, int]:
        """
        Check for null values in critical columns.
        
        Args:
            df: DataFrame to check
            critical_cols: Columns that should not have nulls
            
        Returns:
            Dictionary mapping column names to null counts
        """
        null_counts = {}
        
        for col in critical_cols:
            if col not in df.columns:
                continue
                
            null_count = df[col].null_count()
            if null_count > 0:
                null_counts[col] = null_count
                self._log(f"⚠️  Column '{col}' has {null_count} null values")
        
        if not null_counts:
            self._log(f"✓ No nulls in critical columns")
        
        return null_counts
    
    def detect_duplicates(
        self, 
        df: pl.DataFrame, 
        key_cols: List[str]
    ) -> Tuple[int, pl.DataFrame]:
        """
        Detect duplicate rows based on key columns.
        
        Args:
            df: DataFrame to check
            key_cols: Columns that define uniqueness
            
        Returns:
            Tuple of (duplicate_count, duplicate_rows)
        """
        duplicate_mask = df.select(key_cols).is_duplicated()
        duplicate_count = duplicate_mask.sum()
        duplicates = df.filter(duplicate_mask)
        
        if duplicate_count > 0:
            self._log(f"⚠️  Found {duplicate_count} duplicate rows based on {key_cols}")
        else:
            self._log(f"✓ No duplicates found")
        
        return duplicate_count, duplicates
    
    def validate_event_types(self, df: pl.DataFrame, event_col: str = 'event_type') -> Dict[str, int]:
        """
        Check for invalid event types.
        
        Args:
            df: DataFrame to validate
            event_col: Name of event type column
            
        Returns:
            Dictionary of invalid event types and their counts
        """
        if event_col not in df.columns:
            self._log(f"❌ Column '{event_col}' not found")
            return {}
        
        event_counts = df.group_by(event_col).count().sort('count', descending=True)
        
        invalid_events = {}
        for row in event_counts.iter_rows(named=True):
            event_type = row[event_col]
            count = row['count']
            
            if event_type not in VALID_EVENT_TYPES:
                invalid_events[event_type] = count
                self._log(f"⚠️  Invalid event type '{event_type}': {count} occurrences")
        
        if not invalid_events:
            self._log(f"✓ All event types are valid")
        
        return invalid_events
    
    def validate_addresses(self, df: pl.DataFrame, address_cols: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Validate Ethereum address formats.
        
        Args:
            df: DataFrame to validate
            address_cols: List of columns containing addresses
            
        Returns:
            Dictionary with validation results per column
        """
        results = {}
        
        for col in address_cols:
            if col not in df.columns:
                continue
            
            # Check for valid address format (0x + 40 hex chars or empty/null)
            invalid_addresses = df.filter(
                (pl.col(col).is_not_null()) & 
                (pl.col(col) != "") &
                (~pl.col(col).str.contains(r"^0x[a-fA-F0-9]{40}$"))
            )
            
            invalid_count = len(invalid_addresses)
            null_count = df[col].null_count()
            empty_count = (df[col] == "").sum()
            
            results[col] = {
                'invalid_format': invalid_count,
                'null': null_count,
                'empty': empty_count
            }
            
            if invalid_count > 0:
                self._log(f"⚠️  Column '{col}' has {invalid_count} invalid address formats")
        
        return results
    
    def validate_prices(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Validate price-related fields.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with price validation results
        """
        results = {}
        
        if 'price_total' in df.columns:
            # Check for negative prices
            negative_prices = df.filter(pl.col('price_total') < 0)
            results['negative_price_total'] = len(negative_prices)
            
            if results['negative_price_total'] > 0:
                self._log(f"⚠️  Found {results['negative_price_total']} rows with negative price_total")
        
        if 'price_each' in df.columns:
            # Check consistency: price_each should be price_total / quantity
            inconsistent = df.filter(
                (pl.col('price_each').is_not_null()) &
                (pl.col('price_total').is_not_null()) &
                (pl.col('quantity') > 0) &
                (pl.col('price_each') != pl.col('price_total') / pl.col('quantity'))
            )
            results['inconsistent_price_each'] = len(inconsistent)
            
            if results['inconsistent_price_each'] > 0:
                self._log(f"⚠️  Found {results['inconsistent_price_each']} rows with inconsistent price_each")
        
        return results
    
    def validate_timestamps(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Validate timestamp fields.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with timestamp validation results
        """
        results = {}
        
        if 'timestamp' in df.columns:
            # Check for reasonable timestamp range (after 2015, before now)
            min_ts = 1420070400  # 2015-01-01
            max_ts = 2000000000  # ~2033-05-18
            
            invalid_timestamps = df.filter(
                (pl.col('timestamp') < min_ts) | 
                (pl.col('timestamp') > max_ts)
            )
            results['invalid_timestamps'] = len(invalid_timestamps)
            
            if results['invalid_timestamps'] > 0:
                self._log(f"⚠️  Found {results['invalid_timestamps']} rows with invalid timestamps")
        
        if 'time_utc' in df.columns:
            # Check for parseable datetime strings
            try:
                parsed = df.select(pl.col('time_utc').str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S%z', strict=False))
                null_count = parsed.null_count().sum_horizontal()[0]
                results['unparseable_time_utc'] = null_count
                
                if null_count > 0:
                    self._log(f"⚠️  Found {null_count} rows with unparseable time_utc")
            except Exception as e:
                self._log(f"⚠️  Error parsing time_utc: {e}")
                results['time_utc_parse_error'] = str(e)
        
        return results
    
    def generate_report(self, df: pl.DataFrame) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            String containing validation report
        """
        report_lines = [
            "="*60,
            "DATA VALIDATION REPORT",
            "="*60,
            f"Total rows: {len(df):,}",
            f"Total columns: {len(df.columns)}",
            ""
        ]
        
        # Schema validation
        schema = EventSchema()
        is_valid, missing = self.validate_schema(df, schema.EXPECTED_COLUMNS)
        
        # Null checks
        critical_cols = ['collection', 'token_id', 'event_type', 'timestamp', 'tx']
        null_counts = self.check_nulls(df, critical_cols)
        
        # Duplicate detection
        dup_count, _ = self.detect_duplicates(df, ['tx', 'token_id', 'event_type'])
        report_lines.append(f"Duplicate events: {dup_count:,}")
        
        # Event type validation
        invalid_events = self.validate_event_types(df)
        
        # Price validation
        price_results = self.validate_prices(df)
        
        # Timestamp validation
        timestamp_results = self.validate_timestamps(df)
        
        report_lines.append("")
        report_lines.append("="*60)
        
        return "\n".join(report_lines)
    
    def _log(self, message: str):
        """Log a validation message."""
        if self.verbose:
            print(message)
        self.validation_results.append(message)
