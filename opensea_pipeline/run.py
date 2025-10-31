#!/usr/bin/env python3
"""
OpenSea NFT Transaction Data Pipeline
Main orchestration script for processing raw NFT transaction data.

Usage:
    python run.py                    # Process raw_data CSV files
    python run.py --use-duckdb       # Use DuckDB for large files
    python run.py --validate-only    # Only run validation, no processing
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'pipeline' / 'src'))

from io_utils import DataLoader, DataWriter, VersionedOutput, get_file_stats
from validate import DataValidator
from clean_events import EventCleaner, get_data_quality_metrics
from aggregate import EventAggregator


class OpenSeaPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: dict):
        self.config = config
        self.workspace_root = Path(config['workspace_root'])
        self.raw_data_path = self.workspace_root / config['raw_data_dir']
        self.clean_path = self.workspace_root / config['clean_dir']
        
        # Initialize components
        self.loader = DataLoader(self.raw_data_path)
        self.validator = DataValidator(verbose=True)
        self.cleaner = EventCleaner()
        self.aggregator = EventAggregator()
        self.versioned_output = VersionedOutput(self.clean_path)
        
        self.run_log = []
        
    def log(self, message: str):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.run_log.append(log_entry)
    
    def run(self, validate_only: bool = False, use_duckdb: bool = False):
        """
        Execute the full pipeline.
        
        Args:
            validate_only: If True, only validate data without processing
            use_duckdb: If True, use DuckDB for loading large files
        """
        self.log("="*60)
        self.log("OpenSea NFT Transaction Data Pipeline")
        self.log("="*60)
        
        try:
            # Step 1: Load raw data
            self.log("\n[1/6] Loading raw data...")
            raw_df = self.loader.load_raw_csvs(pattern="*.csv", use_duckdb=use_duckdb)
            self.log(f"Loaded {len(raw_df):,} raw events from {raw_df['collection'].n_unique()} collections")
            
            # Step 2: Validate raw data
            self.log("\n[2/6] Validating raw data...")
            validation_report = self.validator.generate_report(raw_df)
            self.log("Validation complete")
            
            if validate_only:
                self.log("\nValidation-only mode. Exiting.")
                print("\n" + validation_report)
                return
            
            # Step 3: Clean and transform events
            self.log("\n[3/6] Cleaning and transforming events...")
            clean_df = self.cleaner.clean_events(raw_df)
            self.log(f"Cleaned dataset: {len(clean_df):,} events")
            
            # Step 4: Create aggregations
            self.log("\n[4/6] Creating analytical aggregations...")
            
            daily_stats = self.aggregator.create_daily_collection_stats(clean_df)
            token_stats = self.aggregator.create_token_stats(clean_df)
            collection_dim = self.aggregator.create_collection_dimension(clean_df)
            collection_summary = self.aggregator.create_collection_summary(clean_df)
            
            # Optional: Create wallet dimension (can be large)
            if self.config.get('create_wallet_dim', False):
                wallet_dim = self.aggregator.create_wallet_dimension(clean_df, min_transactions=5)
            else:
                wallet_dim = None
                self.log("  Skipping wallet dimension (set create_wallet_dim=True to enable)")
            
            # Step 5: Write outputs
            self.log("\n[5/6] Writing outputs...")
            version_dir = self.versioned_output.create_version_dir()
            writer = DataWriter(version_dir)
            
            # Write main analytical datasets
            writer.write_parquet(clean_df, "minimal_events.parquet")
            writer.write_parquet(daily_stats, "daily_collection_stats.parquet")
            writer.write_parquet(token_stats, "token_stats.parquet")
            writer.write_parquet(collection_dim, "collection_dimension.parquet")
            writer.write_parquet(collection_summary, "collection_summary.parquet")
            
            if wallet_dim is not None:
                writer.write_parquet(wallet_dim, "wallet_dimension.parquet")
            
            # Step 6: Generate metrics and write log
            self.log("\n[6/6] Generating data quality metrics...")
            metrics = get_data_quality_metrics(clean_df)
            
            # Write metrics as JSON
            metrics_path = version_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                # Convert dates to strings for JSON serialization
                metrics_serializable = {
                    k: str(v) if hasattr(v, 'isoformat') else v 
                    for k, v in metrics.items()
                }
                json.dump(metrics_serializable, f, indent=2, default=str)
            self.log(f"Wrote metrics to {metrics_path}")
            
            # Write run log
            log_content = "\n".join(self.run_log)
            log_content += "\n\n" + "="*60 + "\n"
            log_content += "VALIDATION REPORT\n"
            log_content += "="*60 + "\n"
            log_content += validation_report
            log_content += "\n\n" + "="*60 + "\n"
            log_content += "DATA QUALITY METRICS\n"
            log_content += "="*60 + "\n"
            log_content += json.dumps(metrics_serializable, indent=2, default=str)
            
            self.versioned_output.write_run_log(version_dir, log_content)
            
            # Summary
            self.log("\n" + "="*60)
            self.log("Pipeline completed successfully!")
            self.log("="*60)
            self.log(f"\nOutput directory: {version_dir}")
            self.log(f"Total events processed: {len(clean_df):,}")
            self.log(f"Collections: {metrics['total_collections']}")
            self.log(f"Unique tokens: {metrics['total_tokens']:,}")
            self.log(f"Date range: {metrics['date_range']['min']} to {metrics['date_range']['max']}")
            self.log(f"\nGenerated datasets:")
            self.log(f"  - minimal_events.parquet ({len(clean_df):,} rows)")
            self.log(f"  - daily_collection_stats.parquet ({len(daily_stats):,} rows)")
            self.log(f"  - token_stats.parquet ({len(token_stats):,} rows)")
            self.log(f"  - collection_dimension.parquet ({len(collection_dim):,} rows)")
            self.log(f"  - collection_summary.parquet ({len(collection_summary):,} rows)")
            if wallet_dim is not None:
                self.log(f"  - wallet_dimension.parquet ({len(wallet_dim):,} rows)")
            
        except Exception as e:
            self.log(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            self.log(traceback.format_exc())
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenSea NFT Transaction Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Process raw_data CSV files
  python run.py --use-duckdb       # Use DuckDB for large files
  python run.py --validate-only    # Only run validation
  python run.py --create-wallet-dim # Include wallet dimension table
        """
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate data without processing'
    )
    
    parser.add_argument(
        '--use-duckdb',
        action='store_true',
        help='Use DuckDB for loading (better for large files)'
    )
    
    parser.add_argument(
        '--create-wallet-dim',
        action='store_true',
        help='Create wallet dimension table (can be large)'
    )
    
    parser.add_argument(
        '--workspace',
        type=str,
        default='/Users/qingshen/Desktop/opensea',
        help='Path to workspace root directory'
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'workspace_root': args.workspace,
        'raw_data_dir': 'raw_data',
        'clean_dir': 'opensea_pipeline/clean',
        'create_wallet_dim': args.create_wallet_dim,
    }
    
    # Run pipeline
    pipeline = OpenSeaPipeline(config)
    pipeline.run(validate_only=args.validate_only, use_duckdb=args.use_duckdb)


if __name__ == '__main__':
    main()
