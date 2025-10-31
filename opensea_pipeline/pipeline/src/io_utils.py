"""
I/O utilities for reading and writing data.
Supports both Polars (for in-memory processing) and DuckDB (for larger datasets).
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Dict, Any
import polars as pl
import duckdb
from datetime import datetime


class DataLoader:
    """Handles loading data from various sources."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def load_raw_csvs(self, pattern: str = "*.csv", use_duckdb: bool = False) -> pl.DataFrame:
        """
        Load CSV files matching pattern.
        
        Args:
            pattern: Glob pattern for CSV files
            use_duckdb: If True, use DuckDB for loading (better for large files)
            
        Returns:
            Polars DataFrame with combined data
        """
        csv_files = list(self.base_path.glob(pattern))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found matching {pattern} in {self.base_path}")
        
        print(f"Found {len(csv_files)} CSV files to load")
        
        if use_duckdb:
            return self._load_with_duckdb(csv_files)
        else:
            return self._load_with_polars(csv_files)
    
    def _load_with_polars(self, csv_files: List[Path]) -> pl.DataFrame:
        """Load CSVs using Polars (good for medium-sized files)."""
        dfs = []
        
        for csv_file in csv_files:
            print(f"Loading {csv_file.name}...")
            try:
                df = pl.read_csv(
                    csv_file,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                    infer_schema_length=10000
                )
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No data loaded successfully")
        
        # Combine all dataframes
        combined_df = pl.concat(dfs, how="diagonal")
        print(f"Loaded {len(combined_df)} total rows")
        
        return combined_df
    
    def _load_with_duckdb(self, csv_files: List[Path]) -> pl.DataFrame:
        """Load CSVs using DuckDB (better for large files)."""
        conn = duckdb.connect(':memory:')
        
        # Create a view combining all CSV files
        csv_paths = [str(f) for f in csv_files]
        query = f"""
        SELECT * FROM read_csv_auto(
            {csv_paths},
            ignore_errors=true,
            union_by_name=true
        )
        """
        
        result = conn.execute(query).pl()
        conn.close()
        
        print(f"Loaded {len(result)} total rows using DuckDB")
        return result
    
    def load_parquet(self, file_path: str) -> pl.DataFrame:
        """Load a single Parquet file."""
        return pl.read_parquet(file_path)
    
    def load_parquet_partitioned(self, directory: str, filters: Optional[Dict[str, Any]] = None) -> pl.DataFrame:
        """
        Load partitioned Parquet files.
        
        Args:
            directory: Directory containing partitioned parquet files
            filters: Optional filters like {'collection': 'azuki', 'date': '2024-01'}
        """
        parquet_files = glob.glob(os.path.join(directory, "**/*.parquet"), recursive=True)
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {directory}")
        
        # Apply filters if provided
        if filters:
            filtered_files = []
            for file in parquet_files:
                match = all(f"{k}={v}" in file for k, v in filters.items())
                if match:
                    filtered_files.append(file)
            parquet_files = filtered_files
        
        if not parquet_files:
            return pl.DataFrame()
        
        return pl.read_parquet(parquet_files)


class DataWriter:
    """Handles writing data to various formats."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write_parquet(self, df: pl.DataFrame, filename: str, compression: str = "zstd") -> str:
        """
        Write DataFrame to Parquet file.
        
        Args:
            df: Polars DataFrame to write
            filename: Output filename (without path)
            compression: Compression algorithm (zstd, snappy, gzip, lz4)
            
        Returns:
            Full path to written file
        """
        output_path = self.base_path / filename
        df.write_parquet(output_path, compression=compression)
        print(f"Wrote {len(df)} rows to {output_path}")
        return str(output_path)
    
    def write_parquet_partitioned(
        self, 
        df: pl.DataFrame, 
        partition_cols: List[str],
        base_name: str = "data"
    ) -> str:
        """
        Write DataFrame to partitioned Parquet files.
        
        Args:
            df: Polars DataFrame to write
            partition_cols: Columns to partition by (e.g., ['collection', 'date'])
            base_name: Base name for the partition directory
            
        Returns:
            Path to partition directory
        """
        partition_dir = self.base_path / base_name
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # Use DuckDB for efficient partitioned writing
        conn = duckdb.connect(':memory:')
        conn.execute("CREATE TABLE temp_table AS SELECT * FROM df")
        
        partition_str = ', '.join(partition_cols)
        query = f"""
        COPY (SELECT * FROM temp_table) 
        TO '{partition_dir}' 
        (FORMAT PARQUET, PARTITION_BY ({partition_str}))
        """
        
        conn.execute(query)
        conn.close()
        
        print(f"Wrote partitioned data to {partition_dir}")
        return str(partition_dir)
    
    def write_csv(self, df: pl.DataFrame, filename: str) -> str:
        """Write DataFrame to CSV file."""
        output_path = self.base_path / filename
        df.write_csv(output_path)
        print(f"Wrote {len(df)} rows to {output_path}")
        return str(output_path)


class VersionedOutput:
    """Manages versioned output directories."""
    
    def __init__(self, base_clean_path: str):
        self.base_clean_path = Path(base_clean_path)
        self.base_clean_path.mkdir(parents=True, exist_ok=True)
        
    def create_version_dir(self, prefix: str = "") -> Path:
        """
        Create a new timestamped version directory.
        
        Args:
            prefix: Optional prefix for the version directory
            
        Returns:
            Path to the created version directory
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        version_name = f"{prefix}{timestamp}" if prefix else timestamp
        version_dir = self.base_clean_path / version_name
        version_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Created version directory: {version_dir}")
        return version_dir
    
    def get_latest_version(self) -> Optional[Path]:
        """Get the most recent version directory."""
        version_dirs = sorted(self.base_clean_path.glob("*"))
        
        if not version_dirs:
            return None
        
        return version_dirs[-1]
    
    def write_run_log(self, version_dir: Path, log_content: str):
        """Write run log to version directory."""
        log_path = version_dir / "_run.log"
        
        with open(log_path, 'w') as f:
            f.write(f"Run timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n")
            f.write(log_content)
        
        print(f"Wrote run log to {log_path}")


def get_file_stats(file_path: str) -> Dict[str, Any]:
    """Get statistics about a file."""
    path = Path(file_path)
    
    if not path.exists():
        return {"exists": False}
    
    size_mb = path.stat().st_size / (1024 * 1024)
    
    return {
        "exists": True,
        "size_mb": round(size_mb, 2),
        "modified": datetime.fromtimestamp(path.stat().st_mtime)
    }
