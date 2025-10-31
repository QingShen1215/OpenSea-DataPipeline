# OpenSea NFT Transaction Data Pipeline

An extensible ETL pipeline for processing OpenSea NFT transaction data, transforming raw CSV exports into clean, analysis-ready Parquet datasets.

## Project Structure

opensea_pipeline/
├── raw_data                       source/date
├── clean/                        # Versioned output directories
│   └── 2025-10-31_HH-MM-SS/
│       ├── minimal_events.parquet
│       ├── daily_collection_stats.parquet
│       ├── token_stats.parquet
│       ├── collection_dimension.parquet
│       ├── collection_summary.parquet
│       ├── wallet_dimension.parquet (optional)
│       ├── metrics.json
│       └── _run.log
├── pipeline/
│   └── src/
│       ├── schemas.py            # Data schema definitions
│       ├── io_utils.py           # I/O abstractions (Polars/DuckDB)
│       ├── validate.py           # Data quality validation
│       ├── clean_events.py       # Event cleaning & transformation
│       └── aggregate.py          # Analytics aggregation
├── run.py                        # Main orchestration script
├── requirements.txt
├── DESIGN.md                     # Design decisions & roadmap
└── README.md


