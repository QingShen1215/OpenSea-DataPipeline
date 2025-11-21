#!/bin/bash

# Quick setup script for OpenSea NFT Pipeline
# Usage: bash setup.sh

set -e  # Exit on error

echo "========================================"
echo "OpenSea NFT Pipeline - Quick Setup"
echo "========================================"
echo ""

# Check Python version
echo "[1/3] Checking Python version..."
python3 --version || { echo "Error: Python 3 not found. Please install Python 3.9+"; exit 1; }

# Install dependencies
echo ""
echo "[2/3] Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "[3/3] Verifying installation..."
python3 -c "import polars; import duckdb; print('✓ Polars version:', polars.__version__); print('✓ DuckDB version:', duckdb.__version__)"

echo ""
echo "========================================"
echo "✓ Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Ensure your CSV files are in ../raw_data/"
echo "  2. Run the pipeline: python run.py"
echo "  3. For large files: python run.py --use-duckdb"
echo "  4. View results in clean/YYYY-MM-DD_HH-MM-SS/"
echo ""
echo "For more options: python run.py --help"
echo ""
