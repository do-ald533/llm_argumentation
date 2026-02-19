#!/usr/bin/env python3
"""
Split input CSV into batches for parallel processing.

Usage:
    python create_batches.py --input data/Input/texts_AAEC.csv --batch-size 10 --output-dir data/batches/
"""

import argparse
from pathlib import Path
import polars as pl
from typing import Optional


def create_batches(
    input_csv: str,
    batch_size: int = 10,
    output_dir: str = "data/batches",
    prefix: Optional[str] = None
) -> None:
    """Split input CSV into smaller batch files.
    
    Args:
        input_csv: Path to input CSV file
        batch_size: Number of rows per batch
        output_dir: Directory to save batch files
        prefix: Optional prefix for batch files (defaults to input filename)
    """
    df = pl.read_csv(input_csv)
    total_rows = len(df)
    
    if prefix is None:
        input_path = Path(input_csv)
        prefix = input_path.stem.replace("texts_", "")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    print(f"Splitting {total_rows} rows into {num_batches} batches of {batch_size} rows each")
    print(f"Output directory: {output_path.absolute()}")
    print()
    
    batch_info = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_num = i + 1
        
        batch_df = df[start_idx:end_idx]
        
        batch_filename = f"batch_{batch_num:03d}_{prefix}.csv"
        batch_path = output_path / batch_filename
        
        batch_df.write_csv(batch_path)
        
        batch_info.append({
            'batch': batch_num,
            'file': batch_filename,
            'rows': len(batch_df),
            'text_ids': batch_df['text_id'].to_list()
        })
        
        print(f"Created {batch_filename} ({len(batch_df)} rows)")
    
    index_path = output_path / f"batch_index_{prefix}.txt"
    with open(index_path, 'w') as f:
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of batches: {num_batches}\n")
        f.write(f"Source: {input_csv}\n")
        f.write("\n")
        for info in batch_info:
            f.write(f"Batch {info['batch']:03d}: {info['file']} ({info['rows']} rows)\n")
            f.write(f"  Text IDs: {', '.join(info['text_ids'])}\n")
    
    print()
    print(f"Batch index saved to: {index_path}")
    print(f"\nNext steps:")
    print(f"1. Process batches: bash process_batches.sh {output_dir}")
    print(f"2. Merge results: python merge_results.py --prefix {prefix}")


def main():
    parser = argparse.ArgumentParser(
        description="Split input CSV into batches for parallel processing"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of rows per batch (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/batches",
        help="Directory to save batch files (default: data/batches)"
    )
    parser.add_argument(
        "--prefix",
        help="Prefix for batch filenames (default: derived from input filename)"
    )
    
    args = parser.parse_args()
    
    create_batches(
        input_csv=args.input,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()
