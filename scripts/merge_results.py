#!/usr/bin/env python3
"""
Merge batch processing results into final output files.

Usage:
    python merge_results.py --input-dir output/batches --output-dir output --prefix AAEC
"""

import argparse
from pathlib import Path
import polars as pl
from typing import List, Tuple


def find_batch_files(input_dir: Path, file_pattern: str) -> List[Path]:
    """Find all batch files matching the pattern."""
    files = sorted(input_dir.glob(file_pattern))
    return files


def merge_csv_files(files: List[Path], output_path: Path) -> Tuple[int, List[str]]:
    """Merge multiple CSV files into one.
    
    Returns:
        Tuple of (total_rows, list of source files)
    """
    if not files:
        return 0, []
    
    print(f"Merging {len(files)} files into {output_path.name}")
    
    # Read and concatenate all files
    dfs = []
    source_files = []
    
    for file_path in files:
        try:
            df = pl.read_csv(file_path, separator=';')
            dfs.append(df)
            source_files.append(file_path.name)
            print(f"  ✓ {file_path.name} ({len(df)} rows)")
        except Exception as e:
            print(f"  ✗ {file_path.name} - Error: {e}")
    
    if not dfs:
        print(f"  No valid files to merge!")
        return 0, []
    
    # Concatenate all dataframes
    merged_df = pl.concat(dfs)
    
    # Save merged file with semicolon separator
    merged_df.write_csv(output_path, separator=';')
    
    print(f"  → Saved {len(merged_df)} total rows to {output_path}")
    print()
    
    return len(merged_df), source_files


def create_merge_report(
    output_dir: Path,
    prefix: str,
    component_stats: Tuple[int, List[str]],
    relation_stats: Tuple[int, List[str]]
) -> None:
    """Create a merge report file."""
    report_path = output_dir / f"merge_report_{prefix}.txt"
    
    component_count, component_files = component_stats
    relation_count, relation_files = relation_stats
    
    with open(report_path, 'w') as f:
        f.write("Batch Merge Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {prefix}\n\n")
        
        f.write("Components\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total rows: {component_count}\n")
        f.write(f"Source files ({len(component_files)}):\n")
        for file in component_files:
            f.write(f"  - {file}\n")
        f.write("\n")
        
        f.write("Relations\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total rows: {relation_count}\n")
        f.write(f"Source files ({len(relation_files)}):\n")
        for file in relation_files:
            f.write(f"  - {file}\n")
        f.write("\n")
    
    print(f"Merge report saved to: {report_path}")


def merge_results(
    input_dir: str = "output/batches",
    output_dir: str = "output",
    prefix: str = "AAEC"
) -> None:
    """Merge batch processing results.
    
    Args:
        input_dir: Directory containing batch output files
        output_dir: Directory to save merged files
        prefix: Dataset prefix for output filenames
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Merging Batch Results")
    print("=" * 60)
    print(f"Input directory: {input_path.absolute()}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Dataset prefix: {prefix}")
    print()
    
    component_files = find_batch_files(input_path, "components_batch_*.csv")
    relation_files = find_batch_files(input_path, "relations_batch_*.csv")
    
    if not component_files and not relation_files:
        print("Error: No batch output files found!")
        print(f"Looking for: components_batch_*.csv and relations_batch_*.csv")
        print(f"In directory: {input_path}")
        return
    
    component_output = output_path / f"components_{prefix}.csv"
    component_stats = merge_csv_files(component_files, component_output)
    
    relation_output = output_path / f"relations_{prefix}.csv"
    relation_stats = merge_csv_files(relation_files, relation_output)
    
    create_merge_report(output_path, prefix, component_stats, relation_stats)
    
    print()
    print("=" * 60)
    print("Merge Complete!")
    print("=" * 60)
    print(f"Components: {component_stats[0]} rows → {component_output}")
    print(f"Relations: {relation_stats[0]} rows → {relation_output}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Merge batch processing results into final output files"
    )
    parser.add_argument(
        "--input-dir",
        default="output/batches",
        help="Directory containing batch output files (default: output/batches)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save merged files (default: output)"
    )
    parser.add_argument(
        "--prefix",
        default="AAEC",
        help="Dataset prefix for output filenames (default: AAEC)"
    )
    
    args = parser.parse_args()
    
    merge_results(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()
