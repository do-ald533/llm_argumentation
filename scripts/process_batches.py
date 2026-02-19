#!/usr/bin/env python3
"""
Process all batch files in parallel or sequentially.

Usage:
    python scripts/process_batches.py data/batches --mode sequential
    python scripts/process_batches.py data/batches --mode parallel
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


class BatchProcessor:
    def __init__(self, batch_dir: str, output_dir: str = "output/batches"):
        self.batch_dir = Path(batch_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processes: List[subprocess.Popen] = []
        self.interrupted = False
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and termination signals."""
        if not self.interrupted:
            self.interrupted = True
            print("\n")
            print("Caught interrupt signal! Stopping all processes...")
            self._cleanup_processes()
            sys.exit(130)
    
    def _cleanup_processes(self):
        """Terminate all running processes."""
        for process in self.processes:
            if process.poll() is None:  # Process is still running
                print(f"  Terminating process {process.pid}...")
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        
        # Wait for graceful termination
        time.sleep(2)
        
        # Force kill if needed
        for process in self.processes:
            if process.poll() is None:
                print(f"  Force killing process {process.pid}...")
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
    
    def get_batch_files(self) -> List[Path]:
        """Get all batch CSV files."""
        batch_files = sorted(self.batch_dir.glob("batch_*.csv"))
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {self.batch_dir}")
        return batch_files
    
    def process_single_batch(self, batch_file: Path) -> tuple[str, bool]:
        """Process a single batch file.
        
        Returns:
            Tuple of (batch_name, success)
        """
        batch_name = batch_file.stem
        output_prefix = self.output_dir / batch_name
        log_file = self.output_dir / f"{batch_name}.log"
        
        cmd = [
            sys.executable, "-m", "src.main",
            "--input", str(batch_file),
            "--output-prefix", str(output_prefix),
            "--no-mlflow"
        ]
        
        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                self.processes.append(process)
                process.wait()
                
                if process.returncode == 0:
                    return batch_name, True
                else:
                    return batch_name, False
        except Exception as e:
            print(f"Error processing {batch_name}: {e}")
            return batch_name, False
    
    def process_sequential(self):
        """Process batches one at a time."""
        batch_files = self.get_batch_files()
        total = len(batch_files)
        
        print(f"Starting sequential processing of {total} batches...")
        print(f"Output directory: {self.output_dir.absolute()}")
        print()
        
        completed = 0
        failed = 0
        
        for i, batch_file in enumerate(batch_files, 1):
            if self.interrupted:
                break
            
            batch_name = batch_file.stem
            print("=" * 60)
            print(f"Processing batch {i}/{total}: {batch_name}")
            print("=" * 60)
            print()
            
            _, success = self.process_single_batch(batch_file)
            
            if success:
                print(f"\n✓ Completed: {batch_name}")
                completed += 1
            else:
                print(f"\n✗ Failed: {batch_name} (check log: {self.output_dir}/{batch_name}.log)")
                failed += 1
            
            print()
        
        print("=" * 60)
        print("Batch Processing Summary")
        print("=" * 60)
        print(f"Total batches: {total}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print()
        
        if failed == 0:
            print("All batches completed successfully!")
            print()
            print("Next step: Merge results")
            print(f"  python scripts/merge_results.py --input-dir {self.output_dir} --output-dir output")
        else:
            print("Warning: Some batches failed.")
            print("You can re-run failed batches individually or proceed with merging available results.")
    
    def process_parallel(self, max_workers: Optional[int] = None):
        """Process batches in parallel.
        
        Args:
            max_workers: Maximum number of parallel processes (None = CPU count)
        """
        batch_files = self.get_batch_files()
        total = len(batch_files)
        
        print(f"Starting parallel processing of {total} batches...")
        print(f"Max parallel workers: {max_workers or 'CPU count'}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("Press Ctrl+C to cancel all processes.")
        print()
        
        completed = 0
        failed = 0
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_batch = {
                    executor.submit(self.process_single_batch, batch_file): batch_file.stem
                    for batch_file in batch_files
                }
                
                # Process results as they complete
                for future in as_completed(future_to_batch):
                    if self.interrupted:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    batch_name = future_to_batch[future]
                    try:
                        _, success = future.result()
                        if success:
                            print(f"✓ Completed: {batch_name}")
                            completed += 1
                        else:
                            print(f"✗ Failed: {batch_name} (check log: {self.output_dir}/{batch_name}.log)")
                            failed += 1
                    except Exception as e:
                        print(f"✗ Exception in {batch_name}: {e}")
                        failed += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.interrupted = True
        
        print()
        if failed == 0:
            print("All batches completed successfully!")
        else:
            print(f"Warning: {failed} batch(es) failed. Check logs in {self.output_dir}/")
        
        print()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Process batch files in parallel or sequentially"
    )
    parser.add_argument(
        "batch_dir",
        help="Directory containing batch CSV files"
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel"],
        default="sequential",
        help="Processing mode (default: sequential)"
    )
    parser.add_argument(
        "--output-dir",
        default="output/batches",
        help="Directory for output files (default: output/batches)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel workers (parallel mode only)"
    )
    
    args = parser.parse_args()
    
    try:
        processor = BatchProcessor(args.batch_dir, args.output_dir)
        
        if args.mode == "sequential":
            processor.process_sequential()
        else:
            processor.process_parallel(args.max_workers)
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
