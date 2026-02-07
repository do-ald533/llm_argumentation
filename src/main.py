"""Main entry point for the argumentation structuring pipeline."""
import sys
import argparse
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.pipeline import ArgumentationPipeline
from src.logging_config import setup_logging


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM-based Argumentation Structuring Pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file with text_id and text_tokens columns"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="output",
        help="Prefix for output files (default: output)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of texts to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Validate input file
    if not args.input.exists():
        logger.error("input_file_not_found", path=str(args.input))
        return
    
    # Initialize pipeline
    logger.info("pipeline_starting", 
                model=config.openai_model,
                input_file=str(args.input),
                output_dir=str(config.output_data_dir))
    
    pipeline = ArgumentationPipeline(config)
    
    # Process texts
    try:
        graphs = pipeline.process_csv(args.input, args.output_prefix, limit=args.limit)
        
        if args.limit and len(graphs) >= args.limit:
            logger.info("processing_limit_reached", limit=args.limit, processed=len(graphs))
    
    except KeyboardInterrupt:
        logger.warning("pipeline_interrupted_by_user")
    except Exception as e:
        logger.error("pipeline_failed", error=str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
