"""Main entry point for the argumentation structuring pipeline."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.pipeline import ArgumentationPipeline
from src.logging_config import setup_logging, get_logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM-based Argumentation Structuring Pipeline with MLflow Tracking"
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
    parser.add_argument(
        "--golden-components",
        type=Path,
        default=None,
        help="Path to golden standard components CSV for evaluation"
    )
    parser.add_argument(
        "--golden-relations",
        type=Path,
        default=None,
        help="Path to golden standard relations CSV for evaluation"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for MLflow run (auto-generated if not provided)"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking"
    )
    parser.add_argument(
        "--graph-image",
        action="store_true",
        help="Save argumentation graph images (PNG) for each processed text"
    )
    parser.add_argument(
        "--skip-identification",
        type=Path,
        default=None,
        metavar="COMPONENTS_CSV",
        help="Path to a components CSV (same format as golden standard) to use instead of "
             "LLM-based component identification. The pipeline continues normally from "
             "conclusion extraction onward."
    )
    parser.add_argument(
        "--partial-attack",
        action="store_true",
        help="Enable partial-attack relation detection. Only use with datasets that annotate "
             "partial-attack (e.g. AbstRCT). When set, the LLM is asked to distinguish between "
             "full attacks (direct contradiction) and partial-attacks (constraining/weakening)."
    )
    parser.add_argument(
        "--no-merge-cycle",
        dest="merge_cycle",
        action="store_false",
        default=True,
        help="Disable the cycle-merge step in UnvisitedPremisesTask. Cyclic links are kept "
             "as-is instead of being collapsed into a single merged component."
    )

    args = parser.parse_args()
    
    logger = setup_logging()
    
    if not args.input.exists():
        logger.error("input_file_not_found", path=str(args.input))
        return
    
    if args.no_mlflow:
        config.enable_mlflow = False
        logger.info("mlflow_disabled_via_cli")
    
    logger.info("pipeline_starting", 
                model=config.openai_model,
                temperature=config.temperature,
                prompt_version=config.prompt_version,
                input_file=str(args.input),
                output_dir=str(config.output_data_dir),
                mlflow_enabled=config.enable_mlflow)
    
    pipeline = ArgumentationPipeline(config)
    
    golden_components = args.golden_components
    golden_relations = args.golden_relations
    
    if not golden_components and not golden_relations:
        possible_components = config.golden_standard_dir / f"components_{args.output_prefix}.csv"
        possible_relations = config.golden_standard_dir / f"relations_{args.output_prefix}.csv"
        
        if possible_components.exists() and possible_relations.exists():
            golden_components = possible_components
            golden_relations = possible_relations
            logger.info("golden_standard_auto_detected",
                       components=str(golden_components),
                       relations=str(golden_relations))
    
    try:
        graphs = pipeline.process_csv(
            input_file=args.input,
            output_prefix=args.output_prefix,
            limit=args.limit,
            golden_components_path=golden_components,
            golden_relations_path=golden_relations,
            run_name=args.run_name,
            graph_image=args.graph_image,
            skip_identification_path=args.skip_identification,
            enable_partial_attack=args.partial_attack,
            enable_merge_cycle=args.merge_cycle,
        )
        
        logger.info("processing_complete", 
                   total=len(graphs),
                   output_prefix=args.output_prefix)
        
        print(f"\n{'='*60}")
        print(f"Processing Complete!")
        print(f"{'='*60}")
        print(f"Texts processed: {len(graphs)}")
        print(f"Output directory: {config.output_data_dir}")
        print(f"Output files: components_{args.output_prefix}.csv, relations_{args.output_prefix}.csv")
        
        if config.enable_mlflow:
            print(f"\nMLflow tracking URI: {config.mlflow_tracking_uri}")
            print(f"View results: mlflow ui --port 5000")
        
        print(f"{'='*60}\n")
    
    except KeyboardInterrupt:
        logger.warning("pipeline_interrupted_by_user")
    except Exception as e:
        logger.error("pipeline_failed", error=str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
