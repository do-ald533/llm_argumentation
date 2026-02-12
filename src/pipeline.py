"""Main pipeline for processing multiple texts."""
import polars as pl
from pathlib import Path
from typing import List, Optional
import time
import hashlib
from src.config import Config
from src.graph import ArgumentationWorkflow
from src.models import ArgumentGraph
from src.export import export_to_golden_standard
from src.evaluation import evaluate_against_golden_standard
from src.logging_config import get_logger

logger = get_logger("pipeline")

# MLflow imports (optional)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow_not_installed", message="MLflow not available, experiment tracking disabled")


def get_prompt_hash() -> str:
    """Generate hash of prompts file for versioning."""
    try:
        prompts_file = Path("src/llm/prompts.py")
        if prompts_file.exists():
            with open(prompts_file, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:8]
    except Exception:
        pass
    return "unknown"


class ArgumentationPipeline:
    """Main pipeline for processing texts through argumentation analysis."""
    
    def __init__(self, config: Config):
        """Initialize pipeline.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.workflow = ArgumentationWorkflow(config)
        
        # Setup MLflow if enabled
        if config.enable_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment_name)
            logger.info("mlflow_initialized", 
                       tracking_uri=config.mlflow_tracking_uri,
                       experiment=config.mlflow_experiment_name)
    
    def process_csv(
        self, 
        input_file: Path, 
        output_prefix: str,
        limit: Optional[int] = None,
        golden_components_path: Optional[Path] = None,
        golden_relations_path: Optional[Path] = None,
        run_name: Optional[str] = None,
        graph_image: bool = False
    ) -> List[ArgumentGraph]:
        """Process texts from CSV file with MLflow tracking.
        
        Args:
            input_file: Path to input CSV file with columns: text_id, text_tokens
            output_prefix: Prefix for output files
            limit: Optional limit on number of texts to process
            golden_components_path: Path to golden standard components for evaluation
            golden_relations_path: Path to golden standard relations for evaluation
            run_name: Optional custom name for MLflow run
            graph_image: If True, save graph images (PNG) for each text
            
        Returns:
            List of processed ArgumentGraph objects
        """
        start_time = time.time()
        
        # Load input data
        logger.info("loading_input_data", path=str(input_file))
        df = pl.read_csv(input_file)
        
        if "text_id" not in df.columns or "text_tokens" not in df.columns:
            raise ValueError("CSV must have 'text_id' and 'text_tokens' columns")
        
        # Apply limit if specified
        if limit:
            df = df.head(limit)
        
        total_texts = len(df)
        logger.info("texts_to_process", count=total_texts)
        
        # Start MLflow run if enabled
        mlflow_active = self.config.enable_mlflow and MLFLOW_AVAILABLE
        
        if mlflow_active:
            # Generate run name if not provided
            if run_name is None:
                run_name = f"{output_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
            
            mlflow.start_run(run_name=run_name)
            
            # Log parameters
            mlflow.log_params({
                "llm_model": self.config.openai_model,
                "temperature": self.config.temperature,
                "max_retries": self.config.max_retries,
                "dataset": output_prefix,
                "input_file": str(input_file),
                "total_texts": total_texts,
                "prompt_version": self.config.prompt_version,
                "prompt_hash": get_prompt_hash(),
            })
            
            # Log prompts file as artifact
            prompts_file = Path("src/llm/prompts.py")
            if prompts_file.exists():
                mlflow.log_artifact(str(prompts_file), artifact_path="prompts")
        
        # Process each text
        graphs = []
        failed_count = 0
        
        for idx, row in enumerate(df.iter_rows(named=True), 1):
            text_id = row["text_id"]
            text = row["text_tokens"]
            
            logger.info("processing_text", number=idx, total=total_texts, text_id=text_id)
            
            try:
                graph = self.workflow.run(text, text_id)
                graphs.append(graph)
                
                # Save graph image if requested
                if graph_image:
                    images_dir = self.config.output_data_dir / "graphs"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    image_path = images_dir / f"graph_{text_id}.png"
                    graph.visualize(image_path)
                    logger.info("graph_image_saved", text_id=text_id, path=str(image_path))
            except Exception as e:
                logger.error("text_processing_failed", text_id=text_id, error=str(e))
                failed_count += 1
                continue
        
        processing_time = time.time() - start_time
        
        # Export results
        logger.info("exporting_results", prefix=output_prefix)
        
        export_to_golden_standard(
            graphs,
            self.config.output_data_dir,
            prefix=output_prefix
        )
        
        # Paths to generated outputs
        components_path = self.config.output_data_dir / f"components_{output_prefix}.csv"
        relations_path = self.config.output_data_dir / f"relations_{output_prefix}.csv"
        
        # Evaluate if golden standard provided
        evaluation_metrics = None
        if golden_components_path and golden_relations_path:
            logger.info("evaluating_against_golden_standard")
            evaluation_metrics = evaluate_against_golden_standard(
                predicted_components_path=components_path,
                predicted_relations_path=relations_path,
                golden_components_path=golden_components_path,
                golden_relations_path=golden_relations_path
            )
            logger.info("evaluation_complete", metrics=str(evaluation_metrics))
        
        # Log metrics to MLflow
        if mlflow_active:
            # Processing metrics
            mlflow.log_metrics({
                "texts_processed": len(graphs),
                "texts_failed": failed_count,
                "processing_time_seconds": processing_time,
                "avg_time_per_text": processing_time / total_texts if total_texts > 0 else 0,
            })
            
            # Evaluation metrics (if available)
            if evaluation_metrics:
                mlflow.log_metrics(evaluation_metrics.to_dict())
            
            # Log output artifacts
            if components_path.exists():
                mlflow.log_artifact(str(components_path), artifact_path="outputs")
            if relations_path.exists():
                mlflow.log_artifact(str(relations_path), artifact_path="outputs")
            
            # Log graph images as artifacts if generated
            if graph_image:
                images_dir = self.config.output_data_dir / "graphs"
                if images_dir.exists():
                    for img_file in images_dir.glob("*.png"):
                        mlflow.log_artifact(str(img_file), artifact_path="graphs")
            
            # Capture run_id before ending
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "unknown"
            
            # End run
            mlflow.end_run()
            logger.info("mlflow_run_complete", run_id=run_id)
        
        logger.info("pipeline_complete", 
                   processed_count=len(graphs),
                   failed_count=failed_count,
                   processing_time=f"{processing_time:.2f}s",
                   output_dir=str(self.config.output_data_dir))
        
        return graphs
    
    def process_single_text(self, text: str, text_id: str) -> ArgumentGraph:
        """Process a single text.
        
        Args:
            text: Input text to analyze
            text_id: Identifier for the text
            
        Returns:
            ArgumentGraph
        """
        return self.workflow.run(text, text_id)
