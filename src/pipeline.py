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
        
        if config.enable_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment_name)
            logger.info("mlflow_initialized", 
                       tracking_uri=config.mlflow_tracking_uri,
                       experiment=config.mlflow_experiment_name)
    
    def _load_preloaded_components(self, path: Path) -> dict:
        """Load components from a CSV file and index them by text_id.

        Accepts both comma-separated (golden standard) and semicolon-separated
        (pipeline output) files — the separator is detected automatically.
        Required columns: text_id, component_tokens, labels.
        Returns a dict mapping text_id -> {int: ArgumentComponent}.
        """
        from src.models import ArgumentComponent

        # Auto-detect separator: try comma first, fall back to semicolon
        required = {"text_id", "component_tokens", "labels"}
        df = pl.read_csv(path, separator=",")
        if not required.issubset(set(df.columns)):
            df = pl.read_csv(path, separator=";")
        if not required.issubset(set(df.columns)):
            raise ValueError(
                f"Components file '{path}' must have columns: "
                f"text_id, component_tokens, labels. "
                f"Found: {df.columns}"
            )

        # Drop rows with null component text (safety net)
        before = len(df)
        df = df.filter(pl.col("component_tokens").is_not_null())
        dropped = before - len(df)
        if dropped:
            logger.warning("preloaded_components_null_rows_dropped",
                           path=str(path), count=dropped)

        grouped: dict = {}
        for row in df.iter_rows(named=True):
            text_id = row["text_id"]
            if text_id not in grouped:
                grouped[text_id] = {}
            comp_id = len(grouped[text_id]) + 1
            grouped[text_id][comp_id] = ArgumentComponent(
                id=comp_id,
                text=row["component_tokens"],
                text_id=text_id,
                label=row["labels"]
            )

        total = sum(len(v) for v in grouped.values())
        logger.info("preloaded_components_loaded",
                    path=str(path),
                    text_count=len(grouped),
                    total_components=total)

        if total != len(df):
            logger.warning("preloaded_components_count_mismatch",
                           csv_rows=len(df), loaded=total)

        return grouped

    def process_csv(
        self, 
        input_file: Path, 
        output_prefix: str,
        limit: Optional[int] = None,
        golden_components_path: Optional[Path] = None,
        golden_relations_path: Optional[Path] = None,
        run_name: Optional[str] = None,
        graph_image: bool = False,
        skip_identification_path: Optional[Path] = None,
        enable_partial_attack: bool = False
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
        
        logger.info("loading_input_data", path=str(input_file))
        df = pl.read_csv(input_file)
        
        if "text_id" not in df.columns or "text_tokens" not in df.columns:
            raise ValueError("CSV must have 'text_id' and 'text_tokens' columns")
        
        if limit:
            df = df.head(limit)
        
        total_texts = len(df)
        logger.info("texts_to_process", count=total_texts)

        preloaded_components_by_text: dict = {}
        if skip_identification_path:
            if not skip_identification_path.exists():
                raise FileNotFoundError(
                    f"--skip-identification file not found: {skip_identification_path}"
                )
            preloaded_components_by_text = self._load_preloaded_components(
                skip_identification_path
            )
            logger.info("identification_step_skipped",
                        source=str(skip_identification_path))

        mlflow_active = self.config.enable_mlflow and MLFLOW_AVAILABLE
        
        if mlflow_active:
            if run_name is None:
                run_name = f"{output_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
            
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "llm_model": self.config.openai_model,
                "temperature": self.config.temperature,
                "max_retries": self.config.max_retries,
                "dataset": output_prefix,
                "input_file": str(input_file),
                "total_texts": total_texts,
                "prompt_version": self.config.prompt_version,
                "prompt_hash": get_prompt_hash(),
                "skip_identification": str(skip_identification_path) if skip_identification_path else "false",
                "enable_partial_attack": str(enable_partial_attack),
            })
            
            prompts_file = Path("src/llm/prompts.py")
            if prompts_file.exists():
                mlflow.log_artifact(str(prompts_file), artifact_path="prompts")
        
        graphs = []
        failed_count = 0
        
        for idx, row in enumerate(df.iter_rows(named=True), 1):
            text_id = row["text_id"]
            text = row["text_tokens"]
            
            logger.info("processing_text", number=idx, total=total_texts, text_id=text_id)
            
            try:
                preloaded = preloaded_components_by_text.get(text_id)
                graph = self.workflow.run(
                    text, text_id,
                    preloaded_components=preloaded,
                    enable_partial_attack=enable_partial_attack
                )
                graphs.append(graph)
                
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
        
        logger.info("exporting_results", prefix=output_prefix)
        
        export_to_golden_standard(
            graphs,
            self.config.output_data_dir,
            prefix=output_prefix
        )
        
        components_path = self.config.output_data_dir / f"components_{output_prefix}.csv"
        relations_path = self.config.output_data_dir / f"relations_{output_prefix}.csv"
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
        
        if mlflow_active:
            mlflow.log_metrics({
                "texts_processed": len(graphs),
                "texts_failed": failed_count,
                "processing_time_seconds": processing_time,
                "avg_time_per_text": processing_time / total_texts if total_texts > 0 else 0,
            })
            
            if evaluation_metrics:
                mlflow.log_metrics(evaluation_metrics.to_dict())
            if components_path.exists():
                mlflow.log_artifact(str(components_path), artifact_path="outputs")
            if relations_path.exists():
                mlflow.log_artifact(str(relations_path), artifact_path="outputs")
            
            if graph_image:
                images_dir = self.config.output_data_dir / "graphs"
                if images_dir.exists():
                    for img_file in images_dir.glob("*.png"):
                        mlflow.log_artifact(str(img_file), artifact_path="graphs")
            
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "unknown"
            
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
