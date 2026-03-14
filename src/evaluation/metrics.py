"""Metrics for evaluating argumentation extraction against golden standard."""
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel
from src.logging_config import get_logger

logger = get_logger("evaluation")


class EvaluationMetrics(BaseModel):
    """Container for evaluation metrics."""
    
    component_precision: float = 0.0
    component_recall: float = 0.0
    component_f1: float = 0.0
    component_predicted: int = 0
    component_gold: int = 0
    component_correct: int = 0
    
    relation_precision: float = 0.0
    relation_recall: float = 0.0
    relation_f1: float = 0.0
    relation_predicted: int = 0
    relation_gold: int = 0
    relation_correct: int = 0
    
    total_texts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MLflow logging."""
        return self.model_dump()
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"""
Evaluation Metrics
==================
Components:
  Precision: {self.component_precision:.3f} ({self.component_correct}/{self.component_predicted})
  Recall:    {self.component_recall:.3f} ({self.component_correct}/{self.component_gold})
  F1 Score:  {self.component_f1:.3f}

Relations:
  Precision: {self.relation_precision:.3f} ({self.relation_correct}/{self.relation_predicted})
  Recall:    {self.relation_recall:.3f} ({self.relation_correct}/{self.relation_gold})
  F1 Score:  {self.relation_f1:.3f}

Total Texts: {self.total_texts}
"""


def evaluate_against_golden_standard(
    predicted_components_path: Path,
    predicted_relations_path: Path,
    golden_components_path: Optional[Path] = None,
    golden_relations_path: Optional[Path] = None
) -> EvaluationMetrics:
    """Evaluate predicted outputs against golden standard.
    
    Args:
        predicted_components_path: Path to predicted components CSV
        predicted_relations_path: Path to predicted relations CSV
        golden_components_path: Path to golden standard components CSV (optional)
        golden_relations_path: Path to golden standard relations CSV (optional)
        
    Returns:
        EvaluationMetrics object with computed metrics
    """
    metrics = EvaluationMetrics()
    
    try:
        pred_components = pl.read_csv(predicted_components_path, separator=";")
        pred_relations = pl.read_csv(predicted_relations_path, separator=";")
    except Exception as e:
        logger.error("failed_to_load_predictions", error=str(e))
        return metrics
    
    metrics.component_predicted = len(pred_components)
    metrics.relation_predicted = len(pred_relations)
    metrics.total_texts = pred_components.select("text_id").unique().height
    
    if golden_components_path is None or golden_relations_path is None:
        logger.warning("no_golden_standard_provided", 
                      message="Only counting predictions, no comparison metrics")
        return metrics
    
    if not golden_components_path.exists() or not golden_relations_path.exists():
        logger.warning("golden_standard_not_found",
                      components_path=str(golden_components_path),
                      relations_path=str(golden_relations_path))
        return metrics
    
    try:
        gold_components = pl.read_csv(golden_components_path)
        gold_relations = pl.read_csv(golden_relations_path)
    except Exception as e:
        logger.error("failed_to_load_golden_standard", error=str(e))
        return metrics
    
    metrics.component_gold = len(gold_components)
    metrics.relation_gold = len(gold_relations)
    
    metrics.component_correct = _count_matching_components(pred_components, gold_components)
    
    metrics.relation_correct = _count_matching_relations(pred_relations, gold_relations)
    
    metrics.component_precision = _safe_division(metrics.component_correct, metrics.component_predicted)
    metrics.component_recall = _safe_division(metrics.component_correct, metrics.component_gold)
    metrics.component_f1 = _f1_score(metrics.component_precision, metrics.component_recall)
    
    metrics.relation_precision = _safe_division(metrics.relation_correct, metrics.relation_predicted)
    metrics.relation_recall = _safe_division(metrics.relation_correct, metrics.relation_gold)
    metrics.relation_f1 = _f1_score(metrics.relation_precision, metrics.relation_recall)
    
    logger.info("evaluation_complete",
                component_f1=f"{metrics.component_f1:.3f}",
                relation_f1=f"{metrics.relation_f1:.3f}")
    
    return metrics


def _count_matching_components(predicted: pl.DataFrame, golden: pl.DataFrame) -> int:
    """Count components that match between predicted and golden standard.
    
    Matches on (text_id, component_tokens) pairs.
    """
    pred_keys = predicted.select([
        pl.col("text_id"),
        pl.col("component_tokens")
    ]).unique()
    
    gold_keys = golden.select([
        pl.col("text_id"),
        pl.col("component_tokens")
    ]).unique()
    
    matches = pred_keys.join(
        gold_keys,
        on=["text_id", "component_tokens"],
        how="inner"
    )
    
    return len(matches)


def _count_matching_relations(predicted: pl.DataFrame, golden: pl.DataFrame) -> int:
    """Count relations that match between predicted and golden standard.
    
    Matches on (text_id, source_tokens, target_tokens, labels) tuples.
    """
    pred_keys = predicted.select([
        pl.col("text_id"),
        pl.col("source_tokens"),
        pl.col("target_tokens"),
        pl.col("labels")
    ]).unique()
    
    gold_keys = golden.select([
        pl.col("text_id"),
        pl.col("source_tokens"),
        pl.col("target_tokens"),
        pl.col("labels")
    ]).unique()
    
    matches = pred_keys.join(
        gold_keys,
        on=["text_id", "source_tokens", "target_tokens", "labels"],
        how="inner"
    )
    
    return len(matches)


def _safe_division(numerator: float, denominator: float) -> float:
    """Safe division that returns 0 if denominator is 0."""
    return numerator / denominator if denominator > 0 else 0.0


def _f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
