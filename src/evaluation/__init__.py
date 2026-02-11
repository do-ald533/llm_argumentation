"""Evaluation module for comparing outputs against golden standard."""
from .metrics import EvaluationMetrics, evaluate_against_golden_standard

__all__ = ["EvaluationMetrics", "evaluate_against_golden_standard"]
