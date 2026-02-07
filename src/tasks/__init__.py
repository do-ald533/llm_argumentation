"""Task implementations for the argumentation pipeline."""
from .identification import IdentificationTask
from .classification import ClassificationTask
from .relations import RelationExtractionTask
from .conclusion import ConclusionExtractionTask

__all__ = [
    "IdentificationTask",
    "ClassificationTask",
    "RelationExtractionTask",
    "ConclusionExtractionTask"
]
