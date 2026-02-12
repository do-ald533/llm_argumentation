"""Task implementations for the argumentation pipeline."""
from .identification import IdentificationTask
from .conclusion import ConclusionExtractionTask
from .relations import RelationExtractionTask
from .unvisited import UnvisitedPremisesTask

__all__ = [
    "IdentificationTask",
    "ConclusionExtractionTask",
    "RelationExtractionTask",
    "UnvisitedPremisesTask"
]
