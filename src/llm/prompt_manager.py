"""Prompt manager for argumentation tasks.

Thin wrapper that re-exports the prompt functions used by the pipeline.
"""
from src.llm.prompts import (
    argumentative_components,
    argumentative_conclusion,
    premise_support,
    premise_attack,
    premise_relations,
    missing_premise_support,
    missing_premise_attack,
    merge_components_cycle,
)


class PromptManager:
    """Manager for all prompt templates."""
    
    argumentative_components = staticmethod(argumentative_components)
    argumentative_conclusion = staticmethod(argumentative_conclusion)
    premise_support = staticmethod(premise_support)
    premise_attack = staticmethod(premise_attack)
    premise_relations = staticmethod(premise_relations)
    missing_premise_support = staticmethod(missing_premise_support)
    missing_premise_attack = staticmethod(missing_premise_attack)
    merge_components_cycle = staticmethod(merge_components_cycle)
