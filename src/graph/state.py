"""State definitions for the argumentation workflow."""
from typing import TypedDict, Dict, List, Optional
from src.models import ArgumentComponent, ArgumentRelation


class WorkflowState(TypedDict):
    """State that flows through the argumentation pipeline.
    
    This state is passed between nodes in the LangGraph workflow.
    """
    # Input
    text_id: str
    text: str
    
    # Intermediate results
    components: Dict[int, ArgumentComponent]
    relations: List[ArgumentRelation]
    conclusion_id: Optional[int]
    
    # Metadata
    errors: List[str]
    current_step: str
