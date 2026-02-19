"""State definitions for the argumentation workflow."""
from typing import TypedDict, Dict, List, Optional, Set
from src.models import ArgumentComponent, ArgumentRelation


class WorkflowState(TypedDict):
    """State that flows through the argumentation pipeline.
    
    This state is passed between nodes in the LangGraph workflow.
    """
    text_id: str
    text: str
    
    components: Dict[int, ArgumentComponent]
    
    conclusion_id: Optional[int]
    
    relations: List[ArgumentRelation]
    visited: List[int]
    unvisited: List[int]
    
    errors: List[str]
    current_step: str
