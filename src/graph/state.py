"""State definitions for the argumentation workflow."""
from typing import TypedDict, Dict, List, Optional, Set
from src.models import ArgumentComponent, ArgumentRelation


class WorkflowState(TypedDict):
    """State that flows through the argumentation pipeline.
    
    This state is passed between nodes in the LangGraph workflow.
    """
    # Input
    text_id: str
    text: str
    
    # Step 1: Identified components
    components: Dict[int, ArgumentComponent]
    
    # Step 2: Conclusion
    conclusion_id: Optional[int]
    
    # Step 3: Relations (from recursive BFS)
    relations: List[ArgumentRelation]
    visited: List[int]
    unvisited: List[int]
    
    # Metadata
    errors: List[str]
    current_step: str
