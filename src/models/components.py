"""Core data models for argumentative components and relations."""
from pydantic import BaseModel, Field
from typing import Literal, Optional


class ArgumentComponent(BaseModel):
    """Represents a single argumentative component."""
    
    id: int = Field(..., description="Unique identifier for the component")
    text: str = Field(..., description="Text content of the component")
    text_id: str = Field(..., description="Source document identifier (e.g., 'AAEC_004')")
    label: Literal["MajorClaim", "Claim", "Premise"] = Field(
        default="Premise",
        description="Type of argumentative component"
    )
    
    def to_golden_standard(self) -> dict:
        """Convert to golden standard CSV format.
        
        Returns:
            dict with keys: text_id, component_tokens, labels
        """
        return {
            "text_id": self.text_id,
            "component_tokens": self.text,
            "labels": self.label
        }
    
    class Config:
        frozen = False  # Allow mutation for label updates


class ArgumentRelation(BaseModel):
    """Represents a relation between two argumentative components."""
    
    source_id: int = Field(..., description="ID of the source component")
    target_id: int = Field(..., description="ID of the target component")
    text_id: str = Field(..., description="Source document identifier")
    relation_type: Literal["support", "attack"] = Field(..., description="Type of relation")
    is_convergent: bool = Field(default=False, description="Whether part of convergent argument")
    
    def to_golden_standard(self, components: dict[int, ArgumentComponent]) -> dict:
        """Convert to golden standard CSV format.
        
        Args:
            components: Dictionary mapping component IDs to ArgumentComponent objects
            
        Returns:
            dict with keys: text_id, source_tokens, target_tokens, labels
        """
        return {
            "text_id": self.text_id,
            "source_tokens": components[self.source_id].text,
            "target_tokens": components[self.target_id].text,
            "labels": self.relation_type
        }
    
    class Config:
        frozen = True
