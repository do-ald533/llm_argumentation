"""Pydantic models for structured LLM outputs."""
from pydantic import BaseModel, Field
from typing import List, Literal


class ComponentClassification(BaseModel):
    """Classification for a single component."""
    
    component_id: int = Field(..., description="ID of the component")
    label: Literal["MajorClaim", "Claim", "Premise"] = Field(
        ...,
        description="Classification: MajorClaim (main conclusion), Claim (sub-conclusion), or Premise (supporting evidence)"
    )
    reasoning: str = Field(..., description="Brief explanation for the classification")


class BatchClassificationOutput(BaseModel):
    """Structured output for batch component classification."""
    
    classifications: List[ComponentClassification] = Field(
        ...,
        description="List of classifications for all components"
    )


class RelationOutput(BaseModel):
    """A single relation between components."""
    
    source_id: int = Field(..., description="ID of the source component")
    target_id: int = Field(..., description="ID of the target component")
    relation_type: Literal["support", "attack"] = Field(
        ...,
        description="Type of relation: 'support' (agrees with/provides evidence for) or 'attack' (contradicts/opposes)"
    )
    reasoning: str = Field(..., description="Brief explanation for this relation")


class BatchRelationOutput(BaseModel):
    """Structured output for batch relation extraction."""
    
    relations: List[RelationOutput] = Field(
        ...,
        description="List of all relations between components"
    )


class ComponentIdentification(BaseModel):
    """A single identified argumentative component."""
    
    component_id: int = Field(..., description="Sequential ID starting from 1")
    text: str = Field(..., description="The exact text of the component from the original text")


class ComponentIdentificationOutput(BaseModel):
    """Structured output for component identification."""
    
    components: List[ComponentIdentification] = Field(
        ...,
        description="List of all identified argumentative components"
    )


class ConclusionOutput(BaseModel):
    """Structured output for conclusion identification."""
    
    conclusion_id: int = Field(
        ...,
        description="The ID of the component that represents the main conclusion"
    )
    reasoning: str = Field(..., description="Brief explanation for why this is the main conclusion")
