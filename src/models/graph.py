"""Argumentation graph data model."""
from pydantic import BaseModel, Field
from typing import Optional
import networkx as nx
from .components import ArgumentComponent, ArgumentRelation


class ArgumentGraph(BaseModel):
    """Complete argumentation graph for a single text."""
    
    text_id: str = Field(..., description="Source document identifier")
    text: str = Field(..., description="Original text")
    components: dict[int, ArgumentComponent] = Field(
        default_factory=dict,
        description="Components indexed by ID"
    )
    relations: list[ArgumentRelation] = Field(
        default_factory=list,
        description="Relations between components"
    )
    conclusion_id: Optional[int] = Field(
        default=None,
        description="ID of the main conclusion/major claim"
    )
    
    def add_component(self, component: ArgumentComponent) -> None:
        """Add a component to the graph."""
        self.components[component.id] = component
    
    def add_relation(self, relation: ArgumentRelation) -> None:
        """Add a relation to the graph."""
        self.relations.append(relation)
    
    def get_component_by_id(self, component_id: int) -> Optional[ArgumentComponent]:
        """Get a component by its ID."""
        return self.components.get(component_id)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph.
        
        Returns:
            NetworkX DiGraph with components as nodes and relations as edges
        """
        G = nx.DiGraph()
        
        # Add nodes
        for comp_id, comp in self.components.items():
            G.add_node(comp_id, text=comp.text, label=comp.label)
        
        # Add edges
        for rel in self.relations:
            G.add_edge(
                rel.source_id,
                rel.target_id,
                relation=rel.relation_type,
                is_convergent=rel.is_convergent
            )
        
        return G
    
    def to_golden_standard_components(self) -> list[dict]:
        """Export components in golden standard CSV format.
        
        Returns:
            List of dictionaries with keys: text_id, component_tokens, labels
        """
        return [comp.to_golden_standard() for comp in self.components.values()]
    
    def to_golden_standard_relations(self) -> list[dict]:
        """Export relations in golden standard CSV format.
        
        Returns:
            List of dictionaries with keys: text_id, source_tokens, target_tokens, labels
        """
        return [rel.to_golden_standard(self.components) for rel in self.relations]
    
    class Config:
        arbitrary_types_allowed = True
