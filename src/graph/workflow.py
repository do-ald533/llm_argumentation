"""LangGraph workflow for argumentation structuring pipeline."""
from langgraph.graph import StateGraph, END
from typing import Dict, Any
from src.graph.state import WorkflowState
from src.tasks import (
    IdentificationTask,
    ClassificationTask,
    RelationExtractionTask,
    ConclusionExtractionTask
)
from src.llm import LLMClient
from src.models import ArgumentGraph
from src.config import Config
from src.logging_config import get_logger


class ArgumentationWorkflow:
    """Orchestrates the argumentation structuring pipeline using LangGraph."""
    
    def __init__(self, config: Config):
        """Initialize workflow.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.llm_client = LLMClient(config)
        self.logger = get_logger("workflow")
        
        # Initialize tasks
        self.identification_task = IdentificationTask(self.llm_client)
        self.classification_task = ClassificationTask(self.llm_client)
        self.relation_task = RelationExtractionTask(self.llm_client)
        self.conclusion_task = ConclusionExtractionTask(self.llm_client)
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("identify_components", self._identify_components)
        workflow.add_node("extract_conclusion", self._extract_conclusion)
        workflow.add_node("classify_components", self._classify_components)
        workflow.add_node("extract_relations", self._extract_relations)
        workflow.add_node("finalize", self._finalize)
        
        # Define edges
        workflow.set_entry_point("identify_components")
        workflow.add_edge("identify_components", "extract_conclusion")
        workflow.add_edge("extract_conclusion", "classify_components")
        workflow.add_edge("classify_components", "extract_relations")
        workflow.add_edge("extract_relations", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def run(self, text: str, text_id: str) -> ArgumentGraph:
        """Run the workflow on input text.
        
        Args:
            text: Input text to analyze
            text_id: Identifier for the text
            
        Returns:
            Complete ArgumentGraph
        """
        # Initialize state
        initial_state: WorkflowState = {
            "text_id": text_id,
            "text": text,
            "components": {},
            "relations": [],
            "conclusion_id": None,
            "errors": [],
            "current_step": "start"
        }
        
        # Run workflow
        self.logger.info("processing_text", text_id=text_id)
        
        final_state = self.workflow.invoke(initial_state)
        
        # Convert to ArgumentGraph
        graph = ArgumentGraph(
            text_id=final_state["text_id"],
            text=final_state["text"],
            components=final_state["components"],
            relations=final_state["relations"],
            conclusion_id=final_state["conclusion_id"]
        )
        
        self.logger.info("text_completed",
                        text_id=text_id,
                        component_count=len(graph.components),
                        relation_count=len(graph.relations),
                        conclusion_id=graph.conclusion_id)
        
        return graph
    
    def _identify_components(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: Identify argumentative components."""
        self.logger.debug("step_identify_components", text_id=state["text_id"])
        
        try:
            components = self.identification_task.execute(
                state["text"],
                state["text_id"]
            )
            self.logger.debug("components_identified", count=len(components))
            
            return {
                "components": components,
                "current_step": "identify_components"
            }
        except Exception as e:
            self.logger.error("identify_components_failed", error=str(e))
            return {
                "errors": state["errors"] + [f"Identification error: {e}"],
                "current_step": "identify_components"
            }
    
    def _extract_conclusion(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: Extract main conclusion."""
        self.logger.debug("step_extract_conclusion", text_id=state["text_id"])
        
        try:
            conclusion_id = self.conclusion_task.execute(
                state["text"],
                state["components"]
            )
            self.logger.debug("conclusion_extracted", conclusion_id=conclusion_id)
            
            return {
                "conclusion_id": conclusion_id,
                "current_step": "extract_conclusion"
            }
        except Exception as e:
            self.logger.error("extract_conclusion_failed", error=str(e))
            return {
                "errors": state["errors"] + [f"Conclusion error: {e}"],
                "current_step": "extract_conclusion"
            }
    
    def _classify_components(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: Classify components as MajorClaim/Claim/Premise."""
        self.logger.debug("step_classify_components", text_id=state["text_id"])
        
        try:
            components = self.classification_task.execute(
                state["text"],
                state["components"],
                state["conclusion_id"]
            )
            
            # Count classifications
            label_counts = {}
            for comp in components.values():
                label_counts[comp.label] = label_counts.get(comp.label, 0) + 1
            
            self.logger.debug("components_classified", classifications=label_counts)
            
            return {
                "components": components,
                "current_step": "classify_components"
            }
        except Exception as e:
            self.logger.error("classify_components_failed", error=str(e))
            return {
                "errors": state["errors"] + [f"Classification error: {e}"],
                "current_step": "classify_components"
            }
    
    def _extract_relations(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: Extract support/attack relations."""
        self.logger.debug("step_extract_relations", text_id=state["text_id"])
        
        try:
            relations = self.relation_task.execute(
                state["text"],
                state["text_id"],
                state["components"],
                state["conclusion_id"]
            )
            
            # Count relation types
            support_count = sum(1 for r in relations if r.relation_type == "support")
            attack_count = sum(1 for r in relations if r.relation_type == "attack")
            
            self.logger.debug("relations_extracted", support_count=support_count, attack_count=attack_count)
            
            return {
                "relations": relations,
                "current_step": "extract_relations"
            }
        except Exception as e:
            self.logger.error("extract_relations_failed", error=str(e))
            return {
                "errors": state["errors"] + [f"Relation error: {e}"],
                "current_step": "extract_relations"
            }
    
    def _finalize(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: Finalize workflow."""
        return {"current_step": "complete"}
