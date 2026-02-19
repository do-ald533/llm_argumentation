"""LangGraph workflow for argumentation structuring pipeline.

Pipeline steps:
  1. Identify argumentative components
  2. Identify the main conclusion
  3. Recursive BFS relation extraction (support/attack)
  4. Check & assign unvisited premises
  5. Finalize: derive labels from graph structure
"""
from langgraph.graph import StateGraph, END
from typing import Dict, Any
from src.graph.state import WorkflowState
from src.tasks import (
    IdentificationTask,
    ConclusionExtractionTask,
    RelationExtractionTask,
    UnvisitedPremisesTask
)
from src.llm import LLMClient
from src.models import ArgumentGraph
from src.config import Config
from src.logging_config import get_logger


class ArgumentationWorkflow:
    """Orchestrates the argumentation structuring pipeline using LangGraph."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.logger = get_logger("workflow")
        
        self.identification_task = IdentificationTask(self.llm_client)
        self.conclusion_task = ConclusionExtractionTask(self.llm_client)
        self.relation_task = RelationExtractionTask(self.llm_client)
        self.unvisited_task = UnvisitedPremisesTask(self.llm_client)
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Steps: identify → conclude → relations (BFS) → unvisited → finalize
        """
        workflow = StateGraph(WorkflowState)
        
        workflow.add_node("identify_components", self._identify_components)
        workflow.add_node("extract_conclusion", self._extract_conclusion)
        workflow.add_node("extract_relations", self._extract_relations)
        workflow.add_node("check_unvisited", self._check_unvisited)
        workflow.add_node("finalize", self._finalize)
        
        workflow.set_entry_point("identify_components")
        workflow.add_edge("identify_components", "extract_conclusion")
        workflow.add_edge("extract_conclusion", "extract_relations")
        workflow.add_edge("extract_relations", "check_unvisited")
        workflow.add_edge("check_unvisited", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def run(self, text: str, text_id: str) -> ArgumentGraph:
        """Run the workflow on input text.
        
        Args:
            text: Input text to analyze
            text_id: Identifier for the text
            
        Returns:
            Complete ArgumentGraph with derived labels
        """
        initial_state: WorkflowState = {
            "text_id": text_id,
            "text": text,
            "components": {},
            "conclusion_id": None,
            "relations": [],
            "visited": [],
            "unvisited": [],
            "errors": [],
            "current_step": "start"
        }
        
        self.logger.info("processing_text", text_id=text_id)
        
        final_state = self.workflow.invoke(initial_state)
        
        graph = ArgumentGraph(
            text_id=final_state["text_id"],
            text=final_state["text"],
            components=final_state["components"],
            relations=final_state["relations"],
            conclusion_id=final_state["conclusion_id"]
        )
        
        graph.derive_labels()
        
        self.logger.info(
            "text_completed",
            text_id=text_id,
            component_count=len(graph.components),
            relation_count=len(graph.relations),
            conclusion_id=graph.conclusion_id
        )
        
        return graph
    
    def _identify_components(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 1: Identify argumentative components."""
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
        """Node 2: Identify the main conclusion."""
        self.logger.debug("step_extract_conclusion", text_id=state["text_id"])
        
        if not state["components"]:
            return {
                "errors": state["errors"] + ["No components to find conclusion from"],
                "current_step": "extract_conclusion"
            }
        
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
    
    def _extract_relations(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 3: Recursive BFS relation extraction."""
        self.logger.debug("step_extract_relations", text_id=state["text_id"])
        
        if not state["components"] or state["conclusion_id"] is None:
            return {
                "errors": state["errors"] + ["Cannot extract relations without components/conclusion"],
                "current_step": "extract_relations"
            }
        
        try:
            relations, visited, unvisited = self.relation_task.execute(
                state["text"],
                state["text_id"],
                state["components"],
                state["conclusion_id"]
            )
            
            support_count = sum(1 for r in relations if r.relation_type == "support")
            attack_count = sum(1 for r in relations if r.relation_type == "attack")
            self.logger.debug(
                "relations_extracted",
                support=support_count,
                attack=attack_count,
                unvisited=len(unvisited)
            )
            
            return {
                "relations": relations,
                "visited": list(visited),
                "unvisited": unvisited,
                "current_step": "extract_relations"
            }
        except Exception as e:
            self.logger.error("extract_relations_failed", error=str(e))
            return {
                "errors": state["errors"] + [f"Relation error: {e}"],
                "current_step": "extract_relations"
            }
    
    def _check_unvisited(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 4: Assign unvisited premises to the graph."""
        self.logger.debug("step_check_unvisited", text_id=state["text_id"])
        
        if not state["unvisited"]:
            self.logger.debug("no_unvisited_components")
            return {"current_step": "check_unvisited"}
        
        try:
            components, relations, conclusion_id = self.unvisited_task.execute(
                state["text"],
                state["text_id"],
                state["components"],
                state["relations"],
                state["unvisited"],
                state["conclusion_id"]
            )
            
            return {
                "components": components,
                "relations": relations,
                "conclusion_id": conclusion_id,
                "unvisited": [],
                "current_step": "check_unvisited"
            }
        except Exception as e:
            self.logger.error("check_unvisited_failed", error=str(e))
            return {
                "errors": state["errors"] + [f"Unvisited error: {e}"],
                "current_step": "check_unvisited"
            }
    
    def _finalize(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 5: Finalize — validate relations, enforce DAG, transitive reduction."""
        graph = ArgumentGraph(
            text_id=state["text_id"],
            text=state["text"],
            components=state["components"],
            relations=state["relations"],
            conclusion_id=state["conclusion_id"]
        )
        
        graph.validate_relations()
        
        graph.ensure_dag()
        
        graph.transitive_reduction()
        
        self.logger.info(
            "finalize_complete",
            text_id=state["text_id"],
            relations_after_cleanup=len(graph.relations)
        )
        
        return {
            "relations": graph.relations,
            "current_step": "complete"
        }
