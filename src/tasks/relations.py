"""Task for extracting argumentative relations."""
from typing import Dict, List
from src.models import ArgumentComponent, ArgumentRelation
from src.llm import LLMClient, PromptManager, BatchRelationOutput
from src.logging_config import get_logger


class RelationExtractionTask:
    """Extracts support and attack relations between components."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize relation extraction task.
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm = llm_client
        self.prompts = PromptManager()
        self.logger = get_logger("relations")
    
    def execute(
        self,
        text: str,
        text_id: str,
        components: Dict[int, ArgumentComponent],
        conclusion_id: int
    ) -> List[ArgumentRelation]:
        """Extract relations between components using batch structured output.
        
        Args:
            text: Original text
            text_id: Text identifier
            components: Dictionary of components
            conclusion_id: ID of the conclusion
            
        Returns:
            List of ArgumentRelation objects
        """
        # Format components for prompt
        components_list = "\n".join(
            f"{comp_id}. [{comp.label}] {comp.text}"
            for comp_id, comp in sorted(components.items())
        )
        
        # Create batch relation extraction prompt
        prompt = f"""Identify ALL argumentative relations between components in the text.

**Relation Types:**
- **support**: Component A provides evidence for, agrees with, or strengthens component B
- **attack**: Component A contradicts, opposes, or weakens component B

**Original Text:**
{text}

**Components:**
{components_list}

**Main Conclusion (MajorClaim):** Component {conclusion_id}

**Task:** Identify all support and attack relations. Focus on:
1. Which components support the main conclusion ({conclusion_id})?
2. Which components support other claims?
3. Are there any counter-arguments (attacks)?

Return ALL relations you can identify."""
        
        try:
            # Use structured output for reliable parsing
            result = self.llm.generate_structured(
                prompt=prompt,
                response_model=BatchRelationOutput,
                system_message="You are an expert in argumentation mining and discourse analysis.",
            )
            
            # Convert to ArgumentRelation objects
            relations = []
            for rel_output in result.relations:
                # Validate IDs exist
                if rel_output.source_id in components and rel_output.target_id in components:
                    relation = ArgumentRelation(
                        source_id=rel_output.source_id,
                        target_id=rel_output.target_id,
                        text_id=text_id,
                        relation_type=rel_output.relation_type
                    )
                    relations.append(relation)
                    self.logger.debug(
                        "relation_extracted",
                        source=rel_output.source_id,
                        target=rel_output.target_id,
                        type=rel_output.relation_type,
                        reasoning=rel_output.reasoning
                    )
            
            return relations
            
        except Exception as e:
            self.logger.error("batch_relation_extraction_failed", error=str(e))
            # Fallback: return empty list
            return []
