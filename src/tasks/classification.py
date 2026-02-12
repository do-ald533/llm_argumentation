"""Task for classifying argumentative components."""
from typing import Dict
from src.models import ArgumentComponent
from src.llm import LLMClient, PromptManager, BatchClassificationOutput
from src.logging_config import get_logger


class ClassificationTask:
    """Classifies components as MajorClaim, Claim, or Premise."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize classification task.
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm = llm_client
        self.prompts = PromptManager()
        self.logger = get_logger("classification")
    
    def execute(
        self,
        text: str,
        components: Dict[int, ArgumentComponent],
        conclusion_id: int
    ) -> Dict[int, ArgumentComponent]:
        """Classify all components using batch structured output.
        
        Args:
            text: Original text
            components: Dictionary of components to classify
            conclusion_id: ID of the identified conclusion
            
        Returns:
            Updated components dictionary with classifications
        """
        # Mark conclusion as MajorClaim
        components[conclusion_id].label = "MajorClaim"
        
        # Get list of components to classify (excluding conclusion)
        to_classify = {
            comp_id: comp
            for comp_id, comp in components.items()
            if comp_id != conclusion_id
        }
        
        if not to_classify:
            return components
        
        # Format components for prompt
        components_list = "\n".join(
            f"{comp_id}. {comp.text}"
            for comp_id, comp in sorted(components.items())
        )
        
        # Create batch classification prompt
        prompt = f"""Classify each argumentative component as either "Claim" or "Premise".

**Definitions:**
- **Claim**: A sub-conclusion or intermediate argument that supports the main conclusion
- **Premise**: Evidence, facts, or reasons that directly support claims

**Original Text:**
{text}

**All Components:**
{components_list}

**Main Conclusion (MajorClaim):** Component {conclusion_id}

**Task:** Classify each component (except {conclusion_id}) as either "Claim" or "Premise".

Components to classify: {", ".join(str(cid) for cid in sorted(to_classify.keys()))}
"""
        
        try:
            # Use structured output for reliable parsing
            result = self.llm.generate_structured(
                prompt=prompt,
                response_model=BatchClassificationOutput,
                system_message="You are an expert in argumentation mining and discourse analysis.",
            )
            
            # Apply classifications
            for classification in result.classifications:
                if classification.component_id in components:
                    components[classification.component_id].label = classification.label
                    self.logger.debug(
                        "component_classified",
                        component_id=classification.component_id,
                        label=classification.label,
                        reasoning=classification.reasoning
                    )
            
        except Exception as e:
            self.logger.error("batch_classification_failed", error=str(e))
            # Fallback: mark everything as Premise
            for comp_id in to_classify:
                components[comp_id].label = "Premise"
        
        return components
