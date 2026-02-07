"""Task for classifying argumentative components."""
import re
from typing import Dict, Literal
from src.models import ArgumentComponent
from src.llm import LLMClient, PromptManager


class ClassificationTask:
    """Classifies components as MajorClaim, Claim, or Premise."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize classification task.
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm = llm_client
        self.prompts = PromptManager()
    
    def execute(
        self,
        text: str,
        components: Dict[int, ArgumentComponent],
        conclusion_id: int
    ) -> Dict[int, ArgumentComponent]:
        """Classify all components.
        
        Args:
            text: Original text
            components: Dictionary of components to classify
            conclusion_id: ID of the identified conclusion
            
        Returns:
            Updated components dictionary with classifications
        """
        # Format all components for context
        all_components_str = self._format_components(components)
        
        # Classify each component
        for comp_id, component in components.items():
            if comp_id == conclusion_id:
                # Conclusion is the MajorClaim
                component.label = "MajorClaim"
            else:
                # Classify using LLM
                label = self._classify_component(
                    text,
                    component.text,
                    all_components_str
                )
                component.label = label
        
        return components
    
    def _classify_component(
        self,
        text: str,
        component_text: str,
        all_components: str
    ) -> Literal["MajorClaim", "Claim", "Premise"]:
        """Classify a single component.
        
        Args:
            text: Original text
            component_text: Component to classify
            all_components: Formatted string of all components
            
        Returns:
            Classification label
        """
        prompt = self.prompts.component_classification(
            text,
            component_text,
            all_components
        )
        
        response = self.llm.generate(prompt, max_tokens=300)
        
        # Extract classification from response
        label = self._extract_classification(response)
        return label
    
    @staticmethod
    def _extract_classification(response: str) -> Literal["MajorClaim", "Claim", "Premise"]:
        """Extract classification label from LLM response.
        
        Args:
            response: LLM response
            
        Returns:
            Classification label (defaults to "Premise" if unclear)
        """
        response_lower = response.lower()
        
        if "majorclaim" in response_lower or "major claim" in response_lower:
            return "MajorClaim"
        elif "claim" in response_lower and "major" not in response_lower:
            return "Claim"
        else:
            return "Premise"
    
    @staticmethod
    def _format_components(components: Dict[int, ArgumentComponent]) -> str:
        """Format components for prompt context.
        
        Args:
            components: Dictionary of components
            
        Returns:
            Formatted string
        """
        return '\n'.join(
            f"{comp_id} - {comp.text}"
            for comp_id, comp in sorted(components.items())
        )
