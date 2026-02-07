"""Task for identifying argumentative components."""
import re
from typing import Dict
from src.models import ArgumentComponent
from src.llm import LLMClient, PromptManager


class IdentificationTask:
    """Identifies argumentative components in text."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize identification task.
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm = llm_client
        self.prompts = PromptManager()
    
    def execute(self, text: str, text_id: str) -> Dict[int, ArgumentComponent]:
        """Identify argumentative components in text.
        
        Args:
            text: Input text to analyze
            text_id: Identifier for the text
            
        Returns:
            Dictionary mapping component IDs to ArgumentComponent objects
        """
        # Generate components using LLM
        prompt = self.prompts.argumentative_components(text)
        response = self.llm.generate(prompt)
        
        # Parse response into components
        components = self._parse_components(response, text_id)
        
        return components
    
    def _parse_components(self, response: str, text_id: str) -> Dict[int, ArgumentComponent]:
        """Parse LLM response into ArgumentComponent objects.
        
        Args:
            response: LLM response containing numbered components
            text_id: Identifier for the text
            
        Returns:
            Dictionary mapping component IDs to ArgumentComponent objects
        """
        pattern = r'(\d+)\s*-\s*(.+?)(?=\n\d+\s*-|\Z)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        components = {}
        for num_str, text in matches:
            comp_id = int(num_str)
            component = ArgumentComponent(
                id=comp_id,
                text=text.strip(),
                text_id=text_id,
                label="Premise"  # Default, will be classified later
            )
            components[comp_id] = component
        
        return components
