"""Task for extracting the main conclusion."""
import re
from typing import Dict
from src.models import ArgumentComponent
from src.llm import LLMClient, PromptManager


class ConclusionExtractionTask:
    """Extracts the main conclusion from argumentative components."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize conclusion extraction task.
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm = llm_client
        self.prompts = PromptManager()
    
    def execute(
        self,
        text: str,
        components: Dict[int, ArgumentComponent]
    ) -> int:
        """Extract the main conclusion.
        
        Args:
            text: Original text
            components: Dictionary of identified components
            
        Returns:
            ID of the conclusion component
        """
        # Format components for prompt
        arg_components = self._format_components(components)
        
        # Generate prompt and get response
        prompt = self.prompts.argumentative_conclusion(text, arg_components)
        response = self.llm.generate(prompt)
        
        # Extract conclusion ID
        conclusion_id = self._extract_conclusion_id(response)
        
        # Validate conclusion ID
        if conclusion_id not in components:
            # Fallback: use last component
            conclusion_id = max(components.keys())
        
        return conclusion_id
    
    @staticmethod
    def _extract_conclusion_id(response: str) -> int:
        """Extract conclusion ID from LLM response.
        
        Args:
            response: LLM response
            
        Returns:
            Conclusion component ID
        """
        # Look for patterns like "CONCLUSION: 5" or "Answer: 5"
        patterns = [
            r'CONCLUSION:\s*(\d+)',
            r'Answer:\s*(\d+)',
            r'conclusion\s*(?:is|:)?\s*(\d+)',
            r'^(\d+)\s*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return int(match.group(1))
        
        # Fallback: try to find any number
        numbers = re.findall(r'\d+', response)
        if numbers:
            return int(numbers[0])
        
        return 1  # Ultimate fallback
    
    @staticmethod
    def _format_components(components: Dict[int, ArgumentComponent]) -> str:
        """Format components for prompt.
        
        Args:
            components: Dictionary of components
            
        Returns:
            Formatted string
        """
        return '\n'.join(
            f"{comp_id} - {comp.text}"
            for comp_id, comp in sorted(components.items())
        )
