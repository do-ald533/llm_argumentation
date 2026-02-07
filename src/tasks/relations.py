"""Task for extracting argumentative relations."""
import re
from typing import Dict, List, Set, Tuple
from src.models import ArgumentComponent, ArgumentRelation
from src.llm import LLMClient, PromptManager


class RelationExtractionTask:
    """Extracts support and attack relations between components."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize relation extraction task.
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm = llm_client
        self.prompts = PromptManager()
    
    def execute(
        self,
        text: str,
        text_id: str,
        components: Dict[int, ArgumentComponent],
        conclusion_id: int
    ) -> List[ArgumentRelation]:
        """Extract relations between components.
        
        Args:
            text: Original text
            text_id: Text identifier
            components: Dictionary of components
            conclusion_id: ID of the conclusion
            
        Returns:
            List of ArgumentRelation objects
        """
        relations = []
        visited = set()
        to_visit = [conclusion_id]
        
        # Format components for prompts
        arg_components = self._format_components(components)
        
        # Breadth-first traversal from conclusion
        while to_visit:
            current_id = to_visit.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Find components that support current
            support_ids = self._find_support_relations(
                text, arg_components, components, current_id, visited
            )
            
            for source_id in support_ids:
                relation = ArgumentRelation(
                    source_id=source_id,
                    target_id=current_id,
                    text_id=text_id,
                    relation_type="support"
                )
                relations.append(relation)
                
                if source_id not in visited:
                    to_visit.append(source_id)
            
            # Find components that attack current
            attack_ids = self._find_attack_relations(
                text, arg_components, components, current_id, visited
            )
            
            for source_id in attack_ids:
                relation = ArgumentRelation(
                    source_id=source_id,
                    target_id=current_id,
                    text_id=text_id,
                    relation_type="attack"
                )
                relations.append(relation)
                
                if source_id not in visited:
                    to_visit.append(source_id)
        
        return relations
    
    def _find_support_relations(
        self,
        text: str,
        arg_components: str,
        components: Dict[int, ArgumentComponent],
        target_id: int,
        forbidden: Set[int]
    ) -> List[int]:
        """Find components that support the target.
        
        Args:
            text: Original text
            arg_components: Formatted components string
            components: Dictionary of components
            target_id: Target component ID
            forbidden: Set of IDs to exclude
            
        Returns:
            List of source component IDs
        """
        prompt = self.prompts.premise_support(
            target_id,
            text,
            arg_components,
            components
        )
        
        response = self.llm.generate(prompt, max_tokens=300)
        
        # Extract IDs from response
        ids = self._extract_ids(response)
        
        # Filter out forbidden and invalid IDs
        valid_ids = [
            id for id in ids
            if id in components and id not in forbidden and id != target_id
        ]
        
        return valid_ids
    
    def _find_attack_relations(
        self,
        text: str,
        arg_components: str,
        components: Dict[int, ArgumentComponent],
        target_id: int,
        forbidden: Set[int]
    ) -> List[int]:
        """Find components that attack the target.
        
        Args:
            text: Original text
            arg_components: Formatted components string
            components: Dictionary of components
            target_id: Target component ID
            forbidden: Set of IDs to exclude
            
        Returns:
            List of source component IDs
        """
        prompt = self.prompts.premise_attack(
            target_id,
            text,
            arg_components,
            components
        )
        
        response = self.llm.generate(prompt, max_tokens=300)
        
        # Extract IDs from response
        ids = self._extract_ids(response)
        
        # Filter out forbidden and invalid IDs
        valid_ids = [
            id for id in ids
            if id in components and id not in forbidden and id != target_id
        ]
        
        return valid_ids
    
    @staticmethod
    def _extract_ids(response: str) -> List[int]:
        """Extract component IDs from LLM response.
        
        Args:
            response: LLM response
            
        Returns:
            List of extracted IDs
        """
        # Look for "Answer: [id1, id2, ...]" pattern
        answer_match = re.search(r'Answer:\s*\[?([\d,\s]+)\]?', response, re.IGNORECASE)
        if answer_match:
            ids_str = answer_match.group(1)
            ids = [int(x.strip()) for x in ids_str.split(',') if x.strip().isdigit()]
            return [id for id in ids if id != 0]
        
        # Fallback: extract all numbers
        ids = re.findall(r'\b\d+\b', response)
        return [int(id) for id in ids if int(id) != 0][:5]  # Limit to 5
    
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
