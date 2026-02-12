"""Task for extracting argumentative relations via recursive BFS.

Step 3 of the pipeline: Given the main conclusion, recursively identify
support and attack relations by visiting each conclusion and finding its
direct premises. Each conclusion is visited only once. Premises for a
given conclusion must not originate from higher levels (ancestors) or
from the same level (siblings).
"""
from collections import deque
from typing import Dict, List, Set, Tuple
from src.models import ArgumentComponent, ArgumentRelation
from src.llm import LLMClient, PromptManager
from src.utils import parse_answer_ids, format_components_string
from src.logging_config import get_logger


class RelationExtractionTask:
    """Extracts support/attack relations using recursive BFS from the conclusion."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.prompts = PromptManager()
        self.logger = get_logger("relations")
    
    def execute(
        self,
        text: str,
        text_id: str,
        components: Dict[int, ArgumentComponent],
        conclusion_id: int
    ) -> Tuple[List[ArgumentRelation], Set[int], List[int]]:
        """Extract relations via recursive BFS from the main conclusion.
        
        For each visited node (starting from the conclusion), we ask the LLM
        which components directly support it and which directly attack it.
        Premises are forbidden from being ancestors or siblings of the current
        node.  Each identified premise is then queued for visitation.
        
        Args:
            text: Original text
            text_id: Text identifier
            components: Dictionary of components
            conclusion_id: ID of the main conclusion
            
        Returns:
            Tuple of (relations, visited_set, unvisited_list)
        """
        dict_components = {cid: comp.text for cid, comp in components.items()}
        arg_components = format_components_string(dict_components)
        
        relations: List[ArgumentRelation] = []
        visited: Set[int] = set()
        queue: deque = deque([conclusion_id])
        # Track children of each node to compute forbidden siblings
        children: Dict[int, List[int]] = {}
        
        while queue:
            current = queue.popleft()
            
            if current in visited:
                continue
            visited.add(current)
            
            # Compute forbidden nodes: siblings (other premises of same parent)
            forbidden: Set[int] = set()
            for parent, siblings in children.items():
                if current in siblings:
                    forbidden = set(siblings)
                    break
            # Also forbid all ancestors (already visited nodes)
            forbidden |= visited
            
            self.logger.debug("visiting_node", node=current, text_id=text_id)
            
            # --- Find supporting premises ---
            support_ids = self._get_support_premises(
                current, text, arg_components, dict_components, forbidden
            )
            
            if support_ids:
                children.setdefault(current, []).extend(support_ids)
                for prem_id in support_ids:
                    if prem_id not in visited:
                        relations.append(ArgumentRelation(
                            source_id=prem_id,
                            target_id=current,
                            text_id=text_id,
                            relation_type="support"
                        ))
                        queue.append(prem_id)
            
            # --- Find attacking premises ---
            attack_ids = self._get_attack_premises(
                current, text, arg_components, dict_components, forbidden
            )
            
            if attack_ids:
                children.setdefault(current, []).extend(attack_ids)
                for prem_id in attack_ids:
                    if prem_id not in visited:
                        relations.append(ArgumentRelation(
                            source_id=prem_id,
                            target_id=current,
                            text_id=text_id,
                            relation_type="attack"
                        ))
                        queue.append(prem_id)
        
        # Determine unvisited components
        all_ids = set(components.keys())
        unvisited = sorted(all_ids - visited)
        
        self.logger.info(
            "relations_extracted",
            text_id=text_id,
            relations_count=len(relations),
            visited_count=len(visited),
            unvisited_count=len(unvisited)
        )
        
        return relations, visited, unvisited
    
    def _get_support_premises(
        self,
        conclusion_id: int,
        text: str,
        arg_components: str,
        dict_components: Dict[int, str],
        forbidden: Set[int]
    ) -> List[int]:
        """Ask LLM which components directly support the given conclusion."""
        prompt = self.prompts.premise_support(
            conclusion_id, text, arg_components, dict_components
        )
        response = self.llm.generate(prompt)
        raw_ids = parse_answer_ids(response)
        
        # Filter out forbidden and invalid IDs
        valid_ids = [
            pid for pid in raw_ids
            if pid in dict_components and pid not in forbidden
        ]
        
        if valid_ids:
            self.logger.debug(
                "support_found", conclusion=conclusion_id, premises=valid_ids
            )
        
        return valid_ids
    
    def _get_attack_premises(
        self,
        conclusion_id: int,
        text: str,
        arg_components: str,
        dict_components: Dict[int, str],
        forbidden: Set[int]
    ) -> List[int]:
        """Ask LLM which components directly attack the given conclusion."""
        prompt = self.prompts.premise_attack(
            conclusion_id, text, arg_components, dict_components
        )
        response = self.llm.generate(prompt)
        raw_ids = parse_answer_ids(response)
        
        # Filter out forbidden and invalid IDs
        valid_ids = [
            pid for pid in raw_ids
            if pid in dict_components and pid not in forbidden
        ]
        
        if valid_ids:
            self.logger.debug(
                "attack_found", conclusion=conclusion_id, premises=valid_ids
            )
        
        return valid_ids
