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
from src.utils import parse_answer_ids, parse_support_attack_ids, format_components_string
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
        conclusion_id: int,
        enable_partial_attack: bool = False
    ) -> Tuple[List[ArgumentRelation], Set[int], List[int]]:
        """Extract relations via recursive BFS from the main conclusion.
        
        For each visited node (starting from the conclusion), we ask the LLM
        which components directly support it, attack it, and (when enabled)
        partially attack it. Premises are forbidden from being ancestors or
        siblings of the current node. Each identified premise is queued.
        
        Args:
            text: Original text
            text_id: Text identifier
            components: Dictionary of components
            conclusion_id: ID of the main conclusion
            enable_partial_attack: When True, the LLM is also asked to identify
                partial-attack relations (AbstRCT only).
            
        Returns:
            Tuple of (relations, visited_set, unvisited_list)
        """
        dict_components = {cid: comp.text for cid, comp in components.items()}
        arg_components = format_components_string(dict_components)
        
        relations: List[ArgumentRelation] = []
        visited: Set[int] = set()
        queue: deque = deque([conclusion_id])
        children: Dict[int, List[int]] = {}
        
        while queue:
            current = queue.popleft()
            
            if current in visited:
                continue
            visited.add(current)
            
            forbidden: Set[int] = set()
            for parent, siblings in children.items():
                if current in siblings:
                    forbidden = set(siblings)
                    break
            forbidden |= visited
            
            self.logger.debug("visiting_node", node=current, text_id=text_id)

            support_ids, attack_ids, partial_attack_ids = self._get_relations(
                current, text, arg_components, dict_components, forbidden,
                enable_partial_attack=enable_partial_attack
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

            if partial_attack_ids:
                children.setdefault(current, []).extend(partial_attack_ids)
                for prem_id in partial_attack_ids:
                    if prem_id not in visited:
                        relations.append(ArgumentRelation(
                            source_id=prem_id,
                            target_id=current,
                            text_id=text_id,
                            relation_type="partial_attack"
                        ))
                        queue.append(prem_id)
        
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
    
    def _get_relations(
        self,
        conclusion_id: int,
        text: str,
        arg_components: str,
        dict_components: Dict[int, str],
        forbidden: Set[int],
        enable_partial_attack: bool = False
    ) -> Tuple[List[int], List[int], List[int]]:
        """Ask LLM which components support, attack, and partially attack the conclusion.

        Issues a single API call using the combined premise_relations prompt.

        Returns:
            Tuple (support_ids, attack_ids, partial_attack_ids)
        """
        available_ids = {
            cid for cid in dict_components
            if cid not in forbidden and cid != conclusion_id
        }

        if not available_ids:
            return [], [], []

        prompt = self.prompts.premise_relations(
            conclusion_id, text, arg_components, dict_components,
            available_ids=available_ids,
            enable_partial_attack=enable_partial_attack
        )
        response = self.llm.generate(prompt)
        raw_support, raw_attack, raw_partial = parse_support_attack_ids(response)

        valid_support = [
            pid for pid in raw_support
            if pid in dict_components and pid not in forbidden
        ]
        valid_attack = [
            pid for pid in raw_attack
            if pid in dict_components and pid not in forbidden
        ]
        valid_partial = [
            pid for pid in raw_partial
            if pid in dict_components and pid not in forbidden
        ] if enable_partial_attack else []

        if valid_support:
            self.logger.debug(
                "support_found", conclusion=conclusion_id, premises=valid_support
            )
        if valid_attack:
            self.logger.debug(
                "attack_found", conclusion=conclusion_id, premises=valid_attack
            )
        if valid_partial:
            self.logger.debug(
                "partial_attack_found", conclusion=conclusion_id, premises=valid_partial
            )

        return valid_support, valid_attack, valid_partial
