"""Task for handling unvisited premises (Step 4).

After the recursive BFS relation extraction, some components may remain
unassigned. This task prompts the LLM to determine which conclusion each
unvisited component supports or attacks. If two unvisited components
point to each other (cycle), they are merged into one and re-assigned.
"""
from typing import Dict, List, Tuple, Set, FrozenSet
import networkx as nx
from src.models import ArgumentComponent, ArgumentRelation
from src.llm import LLMClient, PromptManager
from src.utils import parse_answer_ids, format_components_string
from src.logging_config import get_logger


class UnvisitedPremisesTask:
    """Assigns unvisited components to the argument graph."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.prompts = PromptManager()
        self.logger = get_logger("unvisited")
    
    def execute(
        self,
        text: str,
        text_id: str,
        components: Dict[int, ArgumentComponent],
        relations: List[ArgumentRelation],
        unvisited: List[int],
        conclusion_id: int
    ) -> Tuple[Dict[int, ArgumentComponent], List[ArgumentRelation], int]:
        """Assign unvisited components to the argument graph.
        
        For each unvisited component, asks the LLM which conclusion it
        supports or attacks. Detects cycles among newly assigned premises
        and merges them if needed.
        
        Args:
            text: Original text
            text_id: Text identifier
            components: Dictionary of components (may be modified)
            relations: Existing relations list (will be extended)
            unvisited: List of unvisited component IDs
            conclusion_id: ID of the main conclusion
            
        Returns:
            Tuple of (updated_components, updated_relations, updated_conclusion_id)
        """
        if not unvisited:
            self.logger.info("no_unvisited_components", text_id=text_id)
            return components, relations, conclusion_id
        
        self.logger.info(
            "processing_unvisited",
            text_id=text_id,
            count=len(unvisited)
        )
        
        dict_components = {cid: comp.text for cid, comp in components.items()}
        arg_components = format_components_string(dict_components)
        
        # Step 1: For each unvisited component, find its target
        temp_links: List[Tuple[int, int, str]] = []
        still_orphan: List[int] = []
        
        for premise_id in unvisited:
            self.logger.debug("checking_unvisited", premise_id=premise_id)
            
            found_any = False
            
            # Check support
            support_targets = self._get_support_targets(
                premise_id, text, arg_components, dict_components
            )
            for t in support_targets:
                if t in dict_components and t != premise_id:
                    temp_links.append((premise_id, t, "support"))
                    found_any = True
            
            # Check attack
            attack_targets = self._get_attack_targets(
                premise_id, text, arg_components, dict_components
            )
            for t in attack_targets:
                if t in dict_components and t != premise_id:
                    temp_links.append((premise_id, t, "attack"))
                    found_any = True
            
            # Fallback: if LLM returned nothing valid, link to conclusion
            if not found_any:
                still_orphan.append(premise_id)
                self.logger.warning(
                    "orphan_fallback_to_conclusion",
                    premise_id=premise_id,
                    conclusion_id=conclusion_id,
                    text_id=text_id,
                )
                if conclusion_id is not None and premise_id != conclusion_id:
                    temp_links.append((premise_id, conclusion_id, "support"))
        
        # Step 2: Detect cycles among the newly created links
        # Build a directed graph from temp_links and find all cycles
        link_graph = nx.DiGraph()
        for a, b, _ in temp_links:
            link_graph.add_edge(a, b)
        
        # Find all simple cycles among the unvisited links
        all_cycles: List[List[int]] = list(nx.simple_cycles(link_graph))
        
        # Collect all component IDs involved in any cycle
        cycle_nodes: Set[int] = set()
        for cycle in all_cycles:
            cycle_nodes.update(cycle)
        
        # Step 3: Handle cycles by merging
        merged_ids: Set[int] = set()
        if all_cycles:
            self.logger.info("cycles_detected", count=len(all_cycles), nodes=sorted(cycle_nodes))
            
            # Merge all cycle-involved nodes together into a single component
            component_ids = sorted(cycle_nodes)
            merged_ids.update(component_ids)
            self.logger.debug("merging_cycle", components=component_ids)
            
            # Merge the cycled components
            components, dict_components, new_id = self._merge_cycle(
                text, arg_components, dict_components,
                components, component_ids, text_id
            )
            arg_components = format_components_string(dict_components)
            
            # Remove any existing relations referencing merged (deleted) IDs
            relations = [
                r for r in relations
                if r.source_id in components and r.target_id in components
            ]
            
            # Ask what the merged component supports/attacks
            for mode in ("support", "attack"):
                if mode == "support":
                    targets = self._get_support_targets(
                        new_id, text, arg_components, dict_components
                    )
                else:
                    targets = self._get_attack_targets(
                        new_id, text, arg_components, dict_components
                    )
                
                for t in targets:
                    if t in dict_components and t != new_id:
                        relations.append(ArgumentRelation(
                            source_id=new_id,
                            target_id=t,
                            text_id=text_id,
                            relation_type=mode
                        ))
        
        # Step 4: Add remaining non-cyclic links (from components not involved in any cycle)
        for a, b, kind in temp_links:
            if a not in merged_ids and b not in merged_ids:
                if a in components and b in components:
                    relations.append(ArgumentRelation(
                        source_id=a,
                        target_id=b,
                        text_id=text_id,
                        relation_type=kind
                    ))
            elif a not in merged_ids and b in merged_ids:
                # The target was merged — redirect to conclusion
                if a in components and conclusion_id is not None and a != conclusion_id:
                    relations.append(ArgumentRelation(
                        source_id=a,
                        target_id=conclusion_id,
                        text_id=text_id,
                        relation_type=kind
                    ))
        
        # Step 5: Final safety net — ensure no component has zero edges
        # Any component still disconnected gets linked to the conclusion
        connected = set()
        for r in relations:
            connected.add(r.source_id)
            connected.add(r.target_id)
        
        for comp_id in components:
            if comp_id not in connected and comp_id != conclusion_id:
                self.logger.warning(
                    "final_orphan_linked_to_conclusion",
                    component_id=comp_id,
                    conclusion_id=conclusion_id,
                    text_id=text_id,
                )
                if conclusion_id is not None:
                    relations.append(ArgumentRelation(
                        source_id=comp_id,
                        target_id=conclusion_id,
                        text_id=text_id,
                        relation_type="support"
                    ))
        
        self.logger.info(
            "unvisited_complete",
            text_id=text_id,
            new_relations=len(temp_links)
        )
        
        return components, relations, conclusion_id
    
    def _get_support_targets(
        self,
        premise_id: int,
        text: str,
        arg_components: str,
        dict_components: Dict[int, str]
    ) -> List[int]:
        """Ask LLM which components this premise directly supports."""
        prompt = self.prompts.missing_premise_support(
            premise_id, text, arg_components, dict_components
        )
        response = self.llm.generate(prompt)
        return parse_answer_ids(response)
    
    def _get_attack_targets(
        self,
        premise_id: int,
        text: str,
        arg_components: str,
        dict_components: Dict[int, str]
    ) -> List[int]:
        """Ask LLM which components this premise directly attacks."""
        prompt = self.prompts.missing_premise_attack(
            premise_id, text, arg_components, dict_components
        )
        response = self.llm.generate(prompt)
        return parse_answer_ids(response)
    
    def _merge_cycle(
        self,
        text: str,
        arg_components: str,
        dict_components: Dict[int, str],
        components: Dict[int, ArgumentComponent],
        component_ids: List[int],
        text_id: str
    ) -> Tuple[Dict[int, ArgumentComponent], Dict[int, str], int]:
        """Merge cycled components into a single new component.
        
        Args:
            text: Original text
            arg_components: Formatted components string
            dict_components: ID-to-text mapping
            components: Full ArgumentComponent dict
            component_ids: IDs of components to merge
            text_id: Text identifier
            
        Returns:
            Tuple of (updated_components, updated_dict, new_component_id)
        """
        prompt = self.prompts.merge_components_cycle(
            text, arg_components, dict_components, component_ids
        )
        merged_text = self.llm.generate(prompt).strip()
        
        # Remove old components
        for cid in component_ids:
            dict_components.pop(cid, None)
            components.pop(cid, None)
        
        # Create new component with next available ID
        new_id = max(dict_components.keys(), default=0) + 1
        dict_components[new_id] = merged_text
        components[new_id] = ArgumentComponent(
            id=new_id,
            text=merged_text,
            text_id=text_id,
            label="Premise"  # Will be derived from graph structure later
        )
        
        self.logger.debug(
            "cycle_merged",
            old_ids=component_ids,
            new_id=new_id,
            merged_text=merged_text[:80]
        )
        
        return components, dict_components, new_id
