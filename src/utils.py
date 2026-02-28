"""Utility functions for parsing LLM responses and formatting components."""
import re
from typing import Dict, List, Tuple


def parse_answer_ids(response: str) -> List[int]:
    """Parse component IDs from LLM response in 'Answer: n, m, ...' format.
    
    Handles formats:
        - 'Answer: 0'       → []
        - 'Answer: 3'       → [3]
        - 'Answer: 1, 2, 5' → [1, 2, 5]
    
    Args:
        response: Raw LLM output string
        
    Returns:
        List of component IDs (empty if none found or answer is 0)
    """
    match = re.search(r'Answer:\s*(.+)', response, re.IGNORECASE)
    if match:
        answer_text = match.group(1).strip()
        if answer_text.strip() == '0':
            return []
        numbers = re.findall(r'\d+', answer_text)
        ids = [int(n) for n in numbers]
        return [i for i in ids if i != 0]
    
    numbers = re.findall(r'\d+', response)
    if numbers:
        ids = [int(n) for n in numbers]
        return [i for i in ids if i != 0]
    
    return []


def parse_support_attack_ids(response: str) -> tuple:
    """Parse support, attack and partial-attack IDs from a combined relation prompt response.

    Expects two or three lines in the response:
        Support:       <comma-separated IDs or 0>
        Attack:        <comma-separated IDs or 0>
        Partial-Attack: <comma-separated IDs or 0>  (optional — absent when not enabled)

    Args:
        response: Raw LLM output string

    Returns:
        Tuple (support_ids, attack_ids, partial_attack_ids) — each is a list of ints
        (empty for 0 / not found). partial_attack_ids is always [] when the line is absent.
    """
    def _extract(label: str) -> List[int]:
        match = re.search(rf'{label}:\s*(.+)', response, re.IGNORECASE)
        if not match:
            return []
        answer_text = match.group(1).strip()
        if answer_text == '0':
            return []
        numbers = re.findall(r'\d+', answer_text)
        return [int(n) for n in numbers if int(n) != 0]

    return _extract('Support'), _extract('Attack'), _extract('Partial-Attack')


def parse_conclusion_id(response: str) -> int:
    """Parse conclusion ID from LLM response in 'CONCLUSION: n' format.
    
    Args:
        response: Raw LLM output string
        
    Returns:
        Conclusion component ID (defaults to 1 if unparseable)
    """
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
    
    numbers = re.findall(r'\d+', response)
    if numbers:
        return int(numbers[0])
    
    return 1


def parse_components(response: str) -> Dict[int, str]:
    """Parse numbered components from LLM response.
    
    Handles format:
        1 - Component text here
        2 - Another component text
    
    Args:
        response: Raw LLM output string
        
    Returns:
        Dictionary mapping component IDs to text
    """
    pattern = r'(\d+)\s*-\s*(.+?)(?=\n\d+\s*-|\Z)'
    matches = re.findall(pattern, response, re.DOTALL)
    
    components = {}
    for num_str, text in matches:
        comp_id = int(num_str)
        components[comp_id] = text.strip()
    
    return components


def format_components_string(components: Dict[int, str]) -> str:
    """Format a components dictionary as a numbered list string.
    
    Args:
        components: Dictionary mapping component IDs to text
        
    Returns:
        Formatted string like '1 - text\\n2 - text\\n...'
    """
    return '\n'.join(
        f"{comp_id} - {text}"
        for comp_id, text in sorted(components.items())
    )
