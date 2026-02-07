"""Prompt templates for argumentation tasks.

This module contains all the prompt templates used in the pipeline.
Keeping the original prompts from prompts.py but organizing them better.
"""
from llm.prompts import (
    argumentative_components,
    components_corrected,
    merge_components,
    argumentative_conclusion,
    premise_support,
    premise_attack,
    missing_premise_support,
    missing_premise_attack,
    convergent_premises_support,
    convergent_premises_attack,
    implicit_prompt_support,
    implicit_prompt_attack,
    get_counterarguments,
    merge_components_cycle
)


class PromptManager:
    """Manager for all prompt templates."""
    
    # Re-export all prompt functions for easy access
    argumentative_components = staticmethod(argumentative_components)
    components_corrected = staticmethod(components_corrected)
    merge_components = staticmethod(merge_components)
    argumentative_conclusion = staticmethod(argumentative_conclusion)
    premise_support = staticmethod(premise_support)
    premise_attack = staticmethod(premise_attack)
    missing_premise_support = staticmethod(missing_premise_support)
    missing_premise_attack = staticmethod(missing_premise_attack)
    convergent_premises_support = staticmethod(convergent_premises_support)
    convergent_premises_attack = staticmethod(convergent_premises_attack)
    implicit_prompt_support = staticmethod(implicit_prompt_support)
    implicit_prompt_attack = staticmethod(implicit_prompt_attack)
    get_counterarguments = staticmethod(get_counterarguments)
    merge_components_cycle = staticmethod(merge_components_cycle)
    
    @staticmethod
    def component_classification(text: str, component_text: str, all_components: str) -> str:
        """Prompt for classifying a component as MajorClaim, Claim, or Premise.
        
        Args:
            text: Full original text
            component_text: The specific component to classify
            all_components: String representation of all components
            
        Returns:
            Formatted prompt
        """
        return f'''You are an expert in argumentation analysis. Your task is to classify an argumentative component as one of three types:

1. **MajorClaim**: The main thesis or central claim of the entire argument. There is typically only ONE major claim per text.
2. **Claim**: A statement that supports or attacks the major claim or other claims. Can be intermediate conclusions.
3. **Premise**: Evidence, reasons, or facts that directly support or attack claims.

Classification Guidelines:
- A MajorClaim is the ultimate conclusion the author wants to establish
- Claims are intermediate conclusions or significant statements
- Premises are supporting evidence, facts, examples, or reasons

Original Text:
{text}

All Argumentative Components:
{all_components}

Component to Classify:
"{component_text}"

Provide your classification in the following format:
Classification: [MajorClaim/Claim/Premise]
Reasoning: [Brief explanation of why]

Classification:'''
