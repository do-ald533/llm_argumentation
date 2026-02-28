"""
Prompt templates for LLM-based argumentation structuring tasks.

This module contains all prompt generation functions used by the pipeline.
Each function returns a formatted prompt string for specific argumentation
analysis tasks.

Pipeline steps:
  1. argumentative_components  – identify components
  2. argumentative_conclusion  – pick the main conclusion
  3. premise_support / premise_attack  – BFS relation extraction
  4. missing_premise_support / missing_premise_attack  – unvisited premises
     merge_components_cycle  – cycle resolution
"""

from typing import Dict, List, Optional, Set



def argumentative_components(text: str) -> str:
    """Generate prompt to identify argumentative components in text."""
    prompt = f'''An argument consists of premises that support or challenge a conclusion that is not self-evident. To reconstruct the argumentative structure of a text, the first step is to identify its argumentative components, which include both premises and conclusions.

Below are examples in which the argumentative components have been extracted from a short passage.

Example 1
Text:
"The death penalty would only be sustained as a legitimate resource if it were fair. It is not fair. Therefore, it should be abolished. It is not fair because it presupposes discriminatory mechanisms. After all, a non-white murderer of a white victim will be much more likely sent to execution than the reverse."
Argumentative Components:
1 - The death penalty would only be sustained as a legitimate resource if it were fair.
2 - It is not fair.
3 - Therefore, it should be abolished.
4 - It is not fair because it presupposes discriminatory mechanisms.
5 - A non-white murderer of a white victim will be much more likely sent to execution than the reverse.

Example 2
Text:
"Intelligence gathered by this and other governments leaves no doubt that the Iraq regime continues to possess and conceal some of the most lethal weapons ever devised. This regime has already used weapons of mass destruction against Iraq's neighbors and against Iraq's people. The regime has a history of reckless aggression in the Middle East. It has a deep hatred of America and our friends. And it has aided trained and harbored terrorists including operatives of al Qaeda. The danger is clear: using chemical biological or one day nuclear weapons obtained with the help of Iraq the terrorists could fulfill their stated ambitions and kill thousands or hundreds of thousands of innocent people in our country or any other."
Argumentative Components:
1 - This regime has already used weapons of mass destruction against Iraq's neighbors and against Iraq's people.
2 - The regime has a history of reckless aggression in the Middle East.
3 - The danger is clear: using chemical biological or one day nuclear weapons obtained with the help of Iraq the terrorists could fulfill their stated ambitions and kill thousands or hundreds of thousands of innocent people in our country or any other.
4 - Intelligence gathered by this and other governments leaves no doubt that the Iraq regime continues to possess and conceal some of the most lethal weapons ever devised.
5 - It has a deep hatred of America and our friends.
6 - And it has aided trained and harbored terrorists including operatives of al Qaeda.

Your Task: Identify and list the argumentative components in the following text.

Instructions:
- Extract only the exact spans of text that correspond to premises or conclusions.
- Do not paraphrase or alter the original wording.
- Each component should be listed on a new line in the format: NUMBER - component
- Respond with the argumentative components and nothing else.

Text: {text}

Argumentative Components:
'''
    return prompt


# =============================================================================
# STEP 2: CONCLUSION IDENTIFICATION
# =============================================================================

def argumentative_conclusion(text: str, arguments: str) -> str:
    """Generate prompt to identify the main conclusion of an argument."""
    prompt = f'''You are given a short argumentative passage plus a list of its numbered statements (called "components"). Your task is to find the main conclusion of the argument.

• The main conclusion is the statement that the other components are offered to prove or justify.
• The main conclusion itself is not used to support any further claim.

How to decide (internal reasoning, do not print):

- Ask "What is the author ultimately trying to convince the reader of?"

- Check which statement is supported by at least one other component but does not itself support anything further.

- Prefer normative/ evaluative claims ("should", "best", "therefore we must…") over purely descriptive ones when both are present.

Worked examples
Example 1

Text
"If we increase funding for early childhood education, we will see long-term improvements in academic performance, because early learning builds foundational skills that support later success."

Components
1 - Early learning builds foundational skills that support later success.
2 - If we increase funding for early childhood education, we will see long-term improvements in academic performance.

Solution (explanation)
1 is a premise that supports 2; 2 supports nothing further.
CONCLUSION: 2

Example 2

Text
"Plastic pollution is a major threat to marine life. Many species are dying from ingestion or entanglement. Therefore, stricter regulations on plastic waste are necessary."

Components
1 - Plastic pollution is a major threat to marine life.
2 - Many species are dying from ingestion or entanglement.
3 - Stricter regulations on plastic waste are necessary.

CONCLUSION: 3

Additional Instructions:
- Give the conclusion number and nothing else.
- Output format: CONCLUSION: <number>

Now solve:
Text
{text}

Components
{arguments}

Your answer:
'''
    return prompt



def premise_support(
    conclusion_number: int,
    text: str,
    arg_components: str,
    dict_components: Dict[int, str],
    available_ids: Optional[Set[int]] = None,
) -> str:
    """Generate prompt to identify premises that directly support a conclusion.

    Args:
        conclusion_number: Target component ID.
        text: Original text.
        arg_components: All numbered components (for context).
        dict_components: ID -> text mapping.
        available_ids: If given, restrict the LLM to only consider these IDs.
    """
    if available_ids:
        eligible = sorted(available_ids)
        eligible_str = ", ".join(str(i) for i in eligible)
        constraint = (
            f"\n**Important**: You may ONLY choose from the following component numbers: "
            f"{eligible_str}.\nDo NOT include any component numbers outside this set.\n"
        )
    else:
        constraint = ""

    instruction = f"""You will be given a short argumentative text and its numbered components.
Your job is to decide **which components directly support** a specified target component.

### What counts as *direct support*?
A component gives *direct support* when it supplies a reason, justification, evidence,
example, or explanation **for the target claim itself**, without first depending on another
intermediate component to make the link.

A component is **not** a direct supporter if it:
* Only supports another component that *then* supports the target (indirect / chained support).
* Undermines, contradicts, or attacks the target.

### Important guidance
- Examine every eligible component and consider whether it provides a reason to believe the target.
- In argumentative texts most components relate to each other. Return 0 only when you are
  confident that no eligible component provides any justification for the target.
{constraint}
### Output format (strict)
Return exactly one line:
* Answer: 0                — when *no* component directly supports the target, or
* Answer: n                — one supporting component, or
* Answer: n, m, ...        — several supporters (ascending order, comma-separated).
No extra text.

### Worked examples

**Example 1**
Text:
Raising the minimum wage improves workers' quality of life. Higher income allows people to afford better housing. Better housing conditions can improve mental health.

Components:
1 - Raising the minimum wage improves workers' quality of life.
2 - Higher income allows people to afford better housing.
3 - Better housing conditions can improve mental health.

Target component: 1
Answer: 2
Explanation: 2 offers a concrete mechanism (better housing) that justifies 1. 3 depends on 2 and therefore is *indirect*.

**Example 2**
Text:
Phones distract students from learning. Many students use phones to cheat during exams. Schools should ban phones during class.

Components:
1 - Phones distract students from learning.
2 - Many students use phones to cheat during exams.
3 - Schools should ban phones during class.

Target component: 3
Answer: 1, 2
Explanation: Both 1 and 2 independently justify the policy in 3.

**Example 3**
Text:
Better public transport means shorter commute times. Improving public transport can reduce car usage. Shorter commutes lead to more productive workers.

Components:
1 - Better public transport means shorter commute times.
2 - Improving public transport can reduce car usage.
3 - Shorter commutes lead to more productive workers.

Target component: 2
Answer: 0
Explanation: Neither 1 nor 3 provides a direct reason for why public transport *reduces car usage*.

**Example 4**
Text:
Vitamin D deficiency is common in winter. Supplementing with vitamin D increases serum levels. A recent RCT showed vitamin D supplementation reduces depressive symptoms. The RCT had a 12-month follow-up confirming sustained mood improvement.

Components:
1 - Vitamin D deficiency is common in winter.
2 - Supplementing with vitamin D increases serum levels.
3 - A recent RCT showed vitamin D supplementation reduces depressive symptoms.
4 - The RCT had a 12-month follow-up confirming sustained mood improvement.

Target component: 3
Answer: 4
Explanation: 4 bolsters 3 by adding further evidence from the *same* study.

**Example 5**
Text:
Drinking water regularly improves focus. Dehydration is linked to lower cognitive performance. Many students forget to bring water bottles to school. Schools should ensure students have easy access to drinking water.

Components:
1 - Drinking water regularly improves focus.
2 - Dehydration is linked to lower cognitive performance.
3 - Many students forget to bring water bottles to school.
4 - Schools should ensure students have easy access to drinking water.

Target component: 4
Answer: 1, 2
Explanation: 1 and 2 each give an independent reason justifying the policy in 4. 3 is contextual, not a justification.

---
Now apply the same logic to the new case:

Text:
{text}

Components:
{arg_components}

Identify the components that **directly support** component {conclusion_number}: "{dict_components[conclusion_number]}".

Remember: examine every eligible component, decide whether it supplies a justification, and output your result in the strict format.

Answer:
"""
    return instruction


def premise_attack(
    conclusion_number: int,
    text: str,
    arg_components: str,
    dict_components: Dict[int, str],
    available_ids: Optional[Set[int]] = None,
) -> str:
    """Generate prompt to identify premises that directly attack a conclusion.

    Args:
        conclusion_number: Target component ID.
        text: Original text.
        arg_components: All numbered components (for context).
        dict_components: ID -> text mapping.
        available_ids: If given, restrict the LLM to only consider these IDs.
    """
    if available_ids:
        eligible = sorted(available_ids)
        eligible_str = ", ".join(str(i) for i in eligible)
        constraint = (
            f"\n**Important**: You may ONLY choose from the following component numbers: "
            f"{eligible_str}.\nDo NOT include any component numbers outside this set.\n"
        )
    else:
        constraint = ""

    instruction = f"""You will be given a list of argumentative components. Your task is to identify
*which* components **directly attack** a given target component.

### What counts as a direct attack?
A component *challenges, contradicts, limits, or undermines* the truth, certainty, relevance,
or persuasive force of the target claim — even if phrased cautiously (e.g. "however", "may not",
"could lead to negative outcomes"). Typical patterns include:

* Explicit disagreement with the claim.
* Pointing out exceptions or counter-examples.
* Questioning feasibility, effectiveness, or sufficiency.
* Introducing conditions that would invalidate the claim.

A component does *not* count as an attack if it:
* Merely elaborates, supports, or restates the target.
* Attacks another component but relies on an intermediate step (indirect attack).

### Important guidance
- Examine every eligible component and consider whether it undermines the target.
- Attack relations are less common than support in most texts.
  Return 0 if no eligible component genuinely challenges the target.
{constraint}
### Output format (strict)
Reply **only** with one of the following (no extra spaces):
* `Answer: 0`                — when nothing attacks the target, or
* `Answer: n`                — single attacking component, or
* `Answer: n, m, ...`        — several attackers, listed in ascending order.

### Worked examples

**Example 1**
Text:
Working from home boosts productivity. However, it can also blur work-life boundaries. Blurred boundaries may lead to burnout.

Components:
1 - Working from home boosts productivity.
2 - However, it can also blur work-life boundaries.
3 - Blurred boundaries may lead to burnout.

Target component: 1
Answer: 2
Explanation: 2 introduces a drawback that undermines the benefit in 1. 3 elaborates on 2 — not a direct attack on 1.

**Example 2**
Text:
Implementing a four-day work week will reduce stress. But this could also decrease overall productivity. Lower productivity may hurt company profits.

Components:
1 - Implementing a four-day work week will reduce stress.
2 - But this could also decrease overall productivity.
3 - Lower productivity may hurt company profits.

Target component: 1
Answer: 2
Explanation: 2 challenges 1 by highlighting a negative consequence. 3 depends on 2 and does not directly address 1.

**Example 3**
Text:
The environmental tax will help reduce emissions. Some argue that the tax is too low to make a real difference. In contrast, others believe it will still shift consumer behavior.

Components:
1 - The environmental tax will help reduce emissions.
2 - Some argue that the tax is too low to make a real difference.
3 - In contrast, others believe it will still shift consumer behavior.

Target component: 1
Answer: 2
Explanation: 2 questions the effectiveness of the tax, directly challenging 1. 3 counters 2, not 1.

**Example 4**
Text:
Extending the school day improves student performance. Longer school hours can lead to burnout and reduced motivation. Reduced motivation negatively impacts learning outcomes.

Components:
1 - Extending the school day improves student performance.
2 - Longer school hours can lead to burnout and reduced motivation.
3 - Reduced motivation negatively impacts learning outcomes.

Target component: 1
Answer: 2
Explanation: 2 presents a negative effect contradicting 1. 3 supports 2, not a direct attack on 1.

**Example 5**
Text:
A high-protein diet improves muscle growth. Some studies claim high-protein diets strain kidneys. A comprehensive meta-analysis found no link between high protein intake and kidney issues. However, the meta-analysis included mostly short-term studies, making its conclusions unreliable.

Components:
1 - A high-protein diet improves muscle growth.
2 - Some studies claim high-protein diets strain kidneys.
3 - A comprehensive meta-analysis found no link between high protein intake and kidney issues.
4 - However, the meta-analysis included mostly short-term studies, making its conclusions unreliable.

Target component: 3
Answer: 4
Explanation: 4 questions the methodology of 3, undermining its conclusion.

---
Now apply the same logic to the new case:

Text:
{text}

Components:
{arg_components}

Identify the components that **directly attack** component {conclusion_number}: "{dict_components[conclusion_number]}".

Answer:
"""
    return instruction



def premise_relations(
    conclusion_number: int,
    text: str,
    arg_components: str,
    dict_components: Dict[int, str],
    available_ids: Optional[Set[int]] = None,
    enable_partial_attack: bool = False,
) -> str:
    """Generate a single prompt that identifies support, attack and (optionally) partial-attack.

    Replaces calling premise_support + premise_attack separately, halving API calls
    during BFS relation extraction.

    Args:
        conclusion_number: Target component ID.
        text: Original text.
        arg_components: All numbered components (for context).
        dict_components: ID -> text mapping.
        available_ids: If given, restrict the LLM to only consider these IDs.
        enable_partial_attack: When True, adds a third Partial-Attack output line.
            Only use for datasets that annotate partial-attack (e.g. AbstRCT).
    """
    if available_ids:
        eligible = sorted(available_ids)
        eligible_str = ", ".join(str(i) for i in eligible)
        constraint = (
            f"\n**Important**: You may ONLY choose from the following component numbers: "
            f"{eligible_str}.\nDo NOT include any component numbers outside this set.\n"
        )
    else:
        constraint = ""

    if enable_partial_attack:
        relation_types = (
            "  • **Support**        – gives a reason, justification, evidence, or explanation directly for the target.\n"
            "  • **Attack**         – fully contradicts, negates, or invalidates the target.\n"
            "  • **Partial-Attack** – does NOT fully contradict the target, but *constrains* or *weakens* it by\n"
            "                         qualifying its scope, questioning its significance, or introducing a caveat\n"
            "                         (e.g. 'however, this should be confirmed by larger trials').\n"
            "  • **Neither**        – no direct relation to the target."
        )
        attack_guidance = (
            "- **Attack vs Partial-Attack**: use Attack only for full contradiction or negation.\n"
            "  Use Partial-Attack when the source *weakens or constrains* the target without fully opposing it.\n"
            "  Partial-Attack is common in medical abstracts as implicit statements about study limitations.\n"
            "- Return 0 for any category where no eligible component qualifies.\n"
            "- A component may appear in at most ONE category."
        )
        output_format = (
            "### Output format (strict — three lines, nothing else)\n"
            "Support: <comma-separated IDs in ascending order, or 0>\n"
            "Attack: <comma-separated IDs in ascending order, or 0>\n"
            "Partial-Attack: <comma-separated IDs in ascending order, or 0>"
        )
        extra_example = """
**Example 5 (Partial-Attack)**
Text:
SLN biopsy is an effective and well-tolerated procedure. However, its safety should be confirmed by the results of larger randomized trials and meta-analyses.

Components:
1 - SLN biopsy is an effective and well-tolerated procedure.
2 - However, its safety should be confirmed by the results of larger randomized trials and meta-analyses.

Target component: 1
Support: 0
Attack: 0
Partial-Attack: 2
Explanation: 2 does not negate 1 outright — it accepts the finding but constrains its certainty by calling for further confirmation. That is a partial-attack, not a full attack.
"""
        closing_reminder = (
            "Remember: examine every eligible component and output strictly three lines.\n"
            "Support: <IDs or 0>\n"
            "Attack: <IDs or 0>\n"
            "Partial-Attack: <IDs or 0>"
        )
    else:
        relation_types = (
            "  • **Support** – it gives a reason, justification, evidence, or explanation directly for the target.\n"
            "  • **Attack**  – it challenges, contradicts, limits, or undermines the target.\n"
            "  • **Neither** – it has no direct relation to the target."
        )
        attack_guidance = (
            "- Attack relations are less common than support. Only mark a component as Attack if it\n"
            "  genuinely challenges the target.\n"
            "- Return 0 for Support (or Attack) when no eligible component fills that role.\n"
            "- A component may only appear in Support OR Attack, never both."
        )
        output_format = (
            "### Output format (strict — two lines, nothing else)\n"
            "Support: <comma-separated IDs in ascending order, or 0>\n"
            "Attack: <comma-separated IDs in ascending order, or 0>"
        )
        extra_example = ""
        closing_reminder = (
            "Remember: examine every eligible component and output strictly two lines.\n"
            "Support: <IDs or 0>\n"
            "Attack: <IDs or 0>"
        )

    instruction = (
        "You will be given a short argumentative text and its numbered components.\n"
        "For a specified **target component**, your task is to classify every eligible component as:\n"
        + relation_types + "\n\n"
        "### Definitions\n"
        "**Direct support**: the component justifies the target *by itself*, without depending on an\n"
        "intermediate component to make the link. Indirect (chained) support does NOT count.\n\n"
        "**Direct attack**: the component contradicts or negates the target directly — explicit\n"
        "disagreement, counter-examples, or conditions that would invalidate the claim.\n"
        "Indirect attacks (via intermediate steps) do NOT count.\n\n"
        "### Important guidance\n"
        + attack_guidance + "\n"
        + constraint + "\n"
        + output_format + "\n\n"
        "### Worked examples\n\n"
        "**Example 1**\n"
        "Text:\n"
        "Raising the minimum wage improves workers' quality of life. Higher income allows people to afford better housing. "
        "Better housing conditions can improve mental health. However, some economists argue that higher wages reduce job availability.\n\n"
        "Components:\n"
        "1 - Raising the minimum wage improves workers' quality of life.\n"
        "2 - Higher income allows people to afford better housing.\n"
        "3 - Better housing conditions can improve mental health.\n"
        "4 - Some economists argue that higher wages reduce job availability.\n\n"
        "Target component: 1\n"
        + ("Support: 2\nAttack: 4\nPartial-Attack: 0\n" if enable_partial_attack else "Support: 2\nAttack: 4\n") +
        "Explanation: 2 justifies 1 directly; 4 undermines 1 by pointing to a negative side-effect. 3 is indirect.\n\n"
        "**Example 2**\n"
        "Text:\n"
        "Schools should ban phones during class. Phones distract students from learning. "
        "Many students use phones to cheat during exams. Smartphones can also be useful educational tools.\n\n"
        "Components:\n"
        "1 - Schools should ban phones during class.\n"
        "2 - Phones distract students from learning.\n"
        "3 - Many students use phones to cheat during exams.\n"
        "4 - Smartphones can also be useful educational tools.\n\n"
        "Target component: 1\n"
        + ("Support: 2, 3\nAttack: 4\nPartial-Attack: 0\n" if enable_partial_attack else "Support: 2, 3\nAttack: 4\n") +
        "Explanation: 2 and 3 each give independent reasons for the policy; 4 directly challenges the ban.\n\n"
        "**Example 3**\n"
        "Text:\n"
        "A recent RCT showed vitamin D supplementation reduces depressive symptoms. "
        "The RCT had a 12-month follow-up confirming sustained mood improvement. "
        "However, the study sample was small, limiting generalisability.\n\n"
        "Components:\n"
        "1 - A recent RCT showed vitamin D supplementation reduces depressive symptoms.\n"
        "2 - The RCT had a 12-month follow-up confirming sustained mood improvement.\n"
        "3 - However, the study sample was small, limiting generalisability.\n\n"
        "Target component: 1\n"
        + ("Support: 2\nAttack: 0\nPartial-Attack: 3\n" if enable_partial_attack else "Support: 2\nAttack: 3\n") +
        "Explanation: 2 bolsters 1; 3 constrains its generalisability "
        + ("(partial-attack, not full negation).\n\n" if enable_partial_attack else "(attack).\n\n") +
        "**Example 4**\n"
        "Text:\n"
        "Improving public transport can reduce car usage. Better public transport means shorter commute times. "
        "Shorter commutes lead to more productive workers.\n\n"
        "Components:\n"
        "1 - Improving public transport can reduce car usage.\n"
        "2 - Better public transport means shorter commute times.\n"
        "3 - Shorter commutes lead to more productive workers.\n\n"
        "Target component: 1\n"
        + ("Support: 0\nAttack: 0\nPartial-Attack: 0\n" if enable_partial_attack else "Support: 0\nAttack: 0\n") +
        "Explanation: Neither 2 nor 3 directly justifies or challenges 1.\n"
        + extra_example +
        "---\n"
        "Now apply the same logic to the new case:\n\n"
        "Text:\n"
        "REPLACE_TEXT\n\n"
        "Components:\n"
        "REPLACE_COMPONENTS\n\n"
        "Target component: REPLACE_NUM – \"REPLACE_TARGET\"\n\n"
        + closing_reminder + "\n"
    )

    instruction = instruction.replace("REPLACE_TEXT", text)
    instruction = instruction.replace("REPLACE_COMPONENTS", arg_components)
    instruction = instruction.replace("REPLACE_NUM", str(conclusion_number))
    instruction = instruction.replace("REPLACE_TARGET", dict_components[conclusion_number])
    return instruction


def missing_premise_support(
    premise: int,
    text: str,
    arg_components: str,
    dict_components: Dict[int, str],
) -> str:
    """Generate prompt to find which components are supported by a given premise.

    Used in Step 4 when a premise was not reached during BFS.
    The question is reversed: "Given this premise, which component does it support?"
    """
    instruction = f'''You are given a short argumentative text and a list of numbered components.
One of these components (the "premise") was not assigned during the main relation-extraction pass.
Your task is to determine **which other component(s) it directly supports**.

A direct support means the premise gives a reason or justification for the target
without relying on any intermediate component.

### Important guidance
- In a well-formed argument every component relates to at least one other.
  Try to find the best match. Return 0 only if the premise truly does not
  justify any other component.
- A component cannot support itself.

### Examples

Example 1
Text:
Raising the minimum wage improves workers' quality of life. Higher income allows people to afford better housing. Better housing conditions can improve mental health.

Components:
1 - Raising the minimum wage improves workers' quality of life.
2 - Higher income allows people to afford better housing.
3 - Better housing conditions can improve mental health.

Premise component: 2
Answer: 1
Explanation: 2 supports 1 directly. 3 supports 2, not 1.

Example 2
Text:
Schools should ban phones during class. Phones distract students from learning. Many students use phones to cheat during exams.

Components:
1 - Schools should ban phones during class.
2 - Phones distract students from learning.
3 - Many students use phones to cheat during exams.

Premise component: 3
Answer: 1
Explanation: 3 directly supports the policy in 1.

Example 3
Text:
Improving public transport can reduce car usage. Better public transport means shorter commute times. Shorter commutes lead to more productive workers.

Components:
1 - Improving public transport can reduce car usage.
2 - Better public transport means shorter commute times.
3 - Shorter commutes lead to more productive workers.

Premise component: 3
Answer: 2
Explanation: 3 directly supports 2 — shorter commutes lead to productivity, which elaborates on the benefit of shorter commute times.

Now, apply this logic to the following case:
Text:
{text}

Components:
{arg_components}

Your task:
- Consider all components except the premise itself.
- For each, ask: "Is this component directly supported by the premise?"
- Include only those that are directly supported.
- Return 0 only if the premise truly does not support any component.

Output Format:
Answer: <numbers>
(e.g., Answer: 2 or Answer: 0)

Premise component: {premise} - "{dict_components[premise]}"
Answer:
'''
    return instruction


def missing_premise_attack(
    premise: int,
    text: str,
    arg_components: str,
    dict_components: Dict[int, str],
) -> str:
    """Generate prompt to find which components are attacked by a given premise.

    Used in Step 4 when a premise was not reached during BFS.
    """
    instruction = f'''You are given a short argumentative text and a list of numbered components.
One of these components (the "premise") was not assigned during the main relation-extraction pass.
Your task is to determine **which other component(s) it directly attacks**.

A direct attack occurs when the premise challenges, contradicts, or weakens another
component without relying on any intermediate component.

### Important guidance
- Attack relations are less common than support. Only return IDs that are
  genuinely contradicted or undermined by the premise.
- A component cannot attack itself.
- Return 0 if the premise does not attack any component.

### Examples

Example 1
Text:
The vaccine is effective against the virus. However, it may cause severe side effects. These side effects are rare and manageable with proper care.

Components:
1 - The vaccine is effective against the virus.
2 - However, it may cause severe side effects.
3 - These side effects are rare and manageable with proper care.

Premise component: 2
Answer: 1
Explanation: 2 directly undermines 1 by introducing a negative consequence. 3 mitigates 2 but doesn't directly address 1.

Example 2
Text:
Implementing a four-day work week will reduce stress. But this could also decrease overall productivity. Lower productivity may hurt company profits.

Components:
1 - Implementing a four-day work week will reduce stress.
2 - But this could also decrease overall productivity.
3 - Lower productivity may hurt company profits.

Premise component: 2
Answer: 1
Explanation: 2 directly challenges the value of 1 by suggesting a drawback. 3 elaborates on 2, so it's not a direct attack on 1.

Example 3
Text:
The environmental tax will help reduce emissions. Some argue that the tax is too low to make a real difference. In contrast, others believe it will still shift consumer behavior.

Components:
1 - The environmental tax will help reduce emissions.
2 - Some argue that the tax is too low to make a real difference.
3 - In contrast, others believe it will still shift consumer behavior.

Premise component: 2
Answer: 1
Explanation: 2 questions the effectiveness of the tax, directly challenging 1. 3 counters 2, not 1.

Example 4
Text:
Extending the school day improves student performance. Longer school hours can lead to burnout and reduced motivation. Reduced motivation negatively impacts learning outcomes.

Components:
1 - Extending the school day improves student performance.
2 - Longer school hours can lead to burnout and reduced motivation.
3 - Reduced motivation negatively impacts learning outcomes.

Premise component: 2
Answer: 1
Explanation: 2 directly undermines 1 by suggesting a consequence that challenges its benefit. 3 supports 2 but does not directly address 1.

Now, apply this logic to the following case:
Text:
{text}

Components:
{arg_components}

Your task:
- Consider all components except the premise itself.
- For each, ask: "Is this component directly attacked by the premise?"
- Include only those that are directly attacked.
- Return 0 if the premise does not attack any component.

Output Format:
Answer: <numbers>
(e.g., Answer: 2 or Answer: 0)

Premise component: {premise} - "{dict_components[premise]}"
Answer:
'''
    return instruction



def missing_premise_partial_attack(
    premise: int,
    text: str,
    arg_components: str,
    dict_components: Dict[int, str],
) -> str:
    """Generate prompt to find which components are partially attacked by a given premise.

    Used in Step 4 (unvisited premises) for datasets that annotate partial-attack
    (e.g. AbstRCT). A partial-attack weakens or constrains the target without fully
    negating it — typically a caveat about study scope, significance, or generalisability.
    """
    instruction = f'''You are given a short argumentative text and a list of numbered components.
One component (the "premise") was not assigned during the main relation-extraction pass.
Your task is to determine **which other component(s) it partially attacks**.

A **partial-attack** occurs when the premise does NOT fully contradict the target, but
*constrains* or *weakens* it by:
  - Questioning the scope, significance, or generalisability of the target claim.
  - Introducing a caveat that limits how strongly the target can be asserted.
  - Calling for further confirmation without outright rejecting the claim.

Distinguish from a full attack (direct contradiction or negation) — use this prompt only
when the relationship is one of *weakening*, not *refuting*.

### Important guidance
- Partial-attack is common in medical abstracts as implicit statements about study limitations.
- Return 0 if the premise does not partially attack any component.
- A component cannot partially attack itself.

### Examples

Example 1
Text:
SLN biopsy is an effective and well-tolerated procedure. However, its safety should be confirmed by the results of larger randomized trials and meta-analyses.

Components:
1 - SLN biopsy is an effective and well-tolerated procedure.
2 - However, its safety should be confirmed by the results of larger randomized trials and meta-analyses.

Premise component: 2
Answer: 1
Explanation: 2 does not negate 1 outright — it accepts the finding but constrains its certainty. That is a partial-attack.

Example 2
Text:
The treatment significantly reduced pain scores at 6 months. However, the study lacked a placebo control group.

Components:
1 - The treatment significantly reduced pain scores at 6 months.
2 - However, the study lacked a placebo control group.

Premise component: 2
Answer: 1
Explanation: 2 does not say the treatment does not work; it weakens the confidence in the result by pointing to a methodological gap.

Example 3
Text:
Drug A lowered blood pressure in the majority of patients. Drug A caused mild side effects in 10% of patients. Mild side effects are generally acceptable in clinical practice.

Components:
1 - Drug A lowered blood pressure in the majority of patients.
2 - Drug A caused mild side effects in 10% of patients.
3 - Mild side effects are generally acceptable in clinical practice.

Premise component: 2
Answer: 0
Explanation: 2 notes a side effect but does not weaken or constrain 1; 3 mitigates 2. Neither is a partial-attack relationship originating from 2.

Now, apply this logic to the following case:
Text:
{text}

Components:
{arg_components}

Your task:
- Consider all components except the premise itself.
- For each, ask: "Does the premise weaken or constrain this component without fully negating it?"
- Return 0 if no partial-attack relationship exists.

Output Format:
Answer: <numbers>
(e.g., Answer: 1 or Answer: 0)

Premise component: {premise} - "{dict_components[premise]}"
Answer:
'''
    return instruction


def merge_components_cycle(
    text: str,
    arg_components: str,
    dict_components: Dict[int, str],
    component_ids: List[int],
) -> str:
    """Generate prompt to merge cycled components into a single component."""
    components_to_merge = "\n".join(
        f"{i} - {dict_components[i]}" for i in component_ids
    )

    prompt = f'''You are given a **text** and a list of **argumentative components** extracted from it.

Each component is numbered like this:
<number> - <component text>

Your task is to **merge the components with numbers {component_ids}** into a single, clear sentence that keeps the main ideas of both.

The merged sentence should:
- Combine the key points from both components,
- Be assertive and self-contained,
- Be suitable for use in an argument graph.

**Text:**
{text}

**All Argumentative Components:**
{arg_components}

**Components to merge:**
{components_to_merge}

**What is the merged version of components {component_ids}?**
Return only the new merged component text. Do not include numbering or any explanation.
'''
    return prompt
