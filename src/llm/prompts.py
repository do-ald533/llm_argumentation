"""
Prompt templates for LLM-based argumentation structuring tasks.

This module contains all prompt generation functions used throughout the pipeline.
Each function returns a formatted prompt string for specific argumentation analysis tasks.

Organization:
- Task 1: Component Identification
- Task 2: Component Correction & Decomposition  
- Task 3: Conclusion Identification
- Task 4: Premise-Conclusion Relations
- Task 5: Missing Premise Detection
- Task 6: Convergent Premises & Implicit Premises
- Task 7: Counterargument Analysis
- Task 8: Premise Evaluation
"""


# =============================================================================
# TASK 1: COMPONENT IDENTIFICATION
# =============================================================================

def argumentative_components(text):
    """
    Generate prompt to identify argumentative components in text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Formatted prompt string
    """
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
# TASK 2: COMPONENT CORRECTION & DECOMPOSITION
# =============================================================================

def components_corrected(argument: str) -> str:
    """
    Generate prompt to decompose complex components into simpler ones.
    
    Args:
        argument: Single sentence to potentially split
        
    Returns:
        Formatted prompt string
    """
    prompt = f'''
        You will receive **one sentence** at a time.  
        Your job is to decide whether it contains **more than one independent argumentative component** (premises or conclusions that could each stand alone as a full sentence).

        ──────────────────────────────────────────────
        WHEN TO SPLIT
        ──────────────────────────────────────────────
        Split the sentence into numbered components **only if BOTH** are true:

        1. A connector such as **because, so, but, and, unless, if, except** links two clauses, **and**  
        2. Each linked clause can still be read as a sensible, self-contained claim after the connector is removed.

        Typical patterns  
        • A conclusion followed by its reason.  
        • Two or more coordinated premises.  
        • A conclusion plus an exception introduced by *unless* or *except*.

        ──────────────────────────────────────────────
        DON'T SPLIT WHEN…
        ──────────────────────────────────────────────
        The connector forms part of a single, embedded claim—for example:

        * Necessary-condition phrases like **"only if"**, **"provided that"**, **"as long as"**  
        * Relative clauses, temporal modifiers, infinitive phrases, etc., that do not read as standalone statements.

        Heuristic: after deleting the connector, if either fragment **cannot** be read as a complete English sentence conveying its own argument, treat the whole thing as **one component** and output **0**.

        ──────────────────────────────────────────────
        EXAMPLES
        ──────────────────────────────────────────────
        Example 1  
        Text: The boy should be arrested, as he was undoubtedly the perpetrator of the crime.  
        Output:  
        1 - The boy should be arrested.  
        2 - He was undoubtedly the perpetrator of the crime.

        Example 2  
        Text: Capital punishment, in many cases, deters people from committing murder, and reducing the murder rate is, in itself, worthwhile.  
        Output:  
        1 - Capital punishment, in many cases, deters people from committing murder.  
        2 - Reducing the murder rate is, in itself, worthwhile.

        Example 3  
        Text: A healthy diet improves your mood because it affects your hormone levels and helps you maintain a stable energy level.  
        Output:  
        1 - A healthy diet improves your mood.  
        2 - It affects your hormone levels.  
        3 - It helps you maintain a stable energy level.

        Example 4  
        Text: Solar power is the best solution for rural areas, unless battery technology fails to improve.  
        Output:  
        1 - Solar power is the best solution for rural areas.  
        2 - Unless battery technology fails to improve.

        ❌ **No split - necessary-condition clause**  
        Example 5  
        Text: The warranty applies **only if** the product is registered within 30 days.  
        Output:  
        0

        ──────────────────────────────────────────────
        FORMAT YOUR ANSWER
        ──────────────────────────────────────────────
        * If you split: write each component on its own line using **NUMBER - component**, starting at 1.  
        * If the sentence remains one argument, output **0** (and nothing else).

        Now analyse the following sentence: {argument}
'''
    return prompt


def rewrite_sentence(text, arg_components):
    """
    Generate prompt to summarize/clarify argumentative components.
    
    Args:
        text: Original text
        arg_components: List of numbered components
        
    Returns:
        Formatted prompt string
    """
    prompt = f'''
        # Prompt: Summarize Argumentative Components for Clarity

        You are given a **text** and a **list of argumentative components** extracted from it. Each component is numbered and formatted as:  
        `<number> - <text>`

        These components represent parts of an **argumentation graph**, such as **claims**, **premises**, or **counterarguments**. Your task is to **summarize each component only if doing so makes it easier to understand within the overall argument structure**.

        ## When to Summarize

        You should rewrite a component in a shorter form if:

        - It is **too long, detailed, or complex**, making the main idea harder to see.  
        - The key argumentative content can be **preserved with fewer words**.  
        - A more concise version would make the argumentation text **easier to interpret**.

        ## When NOT to Summarize

        - If the component is already **short, clear, and self-contained**, leave it unchanged.  
        - Do **not** remove essential content or alter the meaning.  

        ## Assertion Requirement

        - Each component must be rewritten as a **declarative sentence**—a statement that conveys information, judgment, recommendation, or instruction.

        - Acceptable forms include:

            - Claims or statements of fact (e.g., "Climate change affects global agriculture.")

            - Evaluations or judgments (e.g., "This policy is ineffective.")

            - Suggestions (e.g., "Governments should regulate social media.")

            - Recommendations or imperatives (e.g., "You must wear a helmet while riding.")

        - If a component is phrased as a question, rewrite it as a declarative sentence that reflects its intended argumentative purpose.

            ❌ Should governments regulate AI?

            ✅ Governments should regulate AI. or Regulating AI is necessary.

        ## Format Requirements

        - **Keep the original numbering.**  
        - Return each component in the format: `<number> - <rewritten or original text>`.  
        - Be as **brief and precise** as possible, while preserving the original meaning and argumentative role.

        ## Examples

        ### Long → Summarized

        **Input:**  
        `1 - Over the past few decades, the fast fashion industry has contributed significantly to environmental degradation due to mass production, water waste, and increased textile waste.`  
        **Output:**  
        `1 - Fast fashion contributes significantly to environmental harm.`

        **Input:**  
        `2 - According to several recent studies, constant exposure to highly filtered and personalized online content can reduce users' awareness of alternative perspectives and critical thinking.`  
        **Output:**  
        `2 - Personalized content online can reduce awareness of other perspectives.`

        ### Already Clear → Unchanged

        **Input:**  
        `3 - Social media can reinforce existing biases.`  
        **Output:**  
        `3 - Social media can reinforce existing biases.`

        ## Task

        **Text:**  
        {text}

        **Argumentative Components:**  
        {arg_components}

        **Return only the revised list of components, keeping their numbering.**
    '''
    return prompt


def merge_components(text, arg_components):
    """
    Generate prompt to identify components that should be merged.
    
    Args:
        text: Original text
        arg_components: List of numbered components
        
    Returns:
        Formatted prompt string
    """
    prompt = f'''
        You are given a **text** and a **list of argumentative components** extracted from it. Each component is numbered and formatted as:  
        `<number> - <text>`

        These components represent parts of an **argumentation structure**, such as **claims**, **premises**, or **conclusions**.

        Your task is to identify components that can be **merged**. Components should be merged if they:

        ---

        ## When to Merge

        Merge components if:

        - They **express the same idea using different words** (semantic paraphrases).  
        - They are **logically connected** and part of the **same reasoning unit**, such as:
            - A conditional and its antecedent/consequent (e.g., "If p, then q" and "p").  
            - Two pieces of evidence or steps supporting the **same claim**.  
            - A general assertion and its rephrased or elaborated form.

        Merging helps reduce redundancy and improve clarity.

        ---

        ## When NOT to Merge

        - Do **not merge** components that express **different, unrelated, or opposing ideas**.  
        - Do **not merge** components that are **only topically related** without contributing to the same reasoning.  
        - Do **not infer or invent** unstated content. Stick strictly to what is present.

        ---

        ## Output Format

        - If two or more components should be merged, return a **single line** in the format:  
        `<number1, number2, ...> - <merged text>`  
        (Use the **original component numbers** in ascending order.)  
        - Do **not return anything** for unmerged components. They are to be handled later.  
        - If **no components** should be merged, return only:  
        `0`  
        - Return **only** the merged results, one per line.

        ---

        ## Examples

        ### ✅ Merge

        **Input:**  
        1 - Violent video games cause aggression in teens.  
        2 - Teens who play violent games are more aggressive.  

        **Output:**  
        1, 2 - Violent video games increase aggression in teenagers.

        ### ❌ No Merge

        **Input:**  
        1 - Social media improves social connectivity.  
        2 - Social media worsens mental health.

        **Output:**  
        0

        ---

        ## Task

        **Text:**  
        {text}

        **Argumentative Components:**  
        {arg_components}

        **Your output should only include merged components**, using the following format:  
        `<number1, number2, ...> - <merged text>`  
        or `0` if no components should be merged.
    '''
    return prompt


def merge_components_cycle(text, arg_components, dict_components, component_ids):
    """
    Generate prompt to merge specific components (used for cycle resolution).
    
    Args:
        text: Original text
        arg_components: List of all numbered components
        dict_components: Dictionary mapping component IDs to text
        component_ids: List of component IDs to merge
        
    Returns:
        Formatted prompt string
    """
    components_to_merge = "\n".join(f"{i} - {dict_components[i]}" for i in component_ids)

    prompt = f'''
        You are given a **text** and a list of **argumentative components** extracted from it.

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


# =============================================================================
# TASK 3: CONCLUSION IDENTIFICATION
# =============================================================================

def argumentative_conclusion(text, arguments):
    """
    Generate prompt to identify the main conclusion of an argument.
    
    Args:
        text: Original text
        arguments: List of numbered components
        
    Returns:
        Formatted prompt string
    """
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

Additional Instrucions:
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


# =============================================================================
# TASK 4: PREMISE-CONCLUSION RELATIONS
# =============================================================================

def premise_identification(conclusion, text, arg_components, dict_components, links):
    """
    Generate prompt to identify direct premise relations (legacy format).
    
    Args:
        conclusion: Conclusion component ID
        text: Original text
        arg_components: List of numbered components
        dict_components: Dictionary mapping IDs to text
        links: Existing identified links
        
    Returns:
        Formatted prompt string
    """
    instruction = f'''Identify the premises that directly support or attack an argumentative component. By direct link, I understand that the support or attack is not mediated by other premises.

Consider the following example:

Text: New students on the course tend to be more dedicated to their tasks. Devoting more time to assignments helps when it comes to solving exam exercises. Students who do well in exams get better jobs.

Argumentative components:
1 - Students new to the course are usually more dedicated to their tasks;
2 - Being more dedicated to the tasks helps when it comes to solving the exercises in the exam;
3 - Students who do well in exams get better jobs.

Links: 2 > 3 (i.e., component 2 directly supports component 3).

Important: Component 2 ("Being more dedicated to the tasks helps when it comes to solving the exercises in the exam") directly supports component 3 ("Students who do well in exams get better jobs"). Component 1 is also part of the supporting set of 3, but only indirectly (i.e., 1 directly supports 2, which in turn supports 3).

Now, consider the following argument:

Text: {text}

Argumentative components:

{arg_components}
'''
    if links:
        instruction += f'''

    The following links have already been identified:

    {links}
    '''

    instruction += f'''           
Now identify the premises that support or attack the argumentative component {conclusion}: "{dict_components[conclusion]}" 

Use the following formalization:
- '>' to indicate a support relation; 
- '~' to indicate an attack relation;
- '+' to indicate that several premises act together;

Example: '5 + 9 + 2 > 10' means that components 5, 9, and 2 together support component 10.

If no premise supports or attacks the conclusion, answer with number '0'.

''' 
    return instruction


def premise_support(conclusion_number, text, arg_components, dict_components):
    """
    Generate prompt to identify premises that directly support a conclusion.
    
    Args:
        conclusion_number: Target component ID
        text: Original text
        arg_components: List of numbered components
        dict_components: Dictionary mapping IDs to text
        
    Returns:
        Formatted prompt string
    """
    instruction = f"""
        You will be given a short argumentative text and its numbered components.  Your job is to decide
        **which components directly support** a specified target component.

        ---
        ### What counts as *direct support*?
        A component gives *direct support* when it supplies a reason, justification, evidence,
        example, or explanation **for the target claim itself**, *without first depending on any
        other component*.

        A component is **not** a direct supporter if it:
        * Supports another component that then supports the target (indirect support).
        * Undermines, contradicts, or attacks the target.

        ---
        ### Output format (strict)
        Return exactly one line:
        * Answer: 0                ← when *no* component directly supports the target, or
        * Answer: n                ← one supporting component, or
        * Answer: n, m, ...        ← several supporters (ascending order, comma-separated).
        No extra text.

        ---
        ### Worked examples
        (Study them carefully; follow the same reasoning.)

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

        **Example 4** (component 4 directly supports 3)
        Text:
        Vitamin D deficiency is common in winter. Supplementing with vitamin D increases serum levels. A recent RCT showed vitamin D supplementation reduces depressive symptoms. The RCT had a 12-month follow-up confirming sustained mood improvement.

        Components:
        1 - Vitamin D deficiency is common in winter.
        2 - Supplementing with vitamin D increases serum levels.
        3 - A recent RCT showed vitamin D supplementation reduces depressive symptoms.
        4 - The RCT had a 12-month follow-p confirming sustained mood improvement.

        Target component: 3
        Answer: 4
        Explanation: 4 bolsters 3 by adding further evidence from the *same* study, so it directly strengthens 3.

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
        Explanation:
        Component 1 explains a direct benefit of drinking water (improved focus), and 2 explains the downside of not drinking water (lower cognition), both of which justify the policy in 4.
        Component 3 is merely a contextual observation—not a justification—so it's not a direct supporter.
        ---
        Now apply the same logic to the new case:

        Text:
        {text}

        Components:
        {arg_components}

        Identify the components that **directly support** component {conclusion_number}: "{dict_components[conclusion_number]}".

        Remember: examine every component except the target, decide whether it supplies an *independent* justification, and output your result in the strict format.

        Answer:
"""
    return instruction


def premise_attack(conclusion_number, text, arg_components, dict_components):
    """
    Generate prompt to identify premises that directly attack a conclusion.
    
    Args:
        conclusion_number: Target component ID
        text: Original text
        arg_components: List of numbered components
        dict_components: Dictionary mapping IDs to text
        
    Returns:
        Formatted prompt string
    """
    instruction = f"""
        You will be given a list of argumentative components. Your task is to identify *which* components
        **directly attack** a given target component.

        **What counts as a direct attack?**  A component *challenges, contradicts, limits, or undermines* the
        truth, certainty, relevance, or persuasive force of the target claim — even if that attack is phrased
        cautiously (e.g. "however", "may not", "could lead to negative outcomes").  Typical patterns include:

        * Explicit disagreement with the claim.
        * Pointing out exceptions or counter-examples.
        * Questioning feasibility, effectiveness, or sufficiency.
        * Introducing conditions that would invalidate the claim.

        A component does *not* count as an attack if it:
        * Merely elaborates, supports, or restates the target.
        * Attacks another component but relies on an intermediate step (indirect).

        ---
        ### Output format (strict)
        Reply **only** with one of the following (no extra spaces):
        * `Answer: 0`                ← when nothing attacks the target, or
        * `Answer: n`                ← single attacking component, or
        * `Answer: n, m, ...`        ← several attackers, listed in ascending order.

        ---
        ### Worked examples
        (Study these carefully; imitate the logic.)

        **Example 1**
        Text:
        Working from home boosts productivity. However, it can also blur work-life boundaries. Blurred boundaries may lead to burnout.

        Components:
        1 - Working from home boosts productivity.
        2 - However, it can also blur work-life boundaries.
        3 - Blurred boundaries may lead to burnout.

        Target component: 1
        Answer: 2
        Explanation: Component 2 introduces a drawback that undermines the benefit claimed in 1. Component 3 elaborates on 2, so it is *not* a direct attack on 1.

        **Example 2**
        Text:
        Implementing a four-day work week will reduce stress. But this could also decrease overall productivity. Lower productivity may hurt company profits.

        Components:
        1 - Implementing a four-day work week will reduce stress.
        2 - But this could also decrease overall productivity.
        3 - Lower productivity may hurt company profits.

        Target component: 1
        Answer: 2
        Explanation: 2 challenges the value of 1 by highlighting a potential negative consequence. 3 depends on 2 and therefore does not directly address 1.

        **Example 3**
        Text:
        The environmental tax will help reduce emissions. Some argue that the tax is too low to make a real difference. In contrast, others believe it will still shift consumer behavior.

        Components:
        1 - The environmental tax will help reduce emissions.
        2 - Some argue that the tax is too low to make a real difference.
        3 - In contrast, others believe it will still shift consumer behavior.

        Target component: 1
        Answer: 2
        Explanation: 2 asserts the tax is insufficient, directly disputing 1. Component 3 takes issue with 2, so it is not a direct attack on 1.

        **Example 4**
        Text:
        Extending the school day improves student performance. Longer school hours can lead to burnout and reduced motivation. Reduced motivation negatively impacts learning outcomes.

        Components:
        1 - Extending the school day improves student performance.
        2 - Longer school hours can lead to burnout and reduced motivation.
        3 - Reduced motivation negatively impacts learning outcomes.

        Target component: 1
        Answer: 2
        Explanation: 2 presents a negative effect that contradicts the claimed benefit in 1. Component 3 supports 2, so it does not directly attack 1.

        **Example 5**  (component 4 directly attacks 3)
        Text:
        A high-protein diet improves muscle growth. Some studies claim high-protein diets strain kidneys. A comprehensive meta-analysis found no link between high protein intake and kidney issues. However, the meta-analysis included mostly short-term studies, making its conclusions unreliable.

        Components:
        1 - A high-protein diet improves muscle growth.
        2 - Some studies claim high-protein diets strain kidneys.
        3 - A comprehensive meta-analysis found no link between high protein intake and kidney issues.
        4 - However, the meta-analysis included mostly short-term studies, making its conclusions unreliable.

        Target component: 3
        Answer: 4
        Explanation: 4 questions the validity of 3 by attacking the methodology of the cited meta-analysis, thereby undermining its conclusion.

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


# =============================================================================
# TASK 5: MISSING PREMISE DETECTION
# =============================================================================

def missing_premise_support(premise, text, arg_components, dict_components):
    """
    Generate prompt to find which components are supported by a given premise.
    
    Args:
        premise: Premise component ID
        text: Original text
        arg_components: List of numbered components
        dict_components: Dictionary mapping IDs to text
        
    Returns:
        Formatted prompt string
    """
    instruction = f'''
        You are given a short argumentative text and a list of numbered components (each expressing a claim). Your task is to identify which components are directly supported by a given premise.

        A direct support occurs when the premise gives a reason or justification for the conclusion without relying on any other component.

        Important Rules:
        - Only consider direct support (no attacks, no indirect relations).
        - A component cannot support itself.
        - For each other component, ask: "Does this component receive direct support from the premise, without needing other components to bridge the reasoning?"

        Example 1
        Text:
        Raising the minimum wage improves workers' quality of life. Higher income allows people to afford better housing. Better housing conditions can improve mental health.

        Components:
        1 - Raising the minimum wage improves workers' quality of life.
        2 - Higher income allows people to afford better housing.
        3 - Better housing conditions can improve mental health.

        Premise component: 2
        Answer: 1
        Explanation: Component 2 supports 1 directly. Component 3 supports 2, not 1.

        Example 2
        Text:
        Schools should ban phones during class. Phones distract students from learning. Many students use phones to cheat during exams.

        Components:
        1 - Schools should ban phones during class.
        2 - Phones distract students from learning.
        3 - Many students use phones to cheat during exams.

        Premise component: 3
        Answer: 1
        Explanation: Component 3 directly supports the policy in 1.

        Example 3
        Text:
        Improving public transport can reduce car usage. Better public transport means shorter commute times. Shorter commutes lead to more productive workers.

        Components:
        1 - Improving public transport can reduce car usage.
        2 - Better public transport means shorter commute times.
        3 - Shorter commutes lead to more productive workers.

        Premise component: 3
        Answer: 0
        Explanation: Component 3 supports 2, not 1. It doesn't directly support any other component in this case.

        Now, apply this logic to the following case:
        Text:
        {text}

        Components:
        {arg_components}

        Your task:
        - Consider all components except the premise.
        - For each, ask: "Is this component directly supported by the premise?"
        - Include only those that are directly supported.
        - If the premise supports no component directly, return 0.

        Output Format:
        Answer: <numbers>
        (e.g., Answer: 2 or Answer: 0)

        Premise component: {premise} - "{dict_components[premise]}"
        Answer:
'''
    return instruction


def missing_premise_attack(premise, text, arg_components, dict_components):
    """
    Generate prompt to find which components are attacked by a given premise.
    
    Args:
        premise: Premise component ID
        text: Original text
        arg_components: List of numbered components
        dict_components: Dictionary mapping IDs to text
        
    Returns:
        Formatted prompt string
    """
    instruction = f'''
        You are given a short argumentative text and a list of numbered components (each expressing a claim). Your task is to identify which components are directly attacked by a given premise.

        A direct attack occurs when the premise challenges, contradicts, or weakens another component without relying on any other component.

        Important Rules:
        - Only consider direct attacks (no support, no indirect attacks).
        - A component cannot attack itself.
        - For each other component, ask: "Is this component directly challenged or contradicted by the premise, without relying on other components?"

        Example 1
        Text:
        The vaccine is effective against the virus. However, it may cause severe side effects. These side effects are rare and manageable with proper care.

        Components:
        1 - The vaccine is effective against the virus.
        2 - However, it may cause severe side effects.
        3 - These side effects are rare and manageable with proper care.

        Target component: 1
        Answer: 2
        Explanation: Component 2 directly undermines component 1 by introducing a negative consequence. Component 3 mitigates 2 but doesn't directly address 1.

        Example 2
        Text:
        Implementing a four-day work week will reduce stress. But this could also decrease overall productivity. Lower productivity may hurt company profits.

        Components:
        1 - Implementing a four-day work week will reduce stress.
        2 - But this could also decrease overall productivity.
        3 - Lower productivity may hurt company profits.

        Target component: 1
        Answer: 2
        Explanation: Component 2 directly challenges the value of component 1 by suggesting a drawback. Component 3 elaborates on 2, so it's not a direct attack on 1.

        Example 3
        Text:
        The environmental tax will help reduce emissions. Some argue that the tax is too low to make a real difference. In contrast, others believe it will still shift consumer behavior.

        Components:
        1 - The environmental tax will help reduce emissions.
        2 - Some argue that the tax is too low to make a real difference.
        3 - In contrast, others believe it will still shift consumer behavior.

        Example 4
        Text:
        Extending the school day improves student performance. Longer school hours can lead to burnout and reduced motivation. Reduced motivation negatively impacts learning outcomes.

        Components:
        1 - Extending the school day improves student performance.
        2 - Longer school hours can lead to burnout and reduced motivation.
        3 - Reduced motivation negatively impacts learning outcomes.

        Target component: 1
        Answer: 2
        Explanation: Component 2 directly undermines component 1 by suggesting a consequence that challenges its benefit. Although it's not phrased as a strong disagreement, it logically contradicts the idea that extended hours improve performance. Component 3 supports 2, but does not directly address 1.

        Now, apply this logic to the following case:
        Text:
        {text}

        Components:
        {arg_components}

        Your task:
        - Consider all components except the premise.
        - For each, ask: "Is this component directly attacked by the premise?"
        - Include only those that are directly attacked.
        - If the premise does not attack any component directly, return 0.

        Output Format:
        Answer: <numbers>
        (e.g., Answer: 2 or Answer: 0)

        Premise component: {premise} - "{dict_components[premise]}"
        Answer:
'''
    return instruction


def missing_premise_conclusion(conclusion, text, arg_components, dict_components):
    """
    Generate prompt to find premises that support a conclusion (alternative format).
    
    Args:
        conclusion: Conclusion component ID
        text: Original text
        arg_components: List of numbered components
        dict_components: Dictionary mapping IDs to text
        
    Returns:
        Formatted prompt string
    """
    instruction = f'''
        You are given a short argumentative text and a list of numbered components (each expressing a claim). Your task is to identify which components directly support a given target component.

        A direct support occurs when a component gives a reason or justification for the target without depending on any other component.
        
        Important Rules:
        - No component can support itself.
        - You must consider each component one by one and ask: "Does this component give a direct justification for the target component, without relying on other components?"
        - Only include components that directly support the target.
        - Do not include components that:
            - Attack or contradict the target
            - Rely on other components to make their point
            - Are unrelated
            - Are the target itself
        - If no component directly supports the target, return 0.


        Example 1
        Text:
        Raising the minimum wage improves workers' quality of life. Higher income allows people to afford better housing. Better housing conditions can improve mental health.

        Components:
        1 - Raising the minimum wage improves workers' quality of life.
        2 - Higher income allows people to afford better housing.
        3 - Better housing conditions can improve mental health.

        Target component: 1
        Answer: 2
        Explanation: Component 2 directly supports 1. Component 3 supports 2, not 1.

        Example 2
        Text:
        Schools should ban phones during class. Phones distract students from learning. Many students use phones to cheat during exams.

        Components:
        1 - Schools should ban phones during class.
        2 - Phones distract students from learning.
        3 - Many students use phones to cheat during exams.

        Target component: 1
        Answer: 2, 3
        Explanation: Both 2 and 3 give independent reasons supporting 1.

        Example 3
        Text:
        Improving public transport can reduce car usage. Better public transport means shorter commute times. Shorter commutes lead to more productive workers.

        Components:
        1 - Improving public transport can reduce car usage.
        2 - Better public transport means shorter commute times.
        3 - Shorter commutes lead to more productive workers.

        Target component: 1
        Answer: 0
        Explanation: Neither 2 nor 3 directly support the idea that public transport reduces car usage. They describe other benefits.


        Now, apply this logic to the following case:
        Text:
        {text}

        Components:
        {arg_components}

        Your task:
        - Consider each component except the target.
        - For each one, ask: "Does this component directly support the target by offering a justification that doesn't depend on other components?"
        - Include only the components that meet this condition.
        - If no component directly supports the target, return 0.

        Output Format:
        Answer: <numbers>
        (e.g., Answer: 2 or Answer: 3, 4 or Answer: 0)

        Premise component: {conclusion} - "{dict_components[conclusion]}"
        Answer:
'''
    return instruction


# =============================================================================
# TASK 6: CONVERGENT PREMISES & IMPLICIT PREMISES
# =============================================================================

def convergent_premises_support(conclusion, text, arg_components, dict_components, premises):
    """
    Generate prompt to identify convergent premises that jointly support a conclusion.
    
    Args:
        conclusion: Conclusion component ID
        text: Original text
        arg_components: List of numbered components
        dict_components: Dictionary mapping IDs to text
        premises: List of premise component IDs
        
    Returns:
        Formatted prompt string
    """
    instruction = f'''
    You are given a short argumentative text and a list of numbered components (each expressing a claim). Your task is to identify which premises work together in a **convergent way to support** a given conclusion.

    A *convergent support* occurs when multiple premises combine to justify the conclusion. This means that none of the premises alone is strong enough to support the conclusion, but together they form a compelling argument.

    Important Rules:
    - Focus **only** on convergent supports (not independent or separate supports).
    - A component cannot support itself.
    - For each premise, ask: "Does this premise help support the conclusion in a way that depends on the presence of other premises?"
    - Include *only* premises that are part of a convergent supporting structure.

    ### Examples

    **Example 1**    
    Text:
    The sky is full of dark clouds, and the wind has suddenly turned cold. Therefore, we should take an umbrella.

    Components:
    1. The sky is full of dark clouds.
    2. The wind has suddenly turned cold.
    3. Therefore, we should take an umbrella.

    Conclusion component: 3
    Answer: 1, 2
    Explanation: Components 1 and 2 together point to imminent rain; either one alone would be a weaker reason to carry an umbrella.

    **Example 2**
    Text:
    Adopting a dog would give me companionship, motivate me to exercise daily, and also provide a home for a shelter animal. Therefore, I should adopt a dog.

    Components:
    1. A dog would give me companionship.
    2. A dog would motivate me to exercise daily.
    3. A shelter dog needs a home.
    4. Therefore, I should adopt a dog.
    Conclusion component: 4
    Answer: 0
    Explanation: Each premise is an independent reason (companionship, health benefit, altruism). None depends on the others, so no convergent set exists.

    **Example 3**
    Text:
    Our town experiences average summer temperatures above 35 °C, and the nearest public swimming facility is 30 km away. In addition, the state government has offered a matching grant that will cover half the construction costs. Therefore, the town council should build a public swimming pool.

    Components:
    1. The town experiences average summer temperatures above 35 °C.
    2. The nearest public swimming facility is 30 km away.
    3. The state government has offered a matching grant that will cover half the construction costs.
    4. Therefore, the town council should build a public swimming pool.

    Conclusion component: 4
    Answer: 1, 2
    Explanation: High summer heat (1) plus the long distance to the nearest pool (2) jointly show a pressing need for a local facility (4); either premise alone would be far less persuasive. The state grant (3) is an independent financial reason, so it is not part of the convergent set.

    **Example 4**
    Text:
    The valley hosts the last remaining breeding ground of the golden marsh frog, a species whose population has dropped by 90 % in twenty years. Ecologists show that preserving at least 10,000 contiguous hectares is necessary for the frog's annual migration. Satellite imagery confirms that the proposed reserve is the only continuous area of that size left in the region. Hence, the government should designate the valley as a protected wildlife reserve.

    Components:
    1. The valley hosts the last remaining breeding ground of the golden marsh frog.
    2. Preserving ≥10,000 ha is necessary for the frog's migration.
    3. The proposed reserve is the only continuous area that large left in the region.
    4. Therefore, the government should designate the valley as a protected wildlife reserve.

    Conclusion component: 4 
    Answer: 1, 2, 3
    Explanation: Premises 1, 2, and 3 need one another: (1) identifies an endangered species, (2) states a habitat requirement, (3) shows the requirement can be met only by this site. Separately they do not justify protection, but combined they do.

    **Example 5**
    Text:
    City traffic congestion has increased by 25 % since last year. Exhaust from idling cars is the main source of local air pollution. A recent survey shows 68 % of residents would use light-rail if it were available. Therefore, the city council should fund the construction of a light-rail system.

    Components:
    1. Traffic congestion has increased by 25 % since last year.
    2. Exhaust from idling cars is the main source of local air pollution.
    3. 68 % of residents would use light-rail if it were available.
    4. Therefore, the city council should fund the construction of a light-rail system.

    Conclusion component: 4
    Answer: 0
    Explanation: Each premise offers a separate reason: congestion relief (1), environmental benefit (2), public demand (3). None depends on the others, so there is no convergent set.

    Now, apply this logic to the following case:
    Text:
    {text}

    Components:
    {arg_components}

    Your task:
    - Consider **only** the premises that support the given conclusion.
    - Identify the ones that work together to support it in a convergent way.
    - Exclude premises that support the conclusion independently.
    - If no convergent support exists, return 0.

    Output Format:
    Answer: <numbers>
    (e.g., Answer: 2, 3 or Answer: 0)

    Conclusion component: {conclusion} - "{dict_components[conclusion]}"
    Premises: {premises}
    Answer:
    '''
    return instruction


def convergent_premises_attack(conclusion, text, arg_components, dict_components, premises):
    """
    Generate prompt to identify convergent premises that jointly attack a conclusion.
    
    Args:
        conclusion: Conclusion component ID
        text: Original text
        arg_components: List of numbered components
        dict_components: Dictionary mapping IDs to text
        premises: List of premise component IDs
        
    Returns:
        Formatted prompt string
    """
    instruction = f'''
        You are given a short argumentative text and a list of numbered components (each expressing a claim).  
        Your task is to identify which premises work together in a **convergent way to attack** a given conclusion (or other designated claim).

        A *convergent attack* occurs when multiple premises combine to undermine the conclusion.  
        None of the premises alone is strong enough to refute the conclusion, but taken together they form a compelling counter-argument.

        Important Rules:
        - Focus **only** on convergent attacks (not independent or separate objections).
        - A component cannot attack itself.
        - For each premise ask: "Does this premise weaken the conclusion in a way that depends on the presence of other premises?"
        - Include *only* premises that are part of a convergent attacking structure.

        ### Examples

        **Example 1**

        Text:  
        The project's proponents claim the new dam will supply cheap electricity. Therefore, the government should approve its construction.

        Components:  
        1. The dam site sits on a major fault line classified as high-risk for earthquakes.  
        2. Recent seismic studies show the fault line has become more active over the past decade.  
        3. Therefore, the government should approve the dam's construction.

        Conclusion component: 3  

        Explanation: Premise 1 notes earthquake risk; premise 2 reinforces that risk with fresh data. Either alone gives only a tentative reason to hesitate, but together they present a strong safety objection that undermines the approval decision.

        **Example 2**

        Text:  
        The city council should privatize waste collection because it will reduce costs.

        Components:  
        1. A neighboring city privatized waste collection and saw costs rise by 15 %.  
        2. A recent audit shows the council's current waste service already operates at lower cost than private bids.  
        3. Therefore, the city council should privatize waste collection.

        Conclusion component: 3  
        
        Explanation: Premise 1 (cost increased elsewhere) and premise 2 (current service is cheaper) each independently undercut the cost-saving claim. They do not rely on one another, so there is no convergent attack set.

        **Example 3**

        Text:  
        Our town should host a large music festival next summer; it will boost tourism and generate revenue for local businesses.

        Components:  
        1. The town has only 500 hotel rooms, but last year's smaller festival attracted 8,000 visitors.  
        2. The nearest emergency hospital is 45 km away and would be overloaded by a sudden influx.  
        3. Hosting a large festival will boost tourism and revenue.  
        4. Therefore, the town should host a large music festival next summer.

        Conclusion component: 4  
        
        Explanation: Insufficient accommodation (1) plus inadequate emergency facilities (2) together create a compelling logistical and safety objection; neither premise alone fully demonstrates how unprepared the town is.

        **Example 4**

        Text:  
        We should legalize night hunting of wild boar to control their population.

        Components:  
        1. Thermal-imaging rifles used at night have a high rate of misidentifying targets.  
        2. Wildlife-agency data show that 18 % of night-hunting incidents injure protected species.  
        3. Local hospitals report a spike in accidental firearm injuries during the current night-hunting trial period.  
        4. Therefore, we should legalize night hunting of wild boar.

        Conclusion component: 4  

        Explanation: Premise 1 (identification errors), premise 2 (harm to protected species), and premise 3 (human injury data) interlock: together they indicate that night hunting is dangerous to both wildlife and people. Each premise alone signals a risk, but only in combination do they present a robust case against legalization.

        Now, apply this logic to the following case:

        Text:  
        {text}

        Components:  
        {arg_components}

        Your task:  
        - Consider **only** the premises that attack the given conclusion.  
        - Identify the ones that work together to undermine it in a convergent way.  
        - Exclude premises that attack the conclusion independently.  
        - If no convergent attack exists, return 0.

        Output Format:
        Answer: <numbers>
        (e.g., Answer: 2, 3 or Answer: 0)

        Conclusion component: {conclusion} - "{dict_components[conclusion]}"
        Premises: {premises}
        Answer:
    '''
    return instruction


def build_relevant_links(prem_ids, concl_id, relation_type):
    """
    Helper function to format relevant links text.
    
    Args:
        prem_ids: List of premise component IDs
        concl_id: Conclusion component ID
        relation_type: Either 'support' or 'attack'
        
    Returns:
        Formatted string describing the relation
    """
    if len(prem_ids) == 1:
        prem_text = f"{prem_ids[0]}"
    else:
        prem_text = ", ".join(map(str, prem_ids[:-1])) + " and " + str(prem_ids[-1])

    return f"{prem_text} {relation_type} {concl_id}"


def implicit_prompt_support(text, arg_components, prem_ids, concl_id):
    """
    Generate prompt to identify implicit premises in support relations.
    
    Args:
        text: Original text
        arg_components: List of numbered components
        prem_ids: List of premise component IDs
        concl_id: Conclusion component ID
        
    Returns:
        Formatted prompt string
    """
    implicit_premises = f'''
        An argument often contains implicit premises. These premises are assumed in the conversation without being explicitly stated. They are essential to the argument's validity, because they provide the necessary support for the conclusion to follow.

        Your task is that of recovering the implicit premises connected to explicit arguments. Do that only when, without them, the conclusion would not logically follow from the stated premise(s).
        
        ---
        **Example 1:**

        Text:
        These letters attributed to Nestor show a classical rhetorical style. We know that Nestor never attended a literary writing course. Therefore, Nestor could not have been so refined as to write such letters.

        Explicit Components:
        1 - These letters attributed to Nestor show a classical rhetorical style.
        2 - We know that Nestor never attended a literary writing course.
        3 - Nestor could not have been so refined as to write such letters.

        Explicit Relation under Analysis:
        1 and 2 jointly support 3

        Implicit Components:
        1 - Only those who have attended a literary writing course write with a classical rhetorical style.
        2 - Those who write letters in a classical rhetorical style are refined.

        ---
        **Example 2: (No Implicit Premises)**

        Text:
        People who get regular exercise sleep better. I go jogging three times a week. Therefore, I sleep better than before.

        Explicit Components:
        1 - People who get regular exercise sleep better.
        2 - I go jogging three times a week.
        3 - I sleep better than before.

        Explicit Relation:
        1 and 2 jointly support 3

        Implicit Components:
        0

        ---
        **Example 3:**

        Text:
        Our city has high summer temperatures, and the nearest public swimming pool is far away. Therefore, the city should build a local pool.

        Explicit Components:
        1 - The city has high summer temperatures.
        2 - The nearest public swimming pool is far away.
        3 - Therefore, the city should build a local pool.

        Explicit Relation:
        1 and 2 jointly support 3

        Implicit Components:
        1 - High temperatures combined with lack of nearby pools create a significant need for local swimming options.

        ---
        **Example 4:**

        Text:
        It is interesting to note that art critics write more about works they do not appreciate than about those they do. Therefore, art critics write more about works that do not meet their criteria of value. Consequently, art critics write about works that are not great works of art. Now, if art criticism has formative value, then art critics should write about works that are great works of art. You can already imagine the conclusion.

        Explicit Components:
        1 - Art critics write more about works they do not appreciate than about those they do.
        2 - Art critics write more about works that do not meet their criteria of value. 
        3 - Art critics write about works that are not great works of art. 
        4 - If art criticism has formative value, then art critics should write about works that are great works of art.

        Explicit Relation:
        2 supports 3

        Implicit Components:
        1 - Works that do not meet the critics' criteria of evaluation are not great works of art.

        ---

        Now apply the same logic to the following case.

        **Text:**
        {text}

        **Explicit Components:**
        {arg_components}

        **Premise Number:** {prem_ids}
        **Conclusion Number:** {prem_ids}

        ---

        **Task:**
        Write any implicit premise(s) that are *both*:
        (a) necessary to link the given premise(s) to the conclusion, and  
        (b) not already expressed in any explicit component.

        **Quality criteria (duplication check)**  
        ✓ *Novelty*: The statement must add new content not found in any explicit component.  
        ✓ *Necessity*: Removing the statement would break the inference.  
        ✗ *No paraphrases or repetitions*: If a candidate merely rephrases or restates what's already there, omit it.

        **Output rules**  
        1. Number starting at 1 (independent of explicit numbers).  
        2. Separate multiple implicit premises with semicolons (;).  
        3. If none are needed, output exactly **0**.  
        4. Focus **only** on the premise(s) and conclusion mentioned in the relation. Ignore other components.

        ---
        Output format (one line):
        Answer: <implicit premises>  (e.g., Answer: 1 - Implicit sentence one; 2 - Implicit sentence two)
        or
        Answer: 0
        '''
    return implicit_premises


def implicit_prompt_attack(text, arg_components, prem_ids, concl_id):
    """
    Generate prompt to identify implicit premises in attack relations.
    
    Args:
        text: Original text
        arg_components: List of numbered components
        prem_ids: List of premise component IDs
        concl_id: Conclusion component ID
        
    Returns:
        Formatted prompt string
    """
    implicit_premises = f'''
        An argument often contains implicit premises. These premises are assumed in the conversation without being explicitly stated. They are essential to the argument's validity, because they provide necessary support for claims — or explain how a claim defeats another.

        Sometimes a single premise attacks a conclusion; sometimes multiple premises work together (convergently) to attack a conclusion. In both cases, implicit premises may be necessary.

        ---
        **Example 1: (Independent Attack)**

        Text:
        Some people claim meditation improves mental health, but multiple scientific studies show no measurable improvement. Therefore, meditation does not improve mental health.

        Explicit Components:
        1 - Some people claim meditation improves mental health.
        2 - Multiple scientific studies show no measurable improvement.
        3 - Meditation does not improve mental health.

        Explicit Relation:
        2 attacks 1

        Implicit Components:
        4 - Scientific studies are a more reliable source of truth than personal claims.

        ---
        **Example 2: (No Implicit Premises in Attack)**

        Text:
        Eating chocolate late at night makes it harder to fall asleep. Thus, you should avoid chocolate before bedtime.

        Explicit Components:
        1 - Eating chocolate late at night makes it harder to fall asleep.
        2 - You should avoid chocolate before bedtime.

        Explicit Relation:
        1 supports 2

        (There is no attack relation in this case.)

        Implicit Components:
        0

        ---
        **Example 3: (Convergent Attack)**

        Text:
        A company claims their new smartphone battery lasts all day. However, independent tests show it lasts only 6 hours, and hundreds of users report needing to recharge by mid-afternoon. Therefore, the company's claim is false.

        Explicit Components:
        1 - Independent tests show the battery lasts only 6 hours.
        2 - Users report needing to recharge by mid-afternoon.
        3 - The company's claim that the battery lasts all day.

        Explicit Relation:
        1 and 2 jointly attack 3

        Implicit Components:
        4 - A battery that lasts only 6 hours does not count as "lasting all day".

        ---
        **Example 4: (Convergent Attack with No New Implicit Premises)**

        Text:
        An online school advertises "guaranteed job placement after graduation." However, surveys show only 30% of graduates find jobs within six months. Financial audits show the school spends almost nothing on career services.

        Explicit Components:
        1 - Only 30% of graduates find jobs within six months.
        2 - The school spends almost nothing on career services.
        3 - "Guaranteed job placement" claim.

        Explicit Relation:
        1 and 2 jointly attack 3

        Implicit Components:
        0

        ---

        Now apply the same logic to the following case.

        ---
        **Text:**
        {text}

        **Explicit Components:**
        {arg_components}

        **Premise Number:** {prem_ids}
        **Conclusion Number:** {prem_ids}

        ---

        **Task:**
        Identify any implicit premises that are necessary to explain **how the premise(s) attack the conclusion**.

        **Instructions:**
        1. If implicit premises are needed, write each one in the format:  
        `<number> - <text>`, starting with number 1.
        2. Separate multiple implicit premises with semicolons (;).
        3. If no implicit premises are needed, answer only with the number `0`.
        4. Focus **only** on the premise(s) and conclusion mentioned in the relation. Ignore other components.

        ---
        Output format (one line):
        Answer: <implicit premises>  (e.g., Answer: 1 - Implicit sentence one; 2 - Implicit sentence two)
        or
        Answer: 0
        '''
    return implicit_premises


# =============================================================================
# TASK 7: COUNTERARGUMENT ANALYSIS
# =============================================================================

def get_counterarguments(text, arg_components, premises, target, relations_text, arg_components_attack):
    """
    Generate prompt to distinguish direct attacks from inference attacks.
    
    Args:
        text: Original text
        arg_components: List of numbered components
        premises: Attacking premise(s)
        target: Target component being attacked
        relations_text: Existing relations involving the target
        arg_components_attack: Formatted list of attack relations
        
    Returns:
        Formatted prompt string
    """
    instruction = f"""
        You are given an argumentative text and a list of its components (e.g., premises and conclusions), along with an existing attack relation.

        Your task is to determine whether each attacking component in this relation:

        - **Directly attacks** the **content** of a target component (e.g., its truth, value, relevance), or  
        - **Indirectly attacks** the **inference** between a supporting component (a premise or a group of premises) and the conclusion it supports — i.e., the **reasoning link**.

        ---

        🔍 **Key Definitions**

        1. **Direct Attack**:  
        The attacker disputes the **target component itself**.

        2. **Attack on an Inference** (also called **undermining an inference**):  
        The attacker does **not dispute the target directly**, but instead challenges the **connection between premise(s) and the conclusion** — i.e., it claims the premise(s) do **not** validly support the conclusion.


        💡 Consider whether this attack relation really holds.  
        Could it be that the attacker is **actually challenging the inference** (i.e., the reasoning link) rather than directly opposing the target statement?

        ---
        ### ✅ **Examples**

        #### Example 1: Direct Attack

        **Text**:  
        "Climate change is a hoax. The Earth has always gone through natural warming and cooling cycles."

        **Components**:
        - (1) Climate change is a hoax. *(Conclusion)*  
        - (2) The Earth has always gone through natural warming and cooling cycles. *(Premise)*

        **Current attack relation**:  
        Premise (3) "Scientists have found unprecedented rates of warming due to human activity" attacks component (1)

        **Explanation**:  
        Component (3) is **directly denying** the truth of (1). It contradicts the conclusion and presents an alternative claim, without referring to the support provided by any premises. Therefore, this is a **direct attack** on the component.

        ---

        #### Example 2: Attack on Inference

        **Text**:  
        "Students who get good grades work hard. Therefore, Maria works hard."

        **Components**:
        - (1) Students who get good grades work hard. *(Premise)*  
        - (2) Maria gets good grades. *(Premise)*  
        - (3) Therefore, Maria works hard. *(Conclusion)*

        **Current attack relation**:  
        Premise (4) "Some students get good grades by cheating" attacks component (3)

        **Explanation**:  
        Component (4) does not directly reject (3), but instead challenges the logic connecting (1) and (2) to (3). It implies that the premises do **not necessarily** support the conclusion, by introducing a counterexample. This is an **attack on the inference**.

        ---

        #### Example 3: Attack on Inference (Single Premise)

        **Text**:  
        "Lowering taxes increases economic growth."

        **Components**:
        - (1) Lowering taxes increases economic growth. *(Conclusion)*  
        - (2) Countries that cut taxes saw their GDP rise. *(Premise)*

        **Current attack relation**:  
        Premise (3) "Those countries also received large foreign investments at the same time" attacks component (1)

        **Explanation**:  
        Component (3) does not deny that growth occurred, but rather suggests an **alternative explanation**, weakening the support that (2) provides to (1). This challenges the **validity of the inference**.

        ---

        #### Example 4: Direct Attack with Emotional Framing

        **Text**:  
        "The death penalty deters crime."

        **Components**:
        - (1) The death penalty deters crime. *(Claim)*

        **Current attack relation**:  
        Premise (2) "That's barbaric and inhumane" attacks component (1)

        **Explanation**:  
        Component (2) directly challenges the **value** of the claim, asserting that it is unethical. It does **not** focus on reasoning or premises — this is a **direct attack** on the target statement.

        ---
        Now, analyze the current attack:

        - **Text**:  
        {text}

        - **Components**:  
        {arg_components}

        - **Current attack relation**:  
        Premise(s) {premises} attack(s) component {target}"

        - **Relations that involve component {target}. They are expressed in format <link> - <description>:
        {relations_text}

        ---
        📤 **Response Format**

        You must indicate whether the current attack relation should be **kept as is** or **revised**.

        - ✅ If the current attack is **correct** as a direct attack, return:  
        `ANSWER: 0`

        - ✅ If the current attack is actually attacking an inference, indicate the link that is being attacked, with the formar `ANSWER: <number>. Use the numbering above:
        {arg_components_attack}

        ---

        Indicate your final answer with ANSWER:

"""
    return instruction


# =============================================================================
# TASK 8: PREMISE EVALUATION
# =============================================================================

def evaluation_text(text, arg_components, links, idx, premise):
    """
    Generate prompt to evaluate premise quality (acceptability, relevance, sufficiency).
    
    Args:
        text: Original text
        arg_components: List of numbered components
        links: List of identified links
        idx: Index of the link to evaluate
        premise: Premise component being evaluated
        
    Returns:
        Formatted prompt string
    """
    eval_prompt = f'''
    An argument is made up of premises that support or attack a conclusion; premises that, in turn, can be supported or attacked by others. To evaluate the quality of an argument, we need to assess the premises it is composed of and how they relate to the conclusion.
    A premise can be evaluated according to three aspects: acceptability, relevance, and sufficiency.
    1 - Acceptability: whether the premises appear as acceptable to the interlocutors. This criterion, in turn, is divided into three main points: A - Are the premises true or probable?; B - What source are they based on: scientific knowledge, common sense, personal report, etc..; C - Within the context in which the argumentation is inserted, can the premise be considered controversial?
    2 - Relevance: whether the premises are related to the content of the conclusion, in a way that contributes to its support or refutation. For example, consider the text: "I think the inspection done by the engineer on the cracks in the house was not sufficient to determine the cause. After all, he did not bother to visit the construction next door, to assess the impact generated. His measurements were also quite imprecise. Moreover, the service was quite expensive." In this argument, the last sentence ("the service was quite expensive") states that the price of the inspection was high. Although one can question the price charged, the fact that the inspection is expensive is not relevant to the establishment of the point presented in the conclusion, namely, the non-detection of the cause of the problem.
    3 - Sufficiency: whether the elements presented in the premise are enough to support the conclusion in view. Consider the following example: "Mr. X is the president of the country. Mr. Y is from party Y. The most powerful man in the country is from party Y." The premises listed are insufficient to ensure the conclusion: the president is not necessarily the most powerful person in a country; the most important person in a country may be an opposition leader, a businessman, or some other person.

    Now, consider the following argument:

    Text: {text}

    Argumentative components:

    {arg_components}

    The following links have already been identified:

    {links}

    Now, evaluate the premise {premise} in the following argumentative relationship: {links[idx]},  according to the three criteria mentioned above: acceptability, relevance, and sufficiency. 

    Give your answer according to the following format:

    *Acceptability:* A - <text>; B - <text>; C - <text>

    *Relevance*: <text>

    *Sufficiency*: <text>

    *Summary*: <text>

    '''
    return eval_prompt


def individual_premise(text, arg_components, dict_components, premise, conclusion):
    """
    Generate prompt to test individual premise-conclusion relation.
    
    Args:
        text: Original text
        arg_components: List of numbered components
        dict_components: Dictionary mapping IDs to text
        premise: Premise component ID
        conclusion: Conclusion component ID
        
    Returns:
        Formatted prompt string
    """
    instruction = f'''Consider the following argument:

Text: {text}

Argumentative components:

{arg_components}

Indicate the logical relation between element {premise} ("{dict_components[premise]}") and the main conclusion of the argument, which is the argumentative component number {conclusion} ("{dict_components[conclusion]}"). 

Additional instruction:
1 - The relation may be one of the following: direct support (+), direct attack (-) or no direct relation (0). Answer with the correct symbol only.
2 - Consider only the direct relation between the premise and the conclusion; i.e., not mediated by other premises.

    '''
    return instruction
