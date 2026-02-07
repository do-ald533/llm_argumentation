from collections import defaultdict
import networkx as nx
from copy import deepcopy
from src.utils import generate_completion, components_dict, renumber_and_insert, create_arg_components, merge_argumentative_components, extract_answer, add_implicit_sentences, update_convergent_results, extract_last_answer, eliminate_and_insert, renumber_all
from llm.prompts import argumentative_components, components_corrected, merge_components, rewrite_sentence, argumentative_conclusion, premise_support, premise_attack, missing_premise_support, missing_premise_attack, convergent_premises_support, convergent_premises_attack, merge_components_cycle, implicit_prompt_support, implicit_prompt_attack, get_counterarguments

## Task 1: Identify argumentative components
def task_identify_components(model, text, tokenizer):
    arg_prompt = argumentative_components(text)

    # Create string with the arguments
    arg_components_full = generate_completion(model, tokenizer, arg_prompt)

    #Create a dictionary with the argumentative components
    dict_components = components_dict(arg_components_full)

    # Create string with arguments
    arg_components = create_arg_components(dict_components)

    return dict_components, arg_components

def get_corrected_indices(argument, model, tokenizer):
    """
    Generates corrected indices for an argument using the model and prompt.

    Parameters:
    - argument: str - the argument to correct.
    - model, tokenizer: model and tokenizer for generation.
    - components_corrected: function to generate correction prompt.
    - generate_completion: function to get model output.
    - answer_keyword: str - keyword to extract answer from output.

    Returns:
    - List of corrected indices.
    """
    print('Original argument:', argument)
    correct_prompt = components_corrected(argument)
    
    corrected_component_full = generate_completion(model, tokenizer, correct_prompt)
    print('New components:', corrected_component_full)

    extracted_indices = extract_answer(corrected_component_full, 'Answer')

    return extracted_indices

def task_correct_components(components, model, tokenizer):
    """
    Splits and renumbers argumentative components.
    Returns the updated dict and the merged string.
    """
    updated = components.copy()                       # work on a copy
    for idx in sorted(components.keys(), reverse=True):
        argument = components[idx]

        # ask the model whether / how to split
        corrected_prompt       = components_corrected(argument)
        corrected_component_txt = generate_completion(model, tokenizer,
                                                      corrected_prompt)
        print("New components from model:\n", corrected_component_txt)

        # turn "1 - …\n2 - …" into a dict
        new_entries = components_dict(corrected_component_txt)

        # if the model really produced >1 entry, insert them
        if len(new_entries) > 1:
            updated = renumber_and_insert(updated, idx, new_entries)

    merged_string = create_arg_components(updated)
    return updated, merged_string

def task_merge_components(model, tokenizer, text, arg_components, dict_components):
    # Generate prompt for merging
    merge_prompt = merge_components(text, arg_components)

    # Generate merged output
    merge_full = generate_completion(model, tokenizer, merge_prompt)

    # Check if there is a valid merge (not empty and not "0")
    if merge_full and merge_full.strip() != "0":
        # Update dictionary with merged components
        dict_components = merge_argumentative_components(dict_components, merge_full)
        # Recreate merged argumentative components string
        arg_components = create_arg_components(dict_components)

    return dict_components, arg_components

def task_rewrite_sentences(model, tokenizer, text, arg_components):
    # Generate prompt for rewriting
    rewritten_prompt = rewrite_sentence(text, arg_components)

    # Get rewritten components
    rewritten_output = generate_completion(model, tokenizer, rewritten_prompt)

    # Recreate dict from rewritten output
    dict_components = components_dict(rewritten_output)

    return dict_components, rewritten_output

def task_extract_conclusion(model, tokenizer, text, arg_components):
    # Generate prompt for extracting conclusion
    conclusion_prompt = argumentative_conclusion(text, arg_components)

    # Generate model output for the conclusion
    conclusion_component_full = generate_completion(model, tokenizer, conclusion_prompt)

    # Extract the conclusion number or label
    conclusion_number = extract_answer(conclusion_component_full, 'CONCLUSION')[0]

    return conclusion_number

def get_support_numbers(model, tokenizer, current, text, arg_components, dict_components, forbidden_nodes):
    prompt_support = premise_support(current, text, arg_components, dict_components)
    
    support_relation = generate_completion(model, tokenizer, prompt_support)

    support_numbers = extract_answer(support_relation, 'Answer')

    filtered_support_numbers = [num for num in support_numbers if num != 0 and num not in forbidden_nodes]

    return filtered_support_numbers

def get_attack_numbers(model, tokenizer, current, text, arg_components, dict_components, forbidden_nodes):
    prompt_attack = premise_attack(current, text, arg_components, dict_components)

    attack_relation = generate_completion(model, tokenizer, prompt_attack)

    attack_numbers = extract_answer(attack_relation, 'Answer')

    filtered_attack_numbers = [num for num in attack_numbers if num != 0 and num not in forbidden_nodes]

    return filtered_attack_numbers

def task_identify_relations(model, tokenizer, text, arg_components, dict_components, conclusion_number):
    links = []
    visited = set()
    to_be_visited = [conclusion_number]
    children = {}

    while to_be_visited:
        current = to_be_visited.pop(0)

        if current in visited:
            continue
        visited.add(current)

        # --- Find forbidden siblings ---
        forbidden_nodes = set()
        for parent, siblings in children.items():
            if current in siblings:
                forbidden_nodes = set(siblings)
                break

        # --- Support ---
        filtered_support_numbers = get_support_numbers(model, tokenizer, current, text, arg_components, dict_components, forbidden_nodes)

        if filtered_support_numbers:
            children[current] = filtered_support_numbers
            for prem in filtered_support_numbers:
                if prem not in visited:
                    print('Premise:', prem)
                    links.append([[prem], current, 'support'])
                    to_be_visited.append(prem)

        # --- Attack ---
        filtered_attack_numbers = get_attack_numbers(model, tokenizer, current, text, arg_components, dict_components, forbidden_nodes)

        if filtered_attack_numbers:
            if current in children:
                children[current].extend(filtered_attack_numbers)
            else:
                children[current] = filtered_attack_numbers

            for prem in filtered_attack_numbers:
                if prem not in visited:
                    links.append([[prem], current, 'attack'])
                    to_be_visited.append(prem)

    not_in_set = [key for key in dict_components if key not in visited]

    return links, visited, not_in_set

def get_conclusions(premise, text, arg_components, dict_components, model, tokenizer, mode='support'):
    if mode == 'support':
        prompt = missing_premise_support(premise, text, arg_components, dict_components)
    elif mode == 'attack':
        prompt = missing_premise_attack(premise, text, arg_components, dict_components)
    else:
        raise ValueError("Mode must be either 'support' or 'attack'.")

    completion = generate_completion(model, tokenizer, prompt)

    return extract_answer(completion, 'Answer')

def eliminate_cycle(model, tokenizer, text, arg_components, dict_components, component_ids):
    merged_component_prompt = merge_components_cycle(text, arg_components, dict_components, component_ids)
    merged_component = generate_completion(model, tokenizer, merged_component_prompt)
    print('Merged component:', merged_component)

    dict_components, new_id = eliminate_and_insert(dict_components, component_ids, merged_component)
    return dict_components, new_id, merged_component

def task_missing_links(model, tokenizer, text, arg_components, dict_components, not_in_set, links, conclusion_number):
    temp_links = []
    cycles = set()

    # Step 1: Collect links from missing components
    for premise in not_in_set:
        print('Premise not in set:', premise)
        # Support links
        targets = get_conclusions(premise, text, arg_components, dict_components, model, tokenizer, mode='support')
        print('Support:', targets)
        if targets and targets != [0]:
            for t in targets:
                temp_links.append((premise, t, 'support'))
                print((premise, t, 'support'))

        # Attack links
        targets = get_conclusions(premise, text, arg_components, dict_components, model, tokenizer, mode='attack')
        print('Attack:', targets)
        if targets and targets != [0]:
            for t in targets:
                temp_links.append((premise, t, 'attack'))
                print((premise, t, 'attack'))

    # Step 2: Detect 2-node cycles
    for a, b, kind in temp_links:
        for x, y, _ in temp_links:
            if x == b and y == a:
                cycles.add(frozenset({a, b}))

    # Step 3: Handle cycles
    if cycles:
        print("⚠️ Found cycles:")
        for cycle in cycles:
            print(" → ".join(str(x) for x in cycle))

            component_ids = sorted(list(cycle))
            # Eliminate and merge
            dict_components, new_id, merged_text = eliminate_cycle(
                model, tokenizer, text, arg_components, dict_components, component_ids
            )

            # Update arg_components
            arg_components = create_arg_components(dict_components)

            # Ask what the merged component supports/attacks
            for mode in ('support', 'attack'):
                targets = get_conclusions(new_id, text, arg_components, dict_components, model, tokenizer, mode=mode)
                if targets and targets != [0]:
                    for t in targets:
                        links.append([[new_id], t, mode])
    else:
        print("✅ No cycles found.")

    # Step 4: Add remaining (non-cyclic) links
    for a, b, kind in temp_links:
        if frozenset({a, b}) not in cycles:
            links.append([[a], b, kind])

    new_dict_components, new_arg_components, new_links, new_conclusion_number = renumber_all(dict_components, links, conclusion_number)

    return new_dict_components, new_arg_components, new_links, new_conclusion_number



def get_implicit_premises(model, tokenizer, text, arg_components,
                          relation, conclusion, prem_ids):
    """
    Generates a prompt and extracts implicit premises for a given relation.
    """
    if relation == 'support':
        prompt = implicit_prompt_support(text, arg_components, prem_ids, conclusion)
    else:
        prompt = implicit_prompt_attack(text, arg_components, prem_ids, conclusion)

    imps_raw = generate_completion(model, tokenizer, prompt)

    imps_clean = extract_last_answer(imps_raw)

    return imps_clean

def get_convergent_numbers(model, tokenizer, text, arg_components, dict_components, relation, conclusion, premises):
    """
    Generates prompt and returns extracted convergent numbers for a given relation.
    """
    if relation == 'support':
        prompt = convergent_premises_support(conclusion, text, arg_components, dict_components, premises)
    else:
        prompt = convergent_premises_attack(conclusion, text, arg_components, dict_components, premises)

    answer_full = generate_completion(model, tokenizer, prompt)

    answer = extract_answer(answer_full, 'Answer')

    return answer

def task_convergent_premises(links, text, arg_components, dict_components, model, tokenizer):
    """
    Analyzes convergent and independent premises based on 'support' and 'attack' links.

    Parameters:
        links (list): List of tuples with ((src, tgt), relation_type)
        text (str): The full text of the argumentation
        arg_components (dict): Dictionary mapping component indices to their text
        dict_components (dict): Dictionary of argument components with their metadata
        model: Language model used for generating completions
        tokenizer: Tokenizer corresponding to the model

    Returns:
        dict: A dictionary with conclusions as keys and their convergent premises per relation type.
    """
    # Collect every premise that targets the same (conclusion, relation)
    grouped = defaultdict(list)          # key -> flat list of premise ints
    passthrough = []                     # links we won't touch

    for prem_list, conclusion, relation in links:
        # Only "support" or "attack" directed at a numbered conclusion
        if relation in {"support", "attack"} and isinstance(conclusion, int):
            grouped[(conclusion, relation)].extend(prem_list)  # still singletons
        else:
            passthrough.append([prem_list, conclusion, relation])

    new_links = []

    for (conclusion, relation), premise_pool in grouped.items():
        # Nothing to merge if < 2 premises
        if len(premise_pool) < 2:
            new_links.extend([[[p], conclusion, relation] for p in premise_pool])
            continue

        # Ask the model which premises really form a convergent set
        convergent = get_convergent_numbers(
            model, tokenizer, text, arg_components, dict_components,
            relation, conclusion, premise_pool
        )

        # If the model found a convergent set (Answer: 0 means "none")
        if convergent and convergent != [0] and len(convergent) > 1:
            # • One combined link for the convergent group
            new_links.append([sorted(convergent), conclusion, relation])

            # • Keep any leftover premises as independent links
            leftovers = set(premise_pool) - set(convergent)
            new_links.extend([[[p], conclusion, relation] for p in leftovers])
        else:
            # No convergence – keep original single-premise links
            new_links.extend([[[p], conclusion, relation] for p in premise_pool])

    # Preserve every link that never needed processing
    new_links.extend(passthrough)

    return new_links

def task_implicit_premises(text, arg_components, dict_components, convergent_results, model, tokenizer):
    """
    For every (premise->conclusion) or (premise-set->conclusion) relation
    stored in `convergent_results`, ask the model for any missing implicit
    premises, add those sentences to `dict_components`, and extend the
    *same* relation entry with the newly created premise IDs.

    ── convergent_results format ──────────────────────────────────────────
      [
        [[prem_id_1],     conclusion_id, relation],   # simple link
        [[prem_id_2,
          prem_id_3],     conclusion_id, relation],   # convergent link
        …
      ]
    Each element is modified **in place**: implicit premise IDs are appended
    to its first sub-list.  Example after processing:
      [[[1, 9],           7, 'support'], …]           # 9 is implicit
    ----------------------------------------------------------------------
    """

    # Iterate over a *copy* so we can mutate the original list safely
    for link in convergent_results[:]:
        explicit_premises, conclusion, relation = link
        prem_ids = list(explicit_premises)   # defensive copy

        # 1. Ask LLM whether we need implicit premises
        imps_clean = get_implicit_premises(
            model, tokenizer,
            text, arg_components,
            relation,                    # 'support' or 'attack'
            conclusion,                  # conclusion ID
            prem_ids)
        print('Implicit premise(s):', imps_clean)
        # 2. If none are needed, continue to next relation
        if not imps_clean or imps_clean.strip() == "0":
            continue

        # 3. Otherwise, add the sentences to the component dictionary …
        #    `add_implicit_sentences` must:
        #      • create one new component per implicit sentence;
        #      • return the *list* of new IDs (in creation order).
        new_ids = add_implicit_sentences(
            imps_clean,
            dict_components,
            convergent_results,          # still useful for duplicate checks
            (prem_ids, conclusion))      # current explicit link

        print('Original link:', link)
        # 4. Extend the current link *in place* with those IDs …
        link[0].extend(new_ids)
        print('Link with implicit premises:', link)

    # Build new arg_components list
    arg_components = create_arg_components(dict_components)

    return dict_components, arg_components, convergent_results
            
def get_attack_relation(convergent_results, links):
        all_attacks = {}

        # Process the dictionary
        for conclusion, inner_dict in convergent_results.items():
            if 'attack' in inner_dict:
                premises = inner_dict['attack']
                all_attacks.setdefault(conclusion, []).extend(premises)

        # Process the list
        attack_relations = [relation for relation in links if relation[1] == 'attack']
        for (premise, conclusion), _ in attack_relations:
            all_attacks.setdefault(conclusion, []).append(premise)

        return all_attacks

def get_related_links_and_convergents(target, links, attack):
    """
    Return every link in which the *target* component appears either
    • as the *conclusion*, or
    • *inside* the list of *premises*,
    **except** the attack link that is currently being inspected.
    """
    return [
        link
        for link in links
        if (
            link[1] == target          # → target is the conclusion
            or target in link[0]       # → target is one of the premises
        ) and link != attack           # but not the attack itself
    ]

def describe(dict_components, cid: int) -> str:
    return f'Component {cid} (“{dict_components[cid]}”)'


def describe_many(dict_components, cids) -> str:
    """
    Turn [4,5,6] into
      Component 4 (“…”), Component 5 (“…”), and Component 6 (“…”)
    """
    if len(cids) == 1:
        return describe(dict_components, cids[0])
    if len(cids) == 2:
        return f'{describe(dict_components, cids[0])} and {describe(dict_components, cids[1])}'
    all_but_last = ', '.join(describe(dict_components, cid) for cid in cids[:-1])
    return f'{all_but_last}, and {describe(dict_components, cids[-1])}'

def build_relation_lines(dict_components, related_links) -> str:
    """Return a human-readable, line-per-link description."""
    lines = []

    for premises, conclusion, rel in related_links:
        verb       = "support" if rel == "support" else "attack"
        plural     = "Premises" if len(premises) > 1 else "Premise"
        repr_link  = [premises, conclusion, rel]

        if len(premises) == 1:
            lines.append(
                f'{repr_link} – {describe(dict_components, premises[0])} {verb}s '
                f'{describe(dict_components, conclusion)}.'
            )
        else:  # convergent argument
            lines.append(
                f'{repr_link} – {plural} {describe_many(dict_components, premises)} jointly '
                f'{verb} {describe(dict_components, conclusion)}.'
            )

    return '\n'.join(lines) if lines else '– (no relations) –'

def create_dict_attacks(related_links):
    """
    Return {1: link-0, 2: link-1, …} so that helper functions that rely
    on numeric indices (e.g. remove_instance_and_update or the prompt
    you feed to the language model) keep working unchanged.
    """
    return {i + 1: link for i, link in enumerate(related_links)}


def remove_instance_and_update(
    dict_attack,
    updated_links,
    counterarguments,
    answer_token,
    attacker_premises,     # p. ex.  [4, 8]
    attacker_target,       # p. ex.  1
):
    try:
        idx = int(answer_token)
    except (ValueError, TypeError):
        return updated_links, counterarguments

    if idx == 0 or idx not in dict_attack:
        return updated_links, counterarguments

    # inferência que está sendo atacada
    inf_premises, inf_concl, _ = dict_attack[idx]

    # ---------  ❱❱ remove SOMENTE o ataque original ❰❰  ---------
    updated_links = [
        lnk for lnk in updated_links
        if not (lnk[0] == attacker_premises     # mesmas premissas-atacantes
                and lnk[1] == attacker_target   # mesmo alvo
                and lnk[2] == "attack")         # e é do tipo "attack"
    ]

    # registra o contra-argumento (ataque à inferência)
    record = [attacker_premises, [inf_premises, inf_concl], "attack"]
    if record not in counterarguments:
        counterarguments.append(record)

    return updated_links, counterarguments


def task_counterarguments(
    model,
    tokenizer,
    text,
    arg_components,
    dict_components,
    links,
):
    counterarguments = []
    updated_links = [link[:] for link in links]     # shallow copy is enough

    attack_relations = [link for link in updated_links if link[2] == "attack"]

    for attack in attack_relations:
        premises, target, _ = attack
        print('premise(s)', premises, 'attack(s)', target)
        related_links = get_related_links_and_convergents(target, updated_links, attack)
        relations_text        = build_relation_lines(dict_components, related_links)
        dict_attack           = create_dict_attacks(related_links)
        print('Attack Dictionary:', dict_attack)
        arg_components_attack = create_arg_components(dict_attack)

        counterargument_prompt = get_counterarguments(
            text,
            arg_components,
            premises,
            target,
            relations_text,
            arg_components_attack,
        )

        counterargument_full   = generate_completion(model, tokenizer, counterargument_prompt)

        # ▶️  cast to str just to be safe
        counterargument_answer = str(extract_answer(counterargument_full, "ANSWER")[0])

        # ▶️  use the re-written helper
        updated_links, counterarguments = remove_instance_and_update(
        dict_attack,
        updated_links,
        counterarguments,
        counterargument_answer,
        premises,          # attacker_premises  (e.g. [1, 9])
        target,            # attacker_target    (e.g. 7)
        )
        print('updated:', updated_links)
        print('counter:', counterarguments)
    return updated_links, counterarguments

def task_create_acyclic_graph(related_links):
    """
    Build an acyclic graph from `related_links`, perform a transitive
    reduction, and return

        • reduced_links  – the surviving edges with their relations
        • new_conv       – convergent-group information that is still valid
                           after the reduction.

    Parameters
    ----------
    related_links : list
        Each element must be of the form
            [premise_ids, target_id, relation_type]
        where
            premise_ids     – list of one or more component IDs
            target_id       – component ID that the premises support / attack
            relation_type   – e.g. 'support', 'attack'

    Returns
    -------
    reduced_links : list[tuple[((int, int), str)]]
        [((premise, target), relation_type), …]

    new_conv : dict[int, dict[str, list[list[int]]]]
        new_conv[target][relation] → list of premise-lists
        (only groups with ≥2 premises and whose edges survive
         the transitive reduction are kept).
    """
    # ------------------------------------------------------------------ #
    # STEP 0 ─ normalise the input and prepare two helper structures:
    #          (i) edge list     (ii) candidate convergent groups
    # ------------------------------------------------------------------ #
    links = []                 # edges for the graph
    convergent_results = {}    # {target: {relation: [premise_group, …]}}

    for premises, target, rel in related_links:
        # one edge per premise
        for p in premises:
            links.append(((p, target), rel))

        # only groups with ≥2 premises can be convergent
        if len(premises) >= 2:
            convergent_results \
                .setdefault(target, {}) \
                .setdefault(rel, []) \
                .append(premises)

    # ------------------------------------------------------------------ #
    # STEP 1 ─ build the full graph
    # ------------------------------------------------------------------ #
    DG = nx.DiGraph()
    for (u, v), rel in links:
        DG.add_edge(u, v, relation=rel)

    if not nx.is_directed_acyclic_graph(DG):
        raise ValueError("Graph is not acyclic – transitive reduction is undefined.")

    # ------------------------------------------------------------------ #
    # STEP 2 ─ transitive reduction
    # ------------------------------------------------------------------ #
    TR = nx.transitive_reduction(DG)

    # copy edge attributes over
    for u, v in TR.edges():
        TR[u][v].update(DG[u][v])

    reduced_links = [((u, v), TR[u][v]['relation']) for u, v in TR.edges()]

    # ------------------------------------------------------------------ #
    # STEP 3 ─ keep only convergent groups whose edges survived
    # ------------------------------------------------------------------ #
    valid_edges = {(u, v) for (u, v), _ in reduced_links}
    new_conv = {}

    for tgt, rels in convergent_results.items():
        for rel_type, groups in rels.items():
            kept_groups = []
            for group in groups:
                # retain only those premises whose edge (premise, tgt) survived
                valid_premises = [p for p in group if (p, tgt) in valid_edges]
                if len(valid_premises) >= 2:
                    kept_groups.append(valid_premises)

            if kept_groups:
                new_conv.setdefault(tgt, {})[rel_type] = kept_groups

    return reduced_links, new_conv