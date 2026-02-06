import re
from graphviz import Digraph
import numpy as np
import pandas as pd
from itertools import product
from dotenv import load_dotenv, find_dotenv
import openai
import copy
from typing import List, Dict, Tuple, Optional


def get_after_think(output):
    return output.split("</think>", 1)[-1].strip()

# Create a dictionary with the argumentative components
def components_dict(text):
    # Define a regular expression to find numbers followed by text
    # This pattern looks for a digit followed by ' - ' and then captures any characters until a new line or the end of the string
    pattern = r'(\d+) - (.*)'

    # Find all matches
    matches = re.findall(pattern, text)

    # Convert matches to a dictionary
    result = {int(num): desc.strip() for num, desc in matches}

    return result

def remove_self_loops(links):
    """
    Remove self-loops from a list of links.

    Args:
        links (list): List of tuples like [((source, target), label)]

    Returns:
        list: Filtered list without self-loops
    """
    return [link for link in links if link[0][0] != link[0][1]]

# Function to replace tuple keys with dictionary values
def replace_keys_with_values(data, mapping_dict):
    # Iterate over each sublist in the main list
    for sublist in data:
        # The first element of each sublist is a tuple of keys
        tuple_of_keys = sublist[0]
        # Replace the keys in the tuple with values from the dictionary
        replaced_tuple = tuple(mapping_dict[key] for key in tuple_of_keys if key in mapping_dict)
        # Update the tuple in the sublist
        sublist[0] = replaced_tuple
    return data

# Function to renumber and insert new entries
def renumber_and_insert(original_dict, target_key, new_entries):

    new_dict = {}

    additional_shift = len(new_entries) - 1  # We subtract 1 because we're replacing an entry

    for key in sorted(original_dict.keys()):

        if key < target_key:

            new_dict[key] = original_dict[key]

        elif key == target_key:

            for i, (new_key, value) in enumerate(new_entries.items(), start=key):

                new_dict[i] = value

        else:

            new_dict[key + additional_shift] = original_dict[key]

    return new_dict

def eliminate_and_insert(dict_components, component_ids, merged_text):
    # 1. Remove old components
    updated_dict = {k: v for k, v in dict_components.items() if k not in component_ids}

    # 2. Insert merged component with a new unique ID
    new_id = max(updated_dict.keys(), default=0) + 1
    updated_dict[new_id] = merged_text

    return updated_dict, new_id

def renumber_all(dict_components, links, conclusion_number):
    # Step 1: Create mapping from old_id to new_id
    sorted_old_ids = sorted(dict_components.keys())
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_old_ids, start=1)}

    # Step 2: Build new dict_components with new IDs
    new_dict_components = {id_map[old_id]: text for old_id, text in dict_components.items()}

    # Step 3: Build new arg_components list
    new_arg_components = create_arg_components(new_dict_components)

    # Step 4: Update links with new IDs
    new_links = []
    for src_list, target, kind in links:
        new_src = [id_map[src] for src in src_list]
        new_tgt = id_map[target]
        new_links.append([new_src, new_tgt, kind])

    # Step 5: Update conclusion_number
    new_conclusion_number = id_map.get(conclusion_number, None)

    return new_dict_components, new_arg_components, new_links, new_conclusion_number

def filter_and_renumber(dict_components, links, conclusion_number):
    # Step 1: Collect all IDs that appear in the links (as source or target)
    linked_ids = set()
    for src_list, target, _ in links:
        linked_ids.update(src_list)
        linked_ids.add(target)

    # Step 2: Filter the dictionary to include only linked components
    filtered_dict_components = {k: v for k, v in dict_components.items() if k in linked_ids}

    # Step 3: Renumber everything using your existing function
    return renumber_all(filtered_dict_components, links, conclusion_number)


# Create single links
def create_single_links(links):
    single_links = []

    for link in links:
        # Separate antecedent and consequent
        if '>' in link:
            splitted_premises = link.split('>')
        elif '~' in link:
            splitted_premises = link.split('~')

        antecedent = splitted_premises[0]
        consequent = splitted_premises[1]

        # Transform consequent into int
        # Use regular expression to get the numbers
        consequent = get_premises_numbers(consequent)[0]

        # Get premise numbers
        premises = get_premises_numbers(antecedent)

        for premise in premises:

            single_premise = (premise, consequent)
            single_links.append(single_premise) 

    return single_links

openai_api_key = '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'

def generate_completion(model, tokenizer, prompt, openai_api_key=openai_api_key, max_new_tokens=1500):
    """
    Generate a text completion using either a local model or OpenAI's GPT-4 API.

    Args:
        model: Hugging Face model for causal language modeling, or None if using GPT-4.
        tokenizer: Hugging Face tokenizer for the model, or None if using GPT-4.
        prompt (str): The input prompt to generate a response from.
        max_new_tokens (int): Maximum number of tokens to generate (default: 2000).
        model_name (str): If set to "gpt-4", will use OpenAI API; otherwise local.
        openai_api_key (str): Your OpenAI API key (only needed if using GPT-4).

    Returns:
        str: The generated text response.
    """
    if model in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "o4-mini"]:

        api_key = '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'

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=0.0,  # Deterministic
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        # response = client.chat.completions.create(
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ],
        #     max_tokens=max_new_tokens,
        #     temperature=0.0,  # Deterministic
        #     top_p=1.0,
        #     frequency_penalty=0.0,
        #     presence_penalty=0.0
        # )

        return response.choices[0].message.content.strip()

    elif model in ["deepseek-chat", "deepseek-reasoner"]:

        api_key = "sk-395a00b4d61746329113163bbfcdd445"
        
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=0.0,  # Deterministic
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        return response.choices[0].message.content.strip()

def filter_dict_components(dict_components, links):
    # Step 1: collect all unique component IDs from links and counterarguments
    used_ids = set()

    for premises, conclusion, _ in links:
        used_ids.update(premises)
        used_ids.add(conclusion)

    # Step 2: filter dict_components
    return {cid: text for cid, text in dict_components.items() if cid in used_ids}


def create_arg_components(dict_components):
    arg_components = '\n'.join(f"{key} - {value}" for key, value in dict_components.items())

    return arg_components

def merge_argumentative_components(original_dict, merges_str):
    # Step 1: Split string into separate merges (by line)
    merge_lines = [line.strip() for line in merges_str.strip().split('\n') if line.strip()]
    new_components = []
    merged_indices = set()

    # Step 2: Extract numbers and new merged text from each line
    for line in merge_lines:
        match = re.match(r'([\d,\s]+)-\s*(.+)', line)
        if match:
            nums = [int(n.strip()) for n in match.group(1).split(',')]
            text = match.group(2).strip()
            new_components.append((nums, text))
            merged_indices.update(nums)
        else:
            raise ValueError(f"Invalid merge format: {line}")

    # Step 3: Remove merged entries
    updated_dict = {k: v for k, v in original_dict.items() if k not in merged_indices}

    # Step 4: Add new merged texts
    for _, text in new_components:
        updated_dict[max(updated_dict.keys(), default=0) + 1] = text

    # Step 5: Renumber dictionary from 1
    renumbered_dict = {i + 1: v for i, (_, v) in enumerate(sorted(updated_dict.items()))}

    return renumbered_dict

# Create a directed graph
def create_graph(
    components: dict[int, str],
    name: str,
    count: int,
    conclusion: int,
    links_with_relation: list[tuple[tuple[int | str, int | str], str]],
    convergent_arguments: dict[int, dict[str, list[list[int]]]] | None = None,
    counterarguments: list[list] | None = None,
):
    """
    Draw an argument graph showing
      • ordinary support / attack edges
      • convergent groups                 – gray “i*” dots
      • attacks on inferences             – red  “c*” dots

    *counterarguments* format:
        [[attacker_premises], [[inf_premises], inf_concl], "attack"]
    """

    # ─────────────────────────────────────────────────────────────
    # 0 ─  Boiler-plate
    # ─────────────────────────────────────────────────────────────
    components = {k: v.replace(":", " -") for k, v in components.items()}
    dot = Digraph(comment="Argumentation Graph")

    label = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1">'
    for cid, txt in components.items():
        label += f'<TR><TD ALIGN="LEFT">{cid}</TD><TD ALIGN="LEFT">{txt}</TD></TR>'
    label += "</TABLE>>"
    dot.attr(label=label, fontsize="20")

    connector_nodes: set[str] = set()

    def ensure_connector(cid: str, colour: str):
        if cid in connector_nodes:
            return
        if cid.startswith("c"):  # nós c* com preenchimento cinza e borda vermelha
            dot.node(
                cid,
                label="",
                shape="circle",
                width="0.2",
                style="filled",
                fillcolor="none",
                color="red",
                penwidth="2"
            )
        else:  # nós i* (apoio ou ataque) com preenchimento e borda cinza
            dot.node(
                cid,
                label="",
                shape="circle",
                width="0.2",
                style="filled",
                fillcolor=colour,
                color='black'
            )
        connector_nodes.add(cid)

    # ─────────────────────────────────────────────────────────────
    # 1 ─  Convergent groups → “i*”
    # ─────────────────────────────────────────────────────────────
    i_counter = 1
    convergent_map: dict[tuple[int, int], str] = {}          # (premise, target) → i*

    if convergent_arguments:
        for tgt, rels in convergent_arguments.items():
            for rel_type, groups in rels.items():
                if groups and isinstance(groups[0], int):
                    groups = [groups]

                for premises in groups:
                    if len(premises) < 2:
                        continue

                    i_id = f"i{i_counter}"
                    i_counter += 1
                    ensure_connector(i_id, "gray")
                    convergent_map.update({(p, tgt): i_id for p in premises})

                    for p in premises:
                        dot.edge(str(p), i_id, color="black")

                    colour = "black" if rel_type == "support" else "red"
                    dot.edge(i_id, str(tgt), color=colour)

    # ─────────────────────────────────────────────────────────────
    # 2 ─  Attacks on inferences → “c*”
    # ─────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────
    # 2 ─  Attacks on inferences → “c*”
    # ─────────────────────────────────────────────────────────────
    c_counter = 1
    inference_connectors: dict[tuple[tuple[int, ...], int], str] = {}
    edges_to_skip: set[tuple[str, str]] = set()           # u, v *as str*

    if counterarguments:
        for attacker_premises, inf_desc, _ in counterarguments:
            # 2·1 – identify the inference that is being attacked
            inf_premises, inf_concl = inf_desc
            key = (tuple(sorted(inf_premises)), inf_concl)

            # 2·2 – create / reuse a red “c*” connector for that inference
            if key not in inference_connectors:
                c_id = f"c{c_counter}"
                c_counter += 1
                ensure_connector(c_id, "gray")
                inference_connectors[key] = c_id

                # ── find or create the ‘head’ feeding the inference
                if len(inf_premises) > 1:                    # convergent inference
                    head = None
                    for prem in inf_premises:
                        if (prem, inf_concl) in convergent_map:
                            head = convergent_map[(prem, inf_concl)]
                            break

                    # If group was not declared, fall back to “first premise”
                    if head is None:
                        head = str(inf_premises[0])

                        # skip every direct premise → conclusion edge
                        for p in inf_premises:
                            edges_to_skip.add((str(p), str(inf_concl)))

                        # wire *other* premises straight to c*
                        for p in inf_premises[1:]:
                            dot.edge(str(p), c_id, color="black")
                    else:
                        edges_to_skip.add((head, str(inf_concl)))
                else:                                       # single-premise inference
                    head = str(inf_premises[0])
                    edges_to_skip.add((head, str(inf_concl)))

                # re-route the inference through c*
                dot.edge(head, c_id, color="black")
                dot.edge(c_id, str(inf_concl), color="black")

            # 2·3 – add the attacking premises
            c_id = inference_connectors[key]

            if len(attacker_premises) > 1:
                # build a gray “i*” node to converge the attackers
                i_id = f"i{i_counter}"
                i_counter += 1
                ensure_connector(i_id, "gray")

                # premises → i*
                for p in attacker_premises:
                    dot.edge(str(p), i_id, color="black")

                # i*  ─red─►  c*
                dot.edge(i_id, c_id, color="red", arrowhead="normal")
            else:
                # single-premise attack
                dot.edge(str(attacker_premises[0]), c_id, color="red", arrowhead="normal")


    # ─────────────────────────────────────────────────────────────
    # 3 ─  Ordinary edges (skip those rewired / convergent)
    # ─────────────────────────────────────────────────────────────
    for (u, v), rel in links_with_relation:
        if (u, v) in convergent_map:
            continue
        if (str(u), str(v)) in edges_to_skip:
            continue

        colour = "black" if rel == "support" else "red"
        dot.edge(str(u), str(v), color=colour)

    # ─────────────────────────────────────────────────────────────
    # 4 ─  Style statement nodes
    # ─────────────────────────────────────────────────────────────
    for idx in components:
        if idx > count and idx != conclusion:
            dot.node(str(idx), color="gray")
    dot.node(str(conclusion), color="blue")

    # ─────────────────────────────────────────────────────────────
    # 5 ─  Export
    # ─────────────────────────────────────────────────────────────
    #file_path = f"Graphs/{name}"
    file_path = name.rsplit(".", 1)[0]
    dot.render(file_path, format="png", cleanup=True)
    print(f"Graph written to {file_path}.png")


# ---------- 1.  collect what is already convergent -----------------
def get_convergent_sets(conv_results: dict):
    """
    Returns a list like
        [({'premises': (1, 4), 'conclusion': 7, 'relation': 'support'}, ...]    
    One entry per convergent *set* that already exists in convergent_arguments.
    """
    groups = []
    for concl, rels in conv_results.items():
        for rel, prem in rels.items():
            if prem:                                     # ignore empty lists
                groups.append({
                    'premises': tuple(sorted(prem)),      # (1, 4)
                    'conclusion': concl,                  # 7
                    'relation': rel                       # 'support' | 'attack'
                })
    return groups

def all_convergent_premises(conv_results):
    return {p for rels in conv_results.values()
              for groups in rels.values()
              for g in groups
              for p in g}

# ---------- 2.  split the links -----------------------------------
def split_links(links, conv_results):
    """
    Returns two collections:
        * non_convergent   – every single-premise link whose premise is NOT part
                              of an already-known convergent set
        * convergent_sets  – the sets that *are* already in convergent_arguments
    """
    convergent_premises = {p            # {1, 4 …}
                           for rels in conv_results.values()
                           for plist in rels.values()
                           for p in plist}

    non_conv = []       # ➟ case (i)
    for ((prem, concl), rel) in links:
        if prem not in convergent_premises:
            non_conv.append(((prem, concl), rel))
    return non_conv, get_convergent_sets(conv_results)


def update_convergent_results(conv: dict,
                              explicit_premises: list,
                              implicit_premises: list,
                              conclusion: int,
                              relation: str = 'support') -> dict:
    """
    Returns a new convergent_results dictionary with updated information.
    Does not modify the original dictionary in-place.
    """

    # Create a deep copy so we don't mutate the original
    new_conv = copy.deepcopy(conv)

    all_prem = set(explicit_premises + implicit_premises)

    new_conv.setdefault(conclusion, {})
    new_conv[conclusion].setdefault(relation, [])

    groups = new_conv[conclusion][relation]  # list that may still be legacy

    # --- ① legacy upgrade ------------------------------------------
    if groups and isinstance(groups[0], int):  # old flat list -> one set
        new_conv[conclusion][relation] = [set(groups)]
        groups = new_conv[conclusion][relation]
    # ----------------------------------------------------------------

    # does this explicit premise overlap an existing group?
    for g in groups:  # now every g is a *set*
        if g & set(explicit_premises):
            g |= all_prem  # merge into that group
            break
    else:
        groups.append(all_prem)  # brand-new convergent group

    return new_conv

# Get negative premises
def get_negative_premises(links):
    negative_premises = []
    for link in links:
        if '~' in link:
            splitted_elem = link.split('~')[0]

            numbers = re.findall(r'\d+', splitted_elem)

            # Convert found strings to integers
            numbers_int = [int(number) for number in numbers]

            negative_premises.append(numbers_int)

    # Concatenate list of lists
    flat_list = list(np.concatenate(negative_premises))

    return flat_list

def extract_number(text):
    parts = text.split()  # Split the string into parts

    premise = int(parts[0])  # Get the premise number
    relation = parts[1]  # Get the relation
    target = int(parts[2])  # Get the target number

    # premise, relation, target
    return premise, relation, target
    
import re

def sentence_adder(implicit_premises, dict_components, links, idx, link):
    # Using regular expression to split the string by numbers followed by a dash
    splitted_premises = re.split(r'\d+ - ', implicit_premises)

    # Remove empty first element if needed
    if splitted_premises and splitted_premises[0] == '':
        splitted_premises.pop(0)

    # Clean the premises
    splitted_premises = [sentence.strip('; ') for sentence in splitted_premises]
    splitted_premises = [sentence if sentence.endswith('.') else sentence + '.' for sentence in splitted_premises]

    # Get the maximum key
    max_key = max(dict_components.keys())

    # Add to the dictionary and update the link
    for sentence in splitted_premises:
        max_key += 1
        dict_components[max_key] = sentence

        if '>' in links[idx]:
            splitted_link = links[idx].split('>')
            include_premise = splitted_link[0].strip() + ' + ' + str(max_key) + ' '
            links[idx] = include_premise + '>' + splitted_link[1].strip()

        elif '~' in links[idx]:
            splitted_link = links[idx].split('~')
            include_premise = splitted_link[0].strip() + ' + ' + str(max_key) + ' '
            links[idx] = include_premise + '~' + splitted_link[1].strip()

# ------------------------------------------------------------
# a) Add one or more implicit sentences, return their new IDs
# ------------------------------------------------------------

def create_convergent_nodes(links):

    new_links = []
    counter = 1

    for premises, target, relation in links:
        if len(premises) > 1:                  # we need an intermediate node
            cname = f'c{counter}'
            # each single premise points to the convergence node
            for p in premises:
                new_links.append([[p], cname, relation])
            # the convergence node points to the original target
            new_links.append([[cname], target, relation])
            counter += 1
        else:                                  # already a single-premise link
            new_links.append([[premises[0]], target, relation])

    return new_links

def add_implicit_sentences(
        raw_answer     : str,
        dict_components: Dict[int, str],
        link_store     : Optional[list] = None,
        attach_to      : Optional[Tuple[List[int], int]] = None
) -> List[int]:
    """
    Create new components for every implicit premise found in `raw_answer`
    and (optionally) register individual premise→conclusion links.

    Parameters
    ----------
    raw_answer : str
        Model output such as "1 - Cats are mammals; 2 - All mammals breathe air".
    dict_components : dict
        {component_id: component_text}. New components are inserted here.
    link_store : list, optional
        A list that stores binary links.  If provided *and* `attach_to` is
        provided, the function appends `[(new_id, conclusion), 'support']`
        for every new implicit premise.  If `None` (default) no links are added.
    attach_to : tuple, optional
        `(original_prem_ids, conclusion_id)`.  Needed only when `link_store`
        is not `None`.

    Returns
    -------
    List[int]
        The IDs assigned to the newly added implicit premises.
    """
    # 1 ───────────────────── split the model answer into sentences
    sentences = re.split(r'\d+\s*-\s*', raw_answer)
    if sentences and sentences[0] == '':
        sentences.pop(0)
    sentences = [s.strip().rstrip(';.') + '.'             # normalise punctuation
                 for s in sentences if s.strip()]
    if not sentences:
        return []

    # 2 ───────────────────── assign fresh component IDs
    next_id = (max(dict_components.keys()) + 1) if dict_components else 1
    new_ids = []

    for sent in sentences:
        dict_components[next_id] = sent
        new_ids.append(next_id)
        next_id += 1

    return new_ids

def update_convergent_results(convergent_results: dict, 
                               explicit_premises: list, 
                               implicit_premises: list, 
                               conclusion: int, 
                               relation: str = 'support'):
    """
    Updates the convergent_results dictionary by adding explicit and implicit premises
    that jointly support or attack a given conclusion.

    Parameters:
    - convergent_results: dict to be updated
    - explicit_premises: list of explicit premise IDs
    - implicit_premises: list of implicit premise IDs
    - conclusion: the ID of the conclusion
    - relation: 'support' (default) or 'attack'
    """

    # Ensure conclusion exists in the dictionary
    if conclusion not in convergent_results:
        convergent_results[conclusion] = {}

    # Ensure relation exists (support or attack)
    if relation not in convergent_results[conclusion]:
        convergent_results[conclusion][relation] = []

    # Add explicit and implicit premises
    all_premises = explicit_premises + implicit_premises

    # Update the list without duplicates
    convergent_results[conclusion][relation] = list(
        sorted(set(convergent_results[conclusion][relation] + all_premises))
    )

# ------------------------------------------------------------
# b) Build a fast look-up for “is this (prem, concl) convergent?”
# ------------------------------------------------------------
def make_convergent_pair_set(conv_res: dict):
    pair_set = set()
    for concl, rdict in conv_res.items():
        for rel, prems in rdict.items():
            for p in prems:
                pair_set.add((p, concl, rel))
    return pair_set


# ------------------------------------------------------------
# c) Yield all tasks you need to ask about
# ------------------------------------------------------------
def generate_implicit_tasks(links, convergent_results):
    """
    Yields dictionaries with keys:
        'premises'   - list[int]   (may have >1 premise if convergent)
        'relation'   - 'support' | 'attack'
        'conclusion' - int
    """
    # 1. independent edges
    conv_pairs = make_convergent_pair_set(convergent_results)
    for (src, tgt), rel in links:
        if (src, tgt, rel) not in conv_pairs:          # not handled by conv-node
            yield {'premises': [src], 'relation': rel, 'conclusion': tgt}

    # 2. convergent sets (≥2 premises already certified)
    for concl, rdict in convergent_results.items():
        for rel, prem_list in rdict.items():
            if len(prem_list) > 1:
                yield {'premises': prem_list,
                       'relation': rel,
                       'conclusion': concl}


# ------------------------------------------------------------
# d) Update convergent_results with new implicit IDs
# ------------------------------------------------------------
def merge_into_convergent(conv_res: dict, task: dict, new_ids: list[int]):
    if not new_ids:
        return

    concl  = task['conclusion']
    rel    = task['relation']
    if concl not in conv_res:
        conv_res[concl] = {}
    if rel not in conv_res[concl]:
        conv_res[concl][rel] = []
    conv_res[concl][rel].extend(new_ids)


# Get the premises numbers in a complex link
def get_premises_numbers(link):

    # Use regular expression to find all numbers in the part before the relation_symbol
    numbers = re.findall(r'\d+', link)

    # Convert found strings to integers
    numbers_int = [int(number) for number in numbers]

    return numbers_int

# Extracting the argument numbers from the link description
def extract_arg_number(link_description, relation, connection): # relation: support, attack, defeat; connection: + or ,
    # Splitting the string at the relation separator and taking the first part
    before_symbol = link_description.split(relation)[0]

    # Get concluision number
    conclusion_number = int(link_description[-1])
    
    # Getting premise numbers (i.e., the digits before the relation symbol)
    arg_numbers = [int(s) for s in re.findall(r'\b\d+\b', before_symbol)]
    
    return arg_numbers, conclusion_number

# Create dataframe with the links
def link_df(link_list):

    # Prepare a list to collect rows
    rows = []

    # Extract components from the single list
    id = link_list[0]
    text = link_list[1]
    relations = link_list[2]

    # Process each relation in the list of relations
    for relation in relations:
        component_1, component_2 = relation[0]
        rel_type = relation[1]
        
        # Append each row to the list
        rows.append([id, text, component_1, component_2, rel_type])
    
    # Create a DataFrame from the rows
    df = pd.DataFrame(rows, columns=['ID', 'Text', 'Component 1', 'Component 2', 'Relation'])

    return df

def extract_answer(text, keyword):
    # Match the simple pattern: Answer: 4, 6
    simple_pattern = rf'(?:{re.escape(keyword)}):\s*([\d,\s]+)(?!\s*-\s*\()'
    # Match the complex pattern: Answer: 2 - (3, 4)
    attack_pattern = rf'(?:{re.escape(keyword)}):\s*(\d+)\s*-\s*\((\d+)\s*,\s*(\d+)\)'

    # Try to match the complex pattern first
    attack_match = re.search(attack_pattern, text, flags=re.IGNORECASE)
    if attack_match:
        attack = int(attack_match.group(1))
        premise = int(attack_match.group(2))
        conclusion = int(attack_match.group(3))
        return [attack, (premise, conclusion)]

    # Try to match the simple pattern
    simple_match = re.search(simple_pattern, text, flags=re.IGNORECASE)
    if simple_match:
        numbers = [int(n.strip()) for n in simple_match.group(1).split(',') if n.strip().isdigit()]
        return numbers

    return []  # If nothing matches

def extract_last_answer(text):
    """
    Extracts the content after the LAST occurrence of 'Answer:' line in the text.
    """
    # Dividir o texto em linhas
    lines = text.splitlines()
    
    # Encontrar todas as linhas que começam com 'Answer:'
    answer_lines = [line for line in lines if line.strip().startswith('Answer:')]
    
    if answer_lines:
        last_answer_line = answer_lines[-1]
        # Remover 'Answer:' e limpar espaços
        return last_answer_line.replace('Answer:', '').strip()
    else:
        return None
    
# Create expanded link dataset
def full_df(df, dict_components):

    # Extract unique components from the DataFrame
    unique_components_df = pd.unique(df[['Component 1', 'Component 2']].values.ravel('K'))

    # Combine DataFrame components with dictionary components
    all_components = list(set(unique_components_df) | set(dict_components.values()))

    # Generate all possible combinations of these components
    all_combinations = list(product(all_components, repeat=2))

    # Create a new DataFrame with these combinations
    expanded_df = pd.DataFrame(all_combinations, columns=['Component 1', 'Component 2'])
    expanded_df['Relation'] = 'None'  # Initialize all relations with 'None'

    # Map existing relations from the original DataFrame
    # Create a temporary key to merge on based on component pairs
    df['tmp_key'] = df['Component 1'] + '-' + df['Component 2']
    expanded_df['tmp_key'] = expanded_df['Component 1'] + '-' + expanded_df['Component 2']

    # Merge to update relations where they exist
    expanded_df = expanded_df.merge(df[['tmp_key', 'Relation']], on='tmp_key', how='left')
    expanded_df['Relation'] = expanded_df['Relation_y'].fillna(expanded_df['Relation_x'])
    expanded_df.drop(columns=['tmp_key', 'Relation_x', 'Relation_y'], inplace=True)

    return expanded_df