import pandas as pd
import sys
from src.utils import create_graph, filter_and_renumber
from src.tasks import task_identify_components, task_correct_components, task_merge_components, task_rewrite_sentences, task_extract_conclusion, task_identify_relations, task_missing_links, task_convergent_premises, task_implicit_premises, task_counterarguments, task_create_acyclic_graph
import json

def executar_pipeline(argument_name, text, model_name="gpt-4.1", output_path="graph.png"):

    counterarguments = None
    print("Text received:", text)
    
    model = model_name
    tokenizer = None

    ## Task 1: Argument Mining
    dict_components, arg_components = task_identify_components(model, text, tokenizer)
    print("Task: Identify Components", arg_components)

    # Task 4: Conclusion Identification
    conclusion_number = task_extract_conclusion(model, tokenizer, text, arg_components)
    print('Task: Conclusion:', conclusion_number)

    ## Task 5: Premise Relation
    links, visited, not_in_set = task_identify_relations(model, tokenizer, text, arg_components, dict_components, conclusion_number)

    print('Visited:', visited)
    print("Final Links: ", links)
    print(f'Not in set: {not_in_set}')

    ## Task 6: Check Unvisited Premises
    dict_components, new_arg_components, links, conclusion_number = task_missing_links(model, tokenizer, text, arg_components, dict_components, not_in_set, links, conclusion_number)
    print('Task: Links with Missing Premises:', links)

    # Get the number of explicit premises
    count = len(dict_components)

    ## Task 10 and 11: Argumentative Graph and Transitive Reduction
    # Create graph
    reduced_links, convergent_arguments = task_create_acyclic_graph(links)
    print('Reduced links:', reduced_links)
    print('Convergent arguments:', convergent_arguments)
    print('Counterarguments:', counterarguments)
    print('Conclusion number:', conclusion_number)
    print('Dict components:', dict_components),

    ## Task 12: Diagram
    create_graph(
        components=dict_components,
        name=output_path,
        count=count,
        conclusion=conclusion_number,
        links_with_relation=reduced_links,
        convergent_arguments=convergent_arguments, # Create intermediate nodes for convergent premises
        counterarguments=counterarguments # Include counterarguments
    )

    # Save graph description
    data = {
        "dict_components": dict_components,
        "reduced_links": reduced_links,
        "convergent_arguments": convergent_arguments,
        "counterarguments": counterarguments,
        "conclusion": conclusion_number
    }

    # salvar
    with open(f"Graphs/{argument_name}.json", "w") as f:
        json.dump(data, f, indent=4)

df = pd.read_csv("arguments.csv", sep = ';')

def get_argument_text(name):
    row = df.loc[df["name"] == name, "text"]
    if not row.empty:
        return row.values[0]
    else:
        raise ValueError(f"Argument '{name}' not found.")

argument_name = sys.argv[1]  # get argument from terminal
text = get_argument_text(argument_name)

executar_pipeline(argument_name, text, model_name="gpt-5-mini", output_path=f"Diagrams/{argument_name}.png")