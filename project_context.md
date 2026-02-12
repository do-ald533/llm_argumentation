# Project Context — LLM Argumentation Structuring

## Overview

This system uses Large Language Models (LLMs) to automatically extract the argumentative structure from essays. Given a raw text, it identifies argumentative components (sentences), determines their roles, and builds a directed acyclic graph (DAG) of support/attack relations between them.

---

## Theoretical Foundation

### Argumentative Graph

An argument consists of **components** (sentences) and their logical **relations** — either *support* or *attack*. We model arguments as **directed acyclic graphs (DAGs)**, defined as:

$$G = (N, E)$$

where $N$ is the set of nodes (components) and $E$ the set of directed edges (relations).

- Each **node** is an argumentative component (a sentence or clause).
- Each **edge** is a directed relation indicating the source node either *supports* or *attacks* the target node.
- Relations are **unidirectional**: a premise supports/attacks a conclusion, never the reverse.
- The graph must be **acyclic**: a premise cannot be supported by its own conclusion.
- Complex relations like "2 and 3 support 4" are decomposed into one-to-one edges: "2 supports 4" and "3 supports 4".

### Component Labels (derived from graph structure)

Labels are **not** classified by the LLM — they are **derived automatically** from the graph topology:

| Label | Definition | Graph property |
|-------|-----------|---------------|
| **MajorClaim** | The main conclusion of the essay | Root node (the identified conclusion) |
| **Claim** | Intermediate argument that is both supported and supports others | Non-root node that is a *target* of at least one relation |
| **Premise** | Evidence or reason at the base of the argument | Leaf node — only a *source*, never a target |

### Transitive Reduction

After building the graph, we apply **transitive reduction** — removing direct edges between a premise and a conclusion when an indirect path already exists between them. This eliminates redundant connections that are a byproduct of the recursive prompting strategy (the LLM may first find a direct link, then later discover the relation is mediated through intermediate steps).

### DAG Enforcement

If any cycles remain (e.g., from the unvisited-component handling step), we detect them via DFS and remove the last edge in each cycle until the graph is acyclic.

---

## Pipeline Architecture

The system is orchestrated using **LangGraph** as a state machine with 5 sequential nodes:

```
identify_components → extract_conclusion → extract_relations → check_unvisited → finalize
```

### Step 1 — Identify Argumentative Components

**Task:** `IdentificationTask`  
**Input:** Raw essay text  
**Output:** `Dict[int, ArgumentComponent]` — numbered sentences that carry argumentative weight  
**Method:** The LLM is prompted to extract all argumentative sentences from the text. Each is assigned a sequential integer ID. Non-argumentative sentences (pure narrative, transitions) are excluded.

### Step 2 — Conclusion Identification

**Task:** `ConclusionExtractionTask`  
**Input:** Text + identified components  
**Output:** `conclusion_id: int` — the ID of the main conclusion  
**Method:** The LLM is asked which component is the main thesis/conclusion of the essay. This becomes the root of the argument tree.

### Step 3 — Recursive Relation Extraction (BFS)

**Task:** `RelationExtractionTask`  
**Input:** Text, components, conclusion_id  
**Output:** `(List[ArgumentRelation], Set[int] visited, List[int] unvisited)`  
**Method:**

1. Start a BFS queue with the conclusion node.
2. For each node visited:
   - Ask the LLM: *"Which components directly support this conclusion?"*
   - Ask the LLM: *"Which components directly attack this conclusion?"*
   - Valid answers are filtered against **forbidden nodes**: ancestors (already visited) and siblings (other premises of the same parent). This prevents cycles and redundant connections.
3. Each identified premise is added as a relation and queued for its own visitation.
4. Continue until the queue is empty.

After this step, some components may remain **unvisited** — they were not reached by BFS from the conclusion.

### Step 4 — Check Unvisited Premises

**Task:** `UnvisitedPremisesTask`  
**Input:** Text, components, existing relations, unvisited IDs, conclusion_id  
**Output:** Updated components, relations, conclusion_id  
**Method:**

1. For each unvisited component, ask the LLM what it supports/attacks.
2. **Cycle detection:** If two unvisited components point to each other, they form a 2-node cycle. These are merged into a single component via a merge prompt, and the merged component is re-assigned.
3. **Orphan fallback:** If the LLM returns no valid target for a component, it is connected to the main conclusion as a support (with a warning logged).
4. Stale relations (referencing deleted/merged component IDs) are cleaned up.

### Step 5 — Finalize

**Method:**
1. **Validate relations** — remove any edges referencing non-existent component IDs.
2. **Ensure DAG** — detect and break any remaining cycles.
3. **Transitive reduction** — remove redundant edges using `nx.transitive_reduction()`.
4. **Derive labels** — assign MajorClaim/Claim/Premise from graph structure.

---

## Data Flow

```
Input CSV (text_id, text_tokens)
        │
        ▼
┌─────────────────────────────┐
│   Step 1: Identify          │  LLM extracts argumentative sentences
│   Components                │  → Dict[int, ArgumentComponent]
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Step 2: Identify          │  LLM picks the main conclusion
│   Conclusion                │  → conclusion_id
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Step 3: Recursive BFS     │  For each node, LLM finds support/attack
│   Relation Extraction       │  premises, with forbidden-node filtering
│                             │  → relations + visited/unvisited sets
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Step 4: Unvisited         │  Orphan premises assigned to targets
│   Premises                  │  Cycles merged, stale edges cleaned
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Step 5: Finalize          │  validate → ensure DAG → transitive
│                             │  reduction → derive labels
└─────────────┬───────────────┘
              │
              ▼
     ArgumentGraph (DAG)
              │
       ┌──────┴──────┐
       ▼              ▼
  components_     relations_
  {prefix}.csv    {prefix}.csv
  (semicolon)     (semicolon)
```

---

## I/O Formats

### Input

CSV file with columns: `text_id`, `text_tokens`

```
,text_id,text_tokens
1065,AAEC_004,"International tourism is now more common than ever before..."
```

### Output — Components CSV

Separator: `;`

| Column | Description | Example |
|--------|-------------|---------|
| text_id | Source document ID | AAEC_004 |
| component_tokens | Full text of the component | "To conclude, I strongly believe..." |
| labels | MajorClaim / Claim / Premise | MajorClaim |

### Output — Relations CSV

Separator: `;`

| Column | Description | Example |
|--------|-------------|---------|
| text_id | Source document ID | AAEC_004 |
| source_tokens | Text of the source (premise) | "Firstly, it is an undeniable..." |
| target_tokens | Text of the target (conclusion) | "To conclude, I strongly believe..." |
| labels | support / attack | support |

### Output — Graph Images (optional)

When `--graph-image` is passed, PNG files are saved to `output/graphs/graph_{text_id}.png`.  
The visualization is a **tree layout** with:
- Root (MajorClaim) at top, Claims in the middle, Premises at the bottom
- Solid green edges = support, dashed red edges = attack
- Disconnected nodes shown in grey with ⊘ marker

---

## Project Structure

```
src/
├── config.py              # Pydantic Settings (env vars, paths, model config)
├── main.py                # CLI entry point (argparse)
├── pipeline.py            # Orchestrates CSV processing + MLflow tracking
├── logging_config.py      # structlog setup
├── utils.py               # Shared parsing utilities (parse_answer_ids, etc.)
│
├── graph/
│   ├── state.py           # WorkflowState TypedDict (LangGraph state)
│   └── workflow.py        # LangGraph StateGraph — 5-node pipeline
│
├── tasks/
│   ├── identification.py  # Step 1: Component identification
│   ├── conclusion.py      # Step 2: Conclusion extraction
│   ├── relations.py       # Step 3: Recursive BFS relation extraction
│   └── unvisited.py       # Step 4: Unvisited premise handling
│
├── models/
│   ├── components.py      # ArgumentComponent + ArgumentRelation (Pydantic)
│   └── graph.py           # ArgumentGraph (DAG ops, labels, visualization)
│
├── llm/
│   ├── client.py          # OpenAI API wrapper (generate, structured, thinking)
│   ├── prompts.py         # All prompt templates
│   ├── prompt_manager.py  # Centralized prompt access
│   └── structured_models.py  # Pydantic models for structured LLM output
│
├── evaluation/
│   └── metrics.py         # Compare predicted vs golden standard
│
└── export/
    └── golden_standard.py # Export graphs to semicolon-separated CSVs
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM API | `openai` SDK | Direct API calls (chat completions + structured output) |
| Workflow | `langgraph` | State machine orchestration for the 5-step pipeline |
| Data models | `pydantic` / `pydantic-settings` | Typed models, config management |
| Graph algorithms | `networkx` | DAG validation, transitive reduction, cycle detection |
| Data I/O | `polars` | CSV reading/writing |
| Visualization | `matplotlib` (optional) | Tree-layout graph images |
| Logging | `structlog` | Structured JSON logging |
| Experiment tracking | `mlflow` (optional) | Parameter/metric/artifact tracking |

---

## CLI Usage

```bash
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC \
  --limit 5 \
  --graph-image \
  --golden-components "data/Golden Standard/components_AAEC.csv" \
  --golden-relations "data/Golden Standard/relations_AAEC.csv"
```

| Flag | Description |
|------|-------------|
| `--input` | Input CSV (required) |
| `--output-prefix` | Prefix for output files (default: "output") |
| `--limit` | Process only first N texts |
| `--graph-image` | Save PNG graph visualizations |
| `--golden-components` | Golden standard for evaluation |
| `--golden-relations` | Golden standard for evaluation |
| `--run-name` | Custom MLflow run name |
| `--no-mlflow` | Disable MLflow tracking |
