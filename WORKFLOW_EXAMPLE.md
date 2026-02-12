# Complete Workflow Example

This document shows a step-by-step example of how the argumentation structuring pipeline works, from input to output.

---

## Input

### File: `data/Input/texts_AAEC.csv`

```csv
,text_id,text_tokens
1065,AAEC_004,"International tourism is now more common than ever before. The last 50 years have seen a significant increase in the number of tourist traveling worldwide. While some might think the tourism bring large profit for the destination countries, I would contend that this industry has affected the cultural attributes and damaged the natural environment of the tourist destinations..."
```

**Command to run:**
```bash
python src/main.py \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC \
  --limit 1 \
  --golden-components data/Golden\ Standard/components_AAEC.csv \
  --golden-relations data/Golden\ Standard/relations_AAEC.csv
```

---

## Processing Steps

### Step 1: **Load Configuration**
From `.env` file:
```env
OPENAI_MODEL=gpt-4o-mini
TEMPERATURE=0.0
ENABLE_MLFLOW=true
```

### Step 2: **Read Input CSV**
```python
df = pl.read_csv("data/Input/texts_AAEC.csv")
# Extracts:
# - text_id: "AAEC_004"
# - text: "International tourism is now more common..."
```

### Step 3: **Initialize Workflow**
Creates LangGraph workflow with 5 nodes:
1. identify_components
2. extract_conclusion
3. classify_components
4. extract_relations
5. finalize

---

## Workflow Execution (Text: AAEC_004)

### **Node 1: Identify Components**

**LLM Prompt:**
```
Analyze the following text and identify all argumentative components.
List each component as a numbered item.

Text: "International tourism is now more common..."
```

**LLM Response:**
```
1 - International tourism is now more common than ever before
2 - this industry has affected the cultural attributes and damaged the natural environment of the tourist destinations
3 - the tourism bring large profit for the destination countries
4 - international tourism can create negative impacts on the destination countries
5 - tourists from different cultures will probably cause changes to the cultural identity of the tourist destinations
6 - Take Thailand for example, in the Vietnam War, many American soldiers came to Thailand for a break and involved in sexual and drug activities...
7 - This was due to the lack of adequate controls by authorities and lead to a bad image of Thailand tourism
8 - tourism increasingly create harms to the natural habitats of the destination appeals
9 - As the Australia's Great Barrier Reef has shown, the billion visitors per annum has generated immense destruction...
10 - many marine lives have been endangered, in the extremes part of the reef become uninhabitable for these marine species
11 - tourism has threatened the nature environments
```

**Internal State:**
```python
components = {
    1: ArgumentComponent(id=1, text="International tourism is now...", text_id="AAEC_004", label="Premise"),
    2: ArgumentComponent(id=2, text="this industry has affected...", text_id="AAEC_004", label="Premise"),
    3: ArgumentComponent(id=3, text="the tourism bring large profit...", text_id="AAEC_004", label="Premise"),
    # ... etc for all 11 components
}
```

---

### **Node 2: Extract Conclusion**

**LLM Prompt:**
```
Which component represents the main conclusion of the argument?
Components:
1 - International tourism is now more common...
2 - this industry has affected the cultural attributes...
3 - the tourism bring large profit...
...
```

**LLM Response:**
```
Component 2
```

**State Update:**
```python
conclusion_id = 2
```

---

### **Node 3: Classify Components**

For the **conclusion** (id=2):
```python
components[2].label = "MajorClaim"  # Automatically set
```

For **each other component**, LLM is asked:

**LLM Prompt (example for component 4):**
```
Classify the following argumentative component as either "Claim" or "Premise".

Component: "international tourism can create negative impacts on the destination countries"

Context:
All components:
1 - International tourism is now more common...
2 - this industry has affected the cultural attributes... [MajorClaim]
3 - the tourism bring large profit...
...
```

**LLM Response:**
```
Claim
```

**State Update:**
```python
components[4].label = "Claim"
components[5].label = "Premise"
components[6].label = "Premise"
components[11].label = "Claim"
# etc...
```

---

### **Node 4: Extract Relations**

Starting from **conclusion (id=2)**, breadth-first traversal:

**Iteration 1: Target = Component 2 (MajorClaim)**

**LLM Prompt:**
```
Which components SUPPORT component 2: "this industry has affected the cultural attributes and damaged the natural environment"?

Available components:
3 - the tourism bring large profit for the destination countries
4 - international tourism can create negative impacts on the destination countries
11 - tourism has threatened the nature environments
...
```

**LLM Response:**
```
Components 4 and 11 support component 2
```

**LLM Prompt (Attack):**
```
Which components ATTACK component 2?
```

**LLM Response:**
```
Component 3 attacks component 2
```

**Relations Created:**
```python
relations = [
    ArgumentRelation(source_id=4, target_id=2, text_id="AAEC_004", relation_type="support"),
    ArgumentRelation(source_id=11, target_id=2, text_id="AAEC_004", relation_type="support"),
    ArgumentRelation(source_id=3, target_id=2, text_id="AAEC_004", relation_type="attack"),
]
```

**Iteration 2: Target = Component 4 (Claim)**

**LLM finds:**
```
Components 5, 6, 7 support component 4
```

**Relations Created:**
```python
relations.append(ArgumentRelation(source_id=5, target_id=4, relation_type="support"))
relations.append(ArgumentRelation(source_id=6, target_id=4, relation_type="support"))
relations.append(ArgumentRelation(source_id=7, target_id=4, relation_type="support"))
```

**Continues for all components...**

---

### **Node 5: Finalize**

Marks workflow as complete.

---

## 📊 Create ArgumentGraph

```python
graph = ArgumentGraph(
    text_id="AAEC_004",
    text="International tourism is now more common...",
    components={
        1: ArgumentComponent(id=1, text="International tourism is...", label="Premise"),
        2: ArgumentComponent(id=2, text="this industry has affected...", label="MajorClaim"),
        3: ArgumentComponent(id=3, text="the tourism bring large...", label="Claim"),
        4: ArgumentComponent(id=4, text="international tourism can create...", label="Claim"),
        5: ArgumentComponent(id=5, text="tourists from different cultures...", label="Premise"),
        # ... etc
    },
    relations=[
        ArgumentRelation(source_id=4, target_id=2, relation_type="support"),
        ArgumentRelation(source_id=11, target_id=2, relation_type="support"),
        ArgumentRelation(source_id=3, target_id=2, relation_type="attack"),
        # ... etc
    ],
    conclusion_id=2
)
```

---

## 📤 Export to CSV (Golden Standard Format)

### **Export Components**

The `graph.to_golden_standard_components()` method converts:

```python
def to_golden_standard_components(self) -> list[dict]:
    return [comp.to_golden_standard() for comp in self.components.values()]
```

Each component's `to_golden_standard()` returns:
```python
{
    "text_id": "AAEC_004",
    "component_tokens": "this industry has affected the cultural attributes...",
    "labels": "MajorClaim"
}
```

### **Output File: `output/components_AAEC.csv`**

```csv
,text_id,component_tokens,labels
0,AAEC_004,"International tourism is now more common than ever before",Premise
1,AAEC_004,"this industry has affected the cultural attributes and damaged the natural environment of the tourist destinations",MajorClaim
2,AAEC_004,"the tourism bring large profit for the destination countries",Claim
3,AAEC_004,"international tourism can create negative impacts on the destination countries",Claim
4,AAEC_004,"tourists from different cultures will probably cause changes to the cultural identity of the tourist destinations",Premise
5,AAEC_004,"Take Thailand for example, in the Vietnam War, many American soldiers came to Thailand for a break and involved in sexual and drug activities, these huge demands caused many local businesses opened and expanded, even illegally involved in under-age prostitutes to maximize their profits",Premise
6,AAEC_004,"This was due to the lack of adequate controls by authorities and lead to a bad image of Thailand tourism",Premise
7,AAEC_004,"tourism increasingly create harms to the natural habitats of the destination appeals",Premise
8,AAEC_004,"As the Australia's Great Barrier Reef has shown, the billion visitors per annum has generated immense destruction to this nature wonder, namely breaking the corals caused by walking or throwing boat's anchors, dropping fuel and other sorts of pollutions",Premise
9,AAEC_004,"many marine lives have been endangered, in the extremes part of the reef become uninhabitable for these marine species",Premise
10,AAEC_004,"tourism has threatened the nature environments",Claim
```

### **Export Relations**

The `graph.to_golden_standard_relations()` method converts:

```python
def to_golden_standard_relations(self) -> list[dict]:
    return [rel.to_golden_standard(self.components) for rel in self.relations]
```

Each relation's `to_golden_standard()` returns:
```python
{
    "text_id": "AAEC_004",
    "source_tokens": components[source_id].text,  # Actual text, not ID!
    "target_tokens": components[target_id].text,  # Actual text, not ID!
    "labels": "support"
}
```

### **Output File: `output/relations_AAEC.csv`**

```csv
,text_id,source_tokens,target_tokens,labels
0,AAEC_004,"international tourism can create negative impacts on the destination countries","this industry has affected the cultural attributes and damaged the natural environment of the tourist destinations",support
1,AAEC_004,"tourism has threatened the nature environments","this industry has affected the cultural attributes and damaged the natural environment of the tourist destinations",support
2,AAEC_004,"the tourism bring large profit for the destination countries","this industry has affected the cultural attributes and damaged the natural environment of the tourist destinations",attack
3,AAEC_004,"tourists from different cultures will probably cause changes to the cultural identity of the tourist destinations","international tourism can create negative impacts on the destination countries",support
4,AAEC_004,"Take Thailand for example, in the Vietnam War, many American soldiers came to Thailand for a break and involved in sexual and drug activities, these huge demands caused many local businesses opened and expanded, even illegally involved in under-age prostitutes to maximize their profits","international tourism can create negative impacts on the destination countries",support
5,AAEC_004,"This was due to the lack of adequate controls by authorities and lead to a bad image of Thailand tourism","international tourism can create negative impacts on the destination countries",support
6,AAEC_004,"tourism increasingly create harms to the natural habitats of the destination appeals","tourism has threatened the nature environments",support
7,AAEC_004,"many marine lives have been endangered, in the extremes part of the reef become uninhabitable for these marine species","tourism has threatened the nature environments",support
8,AAEC_004,"As the Australia's Great Barrier Reef has shown, the billion visitors per annum has generated immense destruction to this nature wonder, namely breaking the corals caused by walking or throwing boat's anchors, dropping fuel and other sorts of pollutions","many marine lives have been endangered, in the extremes part of the reef become uninhabitable for these marine species",support
```

---

## 📈 Evaluation (Optional)

If golden standard files are provided, the pipeline compares predictions vs. ground truth:

```python
evaluate_against_golden_standard(
    predicted_components="output/components_AAEC.csv",
    golden_components="data/Golden Standard/components_AAEC.csv",
    predicted_relations="output/relations_AAEC.csv",
    golden_relations="data/Golden Standard/relations_AAEC.csv"
)
```

**Metrics calculated:**
- Component extraction: Precision, Recall, F1
- Component classification: Accuracy, per-class metrics
- Relation extraction: Precision, Recall, F1

**Logged to MLflow** for experiment tracking.

---

## Key Points

### **Internal vs. External Representation**

**Internal (during processing):**
- Components stored with **integer IDs** (1, 2, 3, ...)
- Relations reference **component IDs** (source_id=4, target_id=2)
- Efficient for graph traversal and LLM prompting

**External (CSV output):**
- Components export **actual text** in `component_tokens` column
- Relations export **actual text** in `source_tokens` and `target_tokens` columns
- Human-readable format matching golden standard
- No IDs visible in final CSV (though polars adds row indices)

### **Why This Design?**

1. **Human Readability**: Annotators can easily read and verify results
2. **Direct Comparison**: Can diff against golden standard text
3. **No ID Mapping Needed**: Text matching is more robust than ID matching
4. **Standard Format**: Matches academic datasets (AbstRCT, AAEC)

---

## 🚀 Running the Pipeline

### **Setup:**
```bash
# Copy and configure environment
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY
```

### **Process one text (for testing):**
```bash
python src/main.py \
  --input data/Input/texts_AAEC.csv \
  --output-prefix test \
  --limit 1 \
  --no-mlflow
```

### **Process entire dataset:**
```bash
python src/main.py \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC \
  --golden-components data/Golden\ Standard/components_AAEC.csv \
  --golden-relations data/Golden\ Standard/relations_AAEC.csv
```

### **Check results:**
```bash
ls output/
# components_AAEC.csv
# relations_AAEC.csv

head output/components_AAEC.csv
head output/relations_AAEC.csv
```

### **View MLflow results:**
```bash
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000 in browser
```

---

## 📁 Complete File Structure

```
llm_argumentation_structuring/
├── data/
│   ├── Input/
│   │   └── texts_AAEC.csv              # Input texts
│   └── Golden Standard/
│       ├── components_AAEC.csv         # Ground truth components
│       └── relations_AAEC.csv          # Ground truth relations
├── output/
│   ├── components_AAEC.csv             # Generated components (matches format)
│   └── relations_AAEC.csv              # Generated relations (matches format)
├── mlruns/                             # MLflow experiment tracking
├── src/
│   ├── main.py                         # Entry point
│   ├── pipeline.py                     # Orchestrates processing
│   ├── graph/workflow.py               # LangGraph workflow
│   ├── tasks/                          # Individual LLM tasks
│   │   ├── identification.py
│   │   ├── classification.py
│   │   ├── relations.py
│   │   └── conclusion.py
│   ├── models/                         # Data models
│   │   ├── components.py               # ArgumentComponent, ArgumentRelation
│   │   └── graph.py                    # ArgumentGraph
│   ├── export/
│   │   └── golden_standard.py          # CSV export logic
│   ├── evaluation/
│   │   └── metrics.py                  # Evaluation metrics
│   └── llm/
│       ├── client.py                   # LLM client wrapper
│       └── prompts.py                  # Prompt templates
└── .env                                # Configuration (API keys, model settings)
```
