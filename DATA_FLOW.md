# Data Flow: Internal IDs → Text-Based Output

This document illustrates how component IDs are used internally but converted to text for export.

---

## Internal Processing (Using IDs)

### **After Component Identification:**

```python
components = {
    1: ArgumentComponent(
        id=1,
        text="International tourism is now more common",
        text_id="AAEC_004",
        label="Premise"
    ),
    2: ArgumentComponent(
        id=2,
        text="this industry has affected the cultural attributes and damaged the natural environment",
        text_id="AAEC_004",
        label="MajorClaim"
    ),
    3: ArgumentComponent(
        id=3,
        text="the tourism bring large profit for the destination countries",
        text_id="AAEC_004",
        label="Claim"
    ),
    4: ArgumentComponent(
        id=4,
        text="international tourism can create negative impacts",
        text_id="AAEC_004",
        label="Claim"
    )
}

conclusion_id = 2  # Reference by ID
```

### **After Relation Extraction:**

```python
relations = [
    ArgumentRelation(
        source_id=4,           # ← Uses ID
        target_id=2,           # ← Uses ID
        text_id="AAEC_004",
        relation_type="support"
    ),
    ArgumentRelation(
        source_id=3,           # ← Uses ID
        target_id=2,           # ← Uses ID
        text_id="AAEC_004",
        relation_type="attack"
    )
]
```

**Why IDs internally?**
- Efficient graph traversal
- Easy to reference in prompts ("Which components support component 2?")
- Simple to update labels without text matching
- Standard graph data structure

---

## Export Conversion (IDs → Text)

### **Component Export:**

```python
# Method in ArgumentComponent class
def to_golden_standard(self) -> dict:
    return {
        "text_id": self.text_id,                    # Keep text_id
        "component_tokens": self.text,              # Export FULL TEXT (not ID!)
        "labels": self.label                        # Keep label
    }
```

**Result:**
```python
[
    {
        "text_id": "AAEC_004",
        "component_tokens": "International tourism is now more common",
        "labels": "Premise"
    },
    {
        "text_id": "AAEC_004",
        "component_tokens": "this industry has affected the cultural attributes...",
        "labels": "MajorClaim"
    },
    {
        "text_id": "AAEC_004",
        "component_tokens": "the tourism bring large profit...",
        "labels": "Claim"
    },
    {
        "text_id": "AAEC_004",
        "component_tokens": "international tourism can create negative impacts",
        "labels": "Claim"
    }
]
```

### **Relation Export:**

```python
# Method in ArgumentRelation class
def to_golden_standard(self, components: dict[int, ArgumentComponent]) -> dict:
    return {
        "text_id": self.text_id,
        "source_tokens": components[self.source_id].text,  # ← Lookup text by ID
        "target_tokens": components[self.target_id].text,  # ← Lookup text by ID
        "labels": self.relation_type
    }
```

**Conversion:**
```python
# Internal representation (with IDs):
ArgumentRelation(source_id=4, target_id=2, relation_type="support")

# After conversion (with TEXT):
{
    "text_id": "AAEC_004",
    "source_tokens": "international tourism can create negative impacts",
    "target_tokens": "this industry has affected the cultural attributes...",
    "labels": "support"
}
```

**Result for all relations:**
```python
[
    {
        "text_id": "AAEC_004",
        "source_tokens": "international tourism can create negative impacts",
        "target_tokens": "this industry has affected the cultural attributes...",
        "labels": "support"
    },
    {
        "text_id": "AAEC_004",
        "source_tokens": "the tourism bring large profit...",
        "target_tokens": "this industry has affected the cultural attributes...",
        "labels": "attack"
    }
]
```

---

## Final CSV Output

### **components_AAEC.csv**
```csv
,text_id,component_tokens,labels
0,AAEC_004,"International tourism is now more common",Premise
1,AAEC_004,"this industry has affected the cultural attributes and damaged the natural environment",MajorClaim
2,AAEC_004,"the tourism bring large profit for the destination countries",Claim
3,AAEC_004,"international tourism can create negative impacts",Claim
```

**Note:** The first column is the Polars DataFrame index (auto-generated), NOT our internal component ID!

### **relations_AAEC.csv**
```csv
,text_id,source_tokens,target_tokens,labels
0,AAEC_004,"international tourism can create negative impacts","this industry has affected the cultural attributes and damaged the natural environment",support
1,AAEC_004,"the tourism bring large profit for the destination countries","this industry has affected the cultural attributes and damaged the natural environment",attack
```

**Note:** The first column is the Polars DataFrame index (auto-generated), NOT a relation ID!

---

## Format Validation

### **Golden Standard Format (Expected):**

**Components:**
- `text_id`: Document identifier
- `component_tokens`: **Full text of component**
- `labels`: MajorClaim | Claim | Premise

**Relations:**
- `text_id`: Document identifier
- `source_tokens`: **Full text of source component**
- `target_tokens`: **Full text of target component**
- `labels`: support | attack

### **Our Code Output:**
- Matches exactly! 
- Text-based exports via `to_golden_standard()` methods
- No IDs in final CSV (IDs only used internally)

---

## Code Locations

### **Export Logic:**

**File: `src/models/components.py`**
```python
class ArgumentComponent:
    def to_golden_standard(self) -> dict:
        """Convert to golden standard CSV format."""
        return {
            "text_id": self.text_id,
            "component_tokens": self.text,      # ← Text, not ID
            "labels": self.label
        }
```

**File: `src/models/components.py`**
```python
class ArgumentRelation:
    def to_golden_standard(self, components: dict[int, ArgumentComponent]) -> dict:
        """Convert to golden standard CSV format."""
        return {
            "text_id": self.text_id,
            "source_tokens": components[self.source_id].text,  # ← Resolve ID to text
            "target_tokens": components[self.target_id].text,  # ← Resolve ID to text
            "labels": self.relation_type
        }
```

### **Export Function:**

**File: `src/export/golden_standard.py`**
```python
def export_to_golden_standard(graphs: List[ArgumentGraph], output_dir: Path, prefix: str):
    all_components = []
    all_relations = []
    
    for graph in graphs:
        # Calls to_golden_standard() on each component/relation
        all_components.extend(graph.to_golden_standard_components())
        all_relations.extend(graph.to_golden_standard_relations())
    
    # Write to CSV
    pl.DataFrame(all_components).write_csv(f"components_{prefix}.csv")
    pl.DataFrame(all_relations).write_csv(f"relations_{prefix}.csv")
```

---

## Summary

| Aspect | Internal Processing | External Output |
|--------|-------------------|-----------------|
| **Component Reference** | Integer ID (1, 2, 3...) | Full text string |
| **Relation Source** | `source_id: int` | `source_tokens: str` |
| **Relation Target** | `target_id: int` | `target_tokens: str` |
| **Purpose** | Efficient graph operations | Human-readable, standard format |
| **Used For** | LLM prompts, traversal, updates | Evaluation, comparison, sharing |

**The code is already correctly implemented! No modifications needed.**
