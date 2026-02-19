# LLM Argumentation Structuring

Automated extraction and structuring of argumentative components from text using Large Language Models powered by LangGraph workflows.

## Overview

This system analyzes argumentative texts and extracts structured argumentation graphs consisting of:
- Argumentative components (MajorClaim, Claim, Premise)
- Relations between components (support, attack)

The pipeline uses GPT-4o-mini with structured outputs and a multi-stage LangGraph workflow to ensure accurate component identification and classification.

## How It Works

The system processes each text through a 5-stage LangGraph workflow:

1. **Identification**: Extract all argumentative components from text
2. **Conclusion Extraction**: Identify the main conclusion/major claim
3. **Classification**: Classify each component (MajorClaim, Claim, Premise)
4. **Relation Extraction**: Identify support/attack relations between components
5. **Finalization**: Combine results into structured output

Each stage uses structured outputs with Pydantic models to ensure type-safe, validated results.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Create .env file with your API key
echo "OPENAI_API_KEY=your-key-here" > .env

# Process a dataset
python -m src.main --input data/Input/texts_AAEC.csv --output-prefix AAEC
```

## Batch Processing

For large datasets, use the batch processing scripts:

```bash
# Create batches of 10 texts
python scripts/create_batches.py --input data/Input/texts_AbstRCT.csv --batch-size 10 --output-dir data/batches

# Process batches (sequential or parallel)
python scripts/process_batches.py data/batches --mode sequential

# Merge results
python scripts/merge_results.py --input-dir output/batches --output-dir output --prefix AbstRCT
```

## Configuration

Create a `.env` file:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
TEMPERATURE=0.0
ENABLE_MLFLOW=true
```

## Input/Output Format

**Input CSV** (semicolon-separated):
```csv
text_id;text_tokens
AAEC_001;"Text containing argumentative content..."
```

**Output Components CSV**:
```csv
text_id;component_tokens;labels
AAEC_001;"Capital punishment should be abolished";MajorClaim
AAEC_001;"it violates human rights";Premise
```

**Output Relations CSV**:
```csv
text_id;source_tokens;target_tokens;labels
AAEC_001;"it violates human rights";"Capital punishment should be abolished";support
```

## Evaluation

Compare outputs against golden standards:

```bash
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC \
  --golden-components data/Golden\ Standard/components_AAEC.csv \
  --golden-relations data/Golden\ Standard/relations_AAEC.csv
```

Metrics include precision, recall, and F1 scores for both components and relations.

## Experiment Tracking

MLflow automatically tracks experiments:

```bash
mlflow ui --port 5000
```

Tracked data includes:
- Model parameters and configuration
- Evaluation metrics
- Input/output artifacts
- Processing time and API calls

## Project Structure

```
src/
├── main.py              # Entry point
├── pipeline.py          # Main processing pipeline
├── graph/               # LangGraph workflow definition
├── llm/                 # LLM client and prompt management
├── models/              # Pydantic data models
├── tasks/               # Task implementations (identify, classify, extract)
├── evaluation/          # Evaluation metrics
└── export/              # CSV export utilities
scripts/
├── create_batches.py    # Split datasets into batches
├── process_batches.py   # Process batches sequentially/parallel
└── merge_results.py     # Merge batch results
```

## Requirements

- Python 3.10+
- OpenAI API key
- Dependencies listed in pyproject.toml
