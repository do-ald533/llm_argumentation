# LLM Argumentation Structuring

Automatically extract and structure argumentative components from text using Large Language Models.

## What It Does

Given an argumentative text, the system:
1. Identifies argumentative components (premises, claims, conclusions)
2. Classifies each component by type (MajorClaim, Claim, Premise)
3. Extracts support/attack relations between components
4. Outputs structured CSV files for analysis

## Requirements

- Python 3.10+
- OpenAI API key (or DeepSeek API key)

## Installation

```bash
# Install dependencies
pip install -e .

# Install MLflow for experiment tracking
pip install mlflow
```

## Configuration

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Optional (defaults shown)
OPENAI_MODEL=gpt-4o-mini
TEMPERATURE=0.0
ENABLE_MLFLOW=true
PROMPT_VERSION=1.0
```

See `.env.example` for all available options.

## Usage

### Basic Usage

```bash
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC
```

This will create:
- `output/components_AAEC.csv` - Extracted components
- `output/relations_AAEC.csv` - Relations between components

### With Evaluation

Compare outputs against golden standard:

```bash
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC \
  --golden-components data/Golden\ Standard/components_AAEC.csv \
  --golden-relations data/Golden\ Standard/relations_AAEC.csv
```

### Test Run

Process only first 5 texts:

```bash
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix test \
  --limit 5
```

### Disable Experiment Tracking

```bash
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC \
  --no-mlflow
```

## Input Format

CSV file with two columns:
- `text_id`: Unique identifier
- `text_tokens`: The argumentative text to analyze

Example:
```csv
text_id,text_tokens
1,"Capital punishment should be abolished because it violates human rights."
```

## Output Format

### Components CSV
```csv
text_id,component_tokens,labels
1,"Capital punishment should be abolished",MajorClaim
1,"it violates human rights",Premise
```

### Relations CSV
```csv
text_id,source_tokens,target_tokens,labels
1,"it violates human rights","Capital punishment should be abolished",support
```

## Experiment Tracking

View experiment results in MLflow UI:

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

MLflow tracks:
- Parameters (model, temperature, prompt version)
- Metrics (precision, recall, F1 scores)
- Artifacts (prompts used, output files)

See `MLFLOW_GUIDE.md` for detailed experiment tracking documentation.

## Project Structure

```
src/
├── main.py              # Entry point
├── config.py            # Configuration
├── pipeline.py          # Main pipeline
├── graph/               # Workflow orchestration
├── llm/                 # LLM client and prompts
├── models/              # Data models
├── tasks/               # Task implementations
├── evaluation/          # Evaluation metrics
└── export/              # CSV export utilities
```

## Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_MODEL` | `gpt-4o-mini` | Model to use |
| `TEMPERATURE` | `0.0` | Generation temperature |
| `MAX_RETRIES` | `3` | API retry attempts |
| `ENABLE_MLFLOW` | `true` | Enable experiment tracking |
| `PROMPT_VERSION` | `1.0` | Version tag for prompts |

## Common Commands

```bash
# Run on AAEC dataset
python -m src.main --input data/Input/texts_AAEC.csv --output-prefix AAEC

# Run on AbstRCT dataset
python -m src.main --input data/Input/texts_AbstRCT.csv --output-prefix AbstRCT

# Quick test with 5 texts
python -m src.main --input data/Input/texts_AAEC.csv --output-prefix test --limit 5

# Custom run name for MLflow
python -m src.main --input data/Input/texts_AAEC.csv --output-prefix AAEC --run-name "experiment-v1"

# View MLflow UI
mlflow ui --port 5000
```

## Troubleshooting

**"openai_api_key: field required"**
- Create `.env` file with `OPENAI_API_KEY=your-key`

**"No such file or directory: data/Input/texts_AAEC.csv"**
- Ensure input file exists at specified path
- Use absolute path or run from project root

**"Rate limit exceeded"**
- Script automatically retries with 60s delay
- Check API quota on OpenAI dashboard

## Documentation

- `MLFLOW_GUIDE.md` - Detailed experiment tracking guide
- `.vscode/PROJECT_CONTEXT.md` - Complete project documentation for future sessions

## License

See LICENSE file for details.
