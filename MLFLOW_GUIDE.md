# MLflow Experiment Tracking Guide

This guide explains how to use MLflow for tracking experiments with different prompts, models, and configurations.

## 🚀 Quick Start

### 1. Install MLflow

```bash
pip install mlflow
# Or if using pyproject.toml:
pip install -e .
```

### 2. Enable Tracking

MLflow is **enabled by default**. Check your `.env` file:

```bash
ENABLE_MLFLOW=true
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=argumentation-structuring
PROMPT_VERSION=1.0
```

### 3. Run Pipeline

```bash
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC \
  --run-name "gpt4-temp0-v1"
```

### 4. View Results

```bash
mlflow ui --port 5000
# Open browser: http://localhost:5000
```

---

## 📊 What Gets Tracked

### Parameters (Logged Automatically)
- `llm_model`: Model name (e.g., gpt-4o-mini)
- `temperature`: Generation temperature
- `max_retries`: API retry attempts
- `dataset`: Dataset prefix (AAEC, AbstRCT, etc.)
- `input_file`: Path to input CSV
- `total_texts`: Number of texts processed
- `prompt_version`: Version from config (update when prompts change!)
- `prompt_hash`: SHA256 hash of prompts.py file

### Metrics (Logged Automatically)
- `texts_processed`: Successfully processed texts
- `texts_failed`: Failed texts
- `processing_time_seconds`: Total time
- `avg_time_per_text`: Average processing time

**If golden standard provided:**
- `component_precision`, `component_recall`, `component_f1`
- `relation_precision`, `relation_recall`, `relation_f1`
- `component_predicted`, `component_gold`, `component_correct`
- `relation_predicted`, `relation_gold`, `relation_correct`

### Artifacts (Saved)
- `prompts/prompts.py`: Exact prompt file used
- `outputs/components_*.csv`: Generated components
- `outputs/relations_*.csv`: Generated relations

---

## 🎯 Common Use Cases

### Use Case 1: Testing Different Prompts

**Scenario:** You modified the identification prompt and want to compare results.

```bash
# 1. Update PROMPT_VERSION in .env
PROMPT_VERSION=1.1

# 2. Run with descriptive name
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC_v1.1 \
  --run-name "identification-prompt-v1.1" \
  --golden-components data/Golden\ Standard/components_AAEC.csv \
  --golden-relations data/Golden\ Standard/relations_AAEC.csv

# 3. Compare in MLflow UI
mlflow ui --port 5000
# Sort by component_f1 to see which prompt performed better
```

### Use Case 2: Model Comparison

**Scenario:** Compare GPT-4 vs. GPT-4o-mini

```bash
# Run 1: GPT-4o-mini
OPENAI_MODEL=gpt-4o-mini python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC_mini \
  --run-name "gpt4-mini-baseline"

# Run 2: GPT-4o
OPENAI_MODEL=gpt-4o python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC_full \
  --run-name "gpt4-full-baseline"

# Compare in UI (metrics, processing time, cost)
```

### Use Case 3: Temperature Tuning

**Scenario:** Test different temperature values

```bash
# Temperature 0.0 (deterministic)
TEMPERATURE=0.0 python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC_temp0 \
  --run-name "temp-0.0"

# Temperature 0.3
TEMPERATURE=0.3 python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC_temp03 \
  --run-name "temp-0.3"

# Temperature 0.7
TEMPERATURE=0.7 python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC_temp07 \
  --run-name "temp-0.7"
```

### Use Case 4: Small Test Before Full Run

```bash
# Test with first 5 texts
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix test_5 \
  --limit 5 \
  --run-name "quick-test-5-texts"

# Check results, then run full dataset
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix AAEC_full \
  --run-name "full-run-after-test"
```

---

## 🔄 Rollback to Previous Version

**Scenario:** New prompt performed worse, revert to old version.

### Option 1: Download from MLflow UI

1. Open MLflow UI: `mlflow ui --port 5000`
2. Find the best-performing run (sort by F1 score)
3. Click on the run → "Artifacts" tab
4. Download `prompts/prompts.py`
5. Copy to `src/llm/prompts.py`
6. Update `PROMPT_VERSION` in `.env`

### Option 2: Git-based Versioning

```bash
# Tag current prompts before changing
git add src/llm/prompts.py
git commit -m "prompts: v1.0 baseline"
git tag prompt-v1.0

# Make changes, test...

# Rollback if needed
git checkout prompt-v1.0 -- src/llm/prompts.py
```

---

## 📈 MLflow UI Tips

### Comparing Runs

1. **Select Multiple Runs** (checkboxes)
2. **Click "Compare"** button
3. **View side-by-side:**
   - Parameter differences
   - Metric charts
   - Artifact diffs

### Sorting & Filtering

- **Sort by F1:** Click `relation_f1` column header
- **Filter by model:** Use search box: `params.llm_model = "gpt-4o"`
- **Filter by date:** Use date range picker

### Key Metrics to Watch

- **`component_f1`**: How well components are extracted
- **`relation_f1`**: How well relations are identified (usually lower)
- **`processing_time_seconds`**: Cost proxy (API calls)
- **`texts_failed`**: Robustness indicator

---

## 🔧 Advanced: Custom Experiments

### Disable Tracking (for quick tests)

```bash
python -m src.main \
  --input data/Input/texts_AAEC.csv \
  --output-prefix test \
  --no-mlflow
```

### Custom Tracking URI (remote server)

```bash  
# In .env
MLFLOW_TRACKING_URI=http://remote-server:5000
```

### Multiple Experiments

```bash
# Group related runs
MLFLOW_EXPERIMENT_NAME=prompt-engineering-phase1
python -m src.main ...

# Start new experiment
MLFLOW_EXPERIMENT_NAME=model-comparison
python -m src.main ...
```

---

## 💾 Data Organization

```
project/
├── mlruns/                    # MLflow tracking data (auto-generated)
│   ├── 0/                     # Default experiment
│   │   ├── run-id-1/          # Individual runs
│   │   │   ├── artifacts/     # Saved files
│   │   │   ├── params/        # Parameters
│   │   │   ├── metrics/       # Metrics
│   │   │   └── meta.yaml      # Metadata
│   ├── 1/                     # Custom experiment
│   └── ...
├── output/                    # Generated CSVs (outside tracking)
└── data/Golden Standard/      # Reference data
```

**Note:** `mlruns/` is in `.gitignore` - do not commit to git!

---

## 🎓 Best Practices

### 1. **Version Your Prompts**
Update `PROMPT_VERSION` in `.env` when you change prompts:
```bash
PROMPT_VERSION=1.1  # Changed identification examples
PROMPT_VERSION=1.2  # Added relation extraction guidance
PROMPT_VERSION=2.0  # Major restructure
```

### 2. **Use Descriptive Run Names**
```bash
# Good
--run-name "gpt4-temp0-prompt-v1.2-AAEC"

# Bad
--run-name "test1"
```

### 3. **Always Include Golden Standard**
```bash
--golden-components data/Golden\ Standard/components_AAEC.csv \
--golden-relations data/Golden\ Standard/relations_AAEC.csv
```

### 4. **Test Small First**
```bash
# Test with --limit 5 before full run
--limit 5 --run-name "test-5-texts"
```

### 5. **Document in Run Name**
```bash
--run-name "gpt4-temp0.3-added-CoT-in-relations"
# CoT = Chain of Thought
```

---

## 🚨 Troubleshooting

### "mlflow: command not found"
```bash
pip install mlflow
```

### "No runs found"
Check tracking URI:
```bash
# In Python
import mlflow
print(mlflow.get_tracking_uri())  # Should be ./mlruns
```

### "Permission denied: mlruns/"
```bash
chmod -R u+w mlruns/
```

### MLflow UI won't start
```bash
# Try different port
mlflow ui --port 5001

# Or specify host
mlflow ui --host 0.0.0.0 --port 5000
```

### Large artifact warnings
Artifacts (prompts, CSVs) are copied to each run. This is intentional for reproducibility.

---

## 📚 Further Reading

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking APIs](https://mlflow.org/docs/latest/tracking.html)
- [Experiment Comparison](https://mlflow.org/docs/latest/tracking.html#comparing-runs)

---

**Happy Experimenting! 🚀**
