```markdown
# Financial Anomaly Detection Toolkit

A robust, modular Python library for loading, validating, evaluating, and explaining anomaly detection workflows in financial and transactional datasets. 

Designed with a strong focus on **temporal integrity**, **rare-event evaluation**, and **explainable AI**.

## ✨ Features

- **Versatile Data Loading:** Read from CSV, Parquet, or SQL (PostgreSQL, ClickHouse) with automatic UTC timezone normalization.
- **Temporal Safety:** Strict time-based train/val/test splitting and temporal leakage detection.
- **Market Data Processing:** Resample tick data to OHLCV bars, filter market hours, and detect price gaps.
- **Rare-Event Evaluation:** Specialized metrics including Precision@K, PR AUC, Cost-Sensitive Matrices, Detection Delay, and VaR Reduction.
- **Explainability (XAI):** SHAP values, LSTM gradient saliency, counterfactuals, and Isolation Forest rule extraction.
- **Automated Reporting:** Generate structured JSON and HTML anomaly reports.
- **Type-Safe Configuration:** Pydantic-powered settings management via `.env` or YAML.

## 📂 Project Structure

```text
├── config.py        # Pydantic settings (DB, models, calibration, logging)
├── data_loader.py   # Transaction & Market data loaders with validation
├── evaluator.py     # Rare-event evaluation metrics and analysis
├── explainer.py     # XAI utilities and HTML/JSON report generation
└── __init__.py      # Package marker
```

## 📦 Installation

**Core Dependencies:**
```bash
pip install pandas numpy scikit-learn pydantic pydantic-settings pyyaml
```

**Optional Dependencies (based on usage):**
```bash
# For SQL databases
pip install sqlalchemy psycopg2-binary

# For Parquet files
pip install pyarrow

# For Explainability (SHAP & LSTM Saliency)
pip install shap tensorflow

# For Evaluation Plotting
pip install matplotlib
```

## ⚙️ Configuration

Configuration is managed via Pydantic in `config.py`. Settings can be loaded from environment variables (prefixed with `ANOMALY_`) or a `.env` file. Nested variables use double underscores (`__`).

**Example `.env`:**
```env
ANOMALY_ENVIRONMENT=production
ANOMALY_POSTGRES__HOST=db.internal
ANOMALY_POSTGRES__PASSWORD=supersecret
ANOMALY_IFOREST__CONTAMINATION=0.005
ANOMALY_CALIBRATION__MAX_ALERTS_PER_DAY=50
```

**Load in Python:**
```python
from config import get_settings

settings = get_settings()
print(settings.postgres.sqlalchemy_url)
```

## 🚀 Usage Guide

### 1. Data Loading & Temporal Splitting
Prevent data leakage with strict chronological splitting and UTC normalization.

```python
from data_loader import TransactionLoader

loader = TransactionLoader()
df = loader.load("postgresql+psycopg2://user:pass@localhost/transactions")

# Validate data quality
report = loader.validate(df)
print(f"Duplicate IDs: {report.duplicate_ids}")

# Strict temporal split (No shuffling)
train, val, test = loader.time_based_split(
    df, 
    train_end="2023-01-01", 
    val_end="2023-03-01"
)

# Check for temporal leakage
leakage = loader.detect_leakage(df, cutoff="2023-01-01")
```

### 2. Market Data Processing
Resample ticks and detect suspicious price gaps.

```python
from data_loader import MarketDataLoader

market_loader = MarketDataLoader()
ticks = market_loader.load("market_ticks.parquet")

# Resample to 1-minute OHLCV bars
bars = market_loader.resample_tick_data(ticks, freq="1min")

# Flag single-bar moves > 5%
bars = market_loader.detect_price_gaps(bars, threshold_pct=0.05)
```

### 3. Model Evaluation
Evaluate models using metrics tailored for imbalanced financial data.

```python
from evaluator import AnomalyEvaluator

evaluator = AnomalyEvaluator(top_k=100)

metrics = evaluator.compute_metrics(
    model_name="isolation_forest_v1",
    y_true=y_true,
    y_pred_scores=y_scores,
    y_pred_labels=y_preds
)

print(f"Precision@100: {metrics.precision_at_k:.4f}")
print(f"PR AUC: {metrics.pr_auc:.4f}")

# Cost-sensitive analysis
cost = evaluator.compute_confusion_matrix_cost_sensitive(
    y_true, y_preds, 
    cost_matrix={"fp_cost": 1.0, "fn_cost": 50.0}
)
```

### 4. Explainability & Reporting
Understand *why* an anomaly was flagged and generate reports for analysts.

```python
from explainer import AnomalyExplainer, ReportGenerator

explainer = AnomalyExplainer()

# SHAP explanation for tree models
shap_explanation = explainer.explain_tree_model(model, X_background, sample)

# Counterfactual: "What amount would NOT trigger the alert?"
cf = explainer.counterfactual_amount(detector, sample, amount_feature="amount")

# Generate Human-Readable Summary
summary = explainer.natural_language_summary(shap_explanation)

# Export to HTML/JSON
reporter = ReportGenerator(output_dir="reports")
reporter.generate_html(report_id="txn_9921", payload={"summary": summary, "score": 0.98})
```

## ⚠️ Important Notes

- **`schemas.py` Dependency:** The `evaluator.py` module imports `ModelMetrics` from an external `schemas` module. Ensure you have a `schemas.py` file in your Python path that defines the `ModelMetrics` dataclass/model before running the evaluator.
- **Timezones:** All timestamp operations strictly enforce UTC to prevent silent timezone bugs common in financial data.
```