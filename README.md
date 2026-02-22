# NLP Coursework — Patronizing & Condescending Language (PCL) Detection

This repository contains my NLP coursework project for detecting **Patronizing and Condescending Language (PCL)** in texts. The project explores both:

- **Transformer fine-tuning** (RoBERTa)
- **Feature-based ML pipeline design** (TF-IDF + linguistic features + CatBoost)

As a result of the investigations, an ensemble model is created from the fine-tuned RoBERTa and ML-pipeline models. 

The main target used throughout is a **binary label**:
- `labels = 1` if `PCL_category >= 2`
- `labels = 0` otherwise

---

## Repository contents

### Notebooks
- **`EDA.ipynb`** — Exploratory data analysis and dataset inspection.
- **`train_val_ds_split.ipynb`** — Train/validation splitting utilities for given validation indices.
- **`roberta_model.ipynb`** — Fine-tuning `roberta-base` for binary PCL classification (Hugging Face Trainer).
- **`pipeline_and_ensemble_models.ipynb`** — Feature extraction + classical pipeline model and ensemble logic.
- **`local_evaluation.ipynb`** — Run local evaluation.

### Data & outputs
- **`data/`** — TSV datasets (train/val/test).
- **`output/`** — Saved predictions, logs, or artifacts produced by the notebooks.

### Report
- **`NLP Coursework - Report.pdf`** — Spec question answers.

### Model artifacts
- **`models_link.txt`** — Link to downloaded/saved model artifacts (Google Drive folder).
