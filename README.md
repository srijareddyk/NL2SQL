# NL2SQL Preprocessing Setup

This project uses the **WikiSQL** dataset to preprocess data for T5-based Text-to-SQL training.

## 1) Dataset

- Dataset: **WikiSQL**
- Source: official GitHub repository: [salesforce/WikiSQL](https://github.com/salesforce/WikiSQL)

### Download steps

```bash
git clone https://github.com/salesforce/WikiSQL.git
```

After cloning, keep the `data.tar.bz2` file from WikiSQL and make sure your local path is available to the notebook.

## 2) Prerequisites

Install dependencies:

```bash
pip install torch transformers sentencepiece
```

## 3) Configure paths in notebook

Open `NL2SQL_preprocessing.ipynb` and update the path variables in the setup cell:

- `PROJECT_ROOT`
- `WIKISQL_DIR` (folder where WikiSQL repo/data lives)
- `WIKISQL_ARCHIVE` (should point to `data.tar.bz2`)
- `WIKISQL_DATA_DIR` (extracted JSONL folder, usually `data/`)
- `PROCESSED_DIR` (where `.pt` outputs will be saved)

## 4) Run preprocessing

## 5) Output files
- `data/processed/wikisql_t5_train.pt`
- `data/processed/wikisql_t5_val.pt`

These files are used in the downstream training/fine-tuning step.
