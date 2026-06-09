# NL2SQL

Natural Language to SQL query generation using **T5-Base** fine-tuned with **LoRA** on the **WikiSQL** dataset.

## Architecture

Three-stage pipeline: preprocess WikiSQL into tokenized tensors, fine-tune a LoRA adapter with Optuna/Hyperband HP search, then run inference on user-uploaded CSV/SQLite files via `inference.py` or the Streamlit chatbot.



### Model

| Component | Detail |
|-----------|--------|
| Base model | `t5-base` (235M parameters, encoder–decoder) |
| Adaptation | LoRA via PEFT on attention and FFN projections (`q`, `k`, `v`, `o`, `wi`, `wo`) |
| Trainable params | ~ 13.0M (~ 5.5% of total) |
| Task | Seq2seq: natural-language question + table schema → SQL |
| RAG (optional) | FAISS + `all-MiniLM-L6-v2` retrieves similar WikiSQL Q/SQL pairs at inference |

The model was trained on plain prompts and emits **executable SQL strings** directly. RAG can be toggled on.

### Input / output format

**Training / default inference prompt:**

```
generate sql: What are the points for Alice? [SEP] table: Player:text | Position:text | Team:text | Points:integer | Assists:integer
```

**RAG-augmented prompt (inference only, toggle in app):**

```
generate sql: ... [SEP] table: ... [SEP] [SCHEMA] Player:text e.g. Alice,Bob | ... [SEP] [EXAMPLES] Q: ... SQL: SELECT ... ;;
```

**Model output:**

```sql
SELECT Points WHERE Player = 'Alice'
```

## RAG (optional at inference)

RAG is implemented in `rag.py` and enabled via a toggle in the Streamlit app (default: on).

1. **Index** — WikiSQL train questions embedded with `sentence-transformers/all-MiniLM-L6-v2`, stored in `data/processed/wikisql_retrieval/`.
2. **Inference** — For each user question, retrieve similar WikiSQL Q/SQL pairs and append under `[EXAMPLES]`.
3. **Build / rebuild index:**

```bash
python build_retrieval_index.py
```

## Extra credit 

| Criterion | Status |
|-----------|--------|
| **ML operations** | **Optuna + Hyperband** HP search, automated best-checkpoint selection (`SaveBestLoraCallback`), deployable `inference.py` pipeline |
| **RAG** | FAISS retrieval of similar WikiSQL examples, optional inference-time augmentation in `inference.py` / `app.py` |
| **Chatbot GUI** | Streamlit app (`app.py`): upload CSV/SQLite, RAG toggle, view SQL + results |

## Dataset

- **Source:** [WikiSQL](https://github.com/salesforce/WikiSQL) (Salesforce)
- **Splits:** train (56,355), validation (8,421), test (15,878)
- Preprocessing converts WikiSQL struct annotations to executable SQL strings before tokenization.

### Download WikiSQL

```bash
git clone https://github.com/salesforce/WikiSQL.git
```

Place the repo at `WikiSQL/` under the project root. Extract `WikiSQL/data.tar.bz2` so JSONL files are available under `WikiSQL/data/`.

## Prerequisites

```bash
pip install torch transformers sentencepiece peft optuna numpy pandas matplotlib
```

For RAG:

```bash
pip install faiss-cpu sentence-transformers
```

For the chatbot GUI:

```bash
pip install streamlit
```

For GPU training, install a CUDA-enabled PyTorch build. Training uses **bf16** mixed precision when CUDA is available.

## Workflow

### 1. Preprocess data

Open `NL2SQL_preprocessing.ipynb` and run all cells. Outputs:

- `data/processed/wikisql_t5_train.pt`
- `data/processed/wikisql_t5_val.pt`

### 2. Hyperparameter search (optional)

`training.ipynb` includes **Optuna + Hyperband** over learning rate, batch size, LoRA rank, weight decay, and warmup steps. Set `SKIP_HP_SEARCH = True` to skip and use best params from my training:

| Hyperparameter | Range |
|----------------|-------|
| Learning rate | 1e-4 – 5e-4 (log scale) |
| Batch size | 16 or 32 |
| LoRA rank (`r`) | 16, 32, or 64 |
| Weight decay | 0.0 – 0.1 |
| Warmup steps | 200 – 1000 (step 200) |

### 3. Train the model

Run `training.ipynb` on a GPU machine. Training artifacts go to `checkpoints/run4/`; the best adapter is saved to `checkpoints/best_lora/` via `SaveBestLoraCallback`.

**Validation results (run4):**

| Metric | Value |
|--------|-------|
| Best exact match | **74.2%** (epoch 15) |
| Final logged exact match | 73.8% (epoch 16) |

### 4. Build RAG index 

```bash
python build_retrieval_index.py
```

### 5. Run inference

**Python API:**

```python
from inference import NL2SQLPipeline, load_user_data, DEMO_CSV

pipeline = NL2SQLPipeline(use_rag=True) 
schema = load_user_data(DEMO_CSV)
out = pipeline.query("What are the points for Alice?", schema)
print(out["sql"], out["result"])
```

**Chatbot GUI:**

```bash
streamlit run app.py
```

Upload `players.csv` (or any CSV/SQLite file), toggle RAG in the sidebar, and ask questions in plain English.

## Planned extensions
- RAG-augmented training prompts (retrain with `[SCHEMA]` / `[EXAMPLES]` in preprocessing)
