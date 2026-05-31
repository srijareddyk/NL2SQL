# NL2SQL

Natural Language to SQL query generation using **T5-Base** fine-tuned with **LoRA** on the **WikiSQL** dataset.


## Project structure

```
NL2SQL/
├── NL2SQL_preprocessing.ipynb   # WikiSQL → tokenized .pt files
├── training.ipynb                 # LoRA fine-tuning with Hugging Face Trainer
├── project-proposal.md            # Full project vision (RAG, GUI, etc.)
└── data/
    └── processed/
        ├── wikisql_t5_train.pt    # 56,355 training examples
        └── wikisql_t5_val.pt      # 8,421 validation examples
```

## Dataset

- **Dataset:** [WikiSQL](https://github.com/salesforce/WikiSQL) (Salesforce)
- **Splits loaded locally:** train (56,355), validation (8,421), test (15,878)
- WikiSQL stores SQL as structured annotations; the preprocessing notebook converts these to executable SQL strings before tokenization.

### Download WikiSQL

```bash
git clone https://github.com/salesforce/WikiSQL.git
```

Place the repo at `WikiSQL/` under the project root (or update paths in the notebook). The archive `WikiSQL/data.tar.bz2` must be extracted so JSONL files are available under `WikiSQL/data/`.

## Prerequisites

```bash
pip install torch transformers sentencepiece peft
```

For GPU training, install a CUDA-enabled PyTorch build matching your environment.

## Workflow

### 1. Preprocess data

Open `NL2SQL_preprocessing.ipynb` and run all cells. The notebook:

1. Loads WikiSQL JSONL splits and table metadata
2. Converts WikiSQL SQL structures to SQL strings (`SELECT … WHERE …`)
3. Builds T5 inputs: `generate sql: {question} [SEP] table: {col:type | …}`
4. Tokenizes inputs and targets with `t5-base` (max source 256, max target 128)
5. Saves PyTorch tensors to `data/processed/`

**Outputs:**

- `data/processed/wikisql_t5_train.pt`
- `data/processed/wikisql_t5_val.pt`

Each file contains `input_ids`, `attention_mask`, and `labels` (padding tokens masked with `-100`).

### 2. Train the model

Open `training.ipynb` and run all cells. It loads the processed `.pt` files and fine-tunes **T5-Base** with **LoRA** via Hugging Face `Seq2SeqTrainer`.

**Default training config:**

| Setting | Value |
|---------|-------|
| Base model | `t5-base` |
| LoRA rank (`r`) | 32 |
| LoRA alpha | 64 |
| Target modules | `q`, `k`, `v`, `o`, `wi`, `wo` |
| Learning rate | 3e-4 |
| Batch size (train) | 32 |
| Epochs | 10 |
| Warmup steps | 500 |
| Eval metric | Exact-match on decoded SQL |

Checkpoints and the best model (by validation exact-match) are saved under `checkpoints/run2/`

**Current validation result:** ~39.7% exact-match on the WikiSQL validation split after 10 epochs.

### Input format

Each training example pairs a natural-language question with table schema context:

```
generate sql: Tell me what the notes are for South Australia [SEP] table: State/territory:text | Text/background colour:text | ...
```

**Target:**

```sql
SELECT Notes WHERE Current slogan = 'SOUTH AUSTRALIA'
```

### Next steps include:

- **RAG:** Retrieve schema context and similar examples from user-uploaded CSV/SQLite files at inference time
- **Chatbot GUI:** Upload databases, ask questions, view SQL and results, multi-turn history, and error explanations
