"""Build the WikiSQL FAISS retrieval index for RAG inference."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from rag import DEFAULT_INDEX_DIR, WikiSQLRetriever

WIKISQL_DATA_DIR = PROJECT_ROOT / "WikiSQL" / "data"
AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "!="]


def wikisql_to_sql(sql_struct: dict, headers: list[str]) -> str:
    sel = sql_struct.get("sel", 0)
    agg = sql_struct.get("agg", 0)
    conds = sql_struct.get("conds", [])
    sel_col = headers[sel] if sel < len(headers) else f"col_{sel}"
    agg_str = AGG_OPS[agg] if agg < len(AGG_OPS) else ""
    select_clause = f"SELECT {agg_str}({sel_col})" if agg_str else f"SELECT {sel_col}"
    where_parts = []
    for cond in conds:
        col_idx, op_idx, value = cond[0], cond[1], cond[2]
        col_name = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
        op = COND_OPS[op_idx] if op_idx < len(COND_OPS) else "="
        val_str = f"'{value}'" if isinstance(value, str) else str(value)
        where_parts.append(f"{col_name} {op} {val_str}")
    return (select_clause + " WHERE " + " AND ".join(where_parts)) if where_parts else select_clause


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_train_examples() -> list[dict]:
    table_files = sorted(WIKISQL_DATA_DIR.glob("*.tables.jsonl"))
    train_file = WIKISQL_DATA_DIR / "train.jsonl"
    if not table_files or not train_file.exists():
        raise FileNotFoundError(f"WikiSQL data not found under {WIKISQL_DATA_DIR}")

    table_map = {}
    for table_file in table_files:
        for table in _read_jsonl(table_file):
            table_map[table["id"]] = table

    examples = []
    for row in _read_jsonl(train_file):
        table_id = row["table_id"]
        if table_id not in table_map:
            continue
        sql_struct = row.get("sql", {})
        if isinstance(sql_struct, str):
            sql_struct = json.loads(sql_struct)
        headers = table_map[table_id]["header"]
        examples.append({
            "question": row["question"],
            "sql": wikisql_to_sql(sql_struct, headers),
            "table_id": table_id,
        })
    return examples


def main() -> None:
    examples = load_train_examples()
    print(f"Loaded {len(examples)} train examples.")
    retriever = WikiSQLRetriever.from_examples(examples)
    retriever.save(DEFAULT_INDEX_DIR)
    print(f"Saved retrieval index to {DEFAULT_INDEX_DIR}")


if __name__ == "__main__":
    main()
