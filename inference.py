from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from rag import (
    DEFAULT_INDEX_DIR,
    DEFAULT_TOP_K,
    RetrievedExample,
    WikiSQLRetriever,
    build_plain_prompt,
    build_rag_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "best_lora"
DEMO_CSV = PROJECT_ROOT / "players.csv"
BASE_MODEL = "t5-base"

MAX_SOURCE_LEN = 384
MAX_SOURCE_LEN_RAG = 512
MAX_TARGET_LEN = 128
NUM_BEAMS = 4
RAG_TOP_K = DEFAULT_TOP_K

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


def _parse_wikisql_struct(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.upper().startswith("SELECT"):
        return None
    if not raw.startswith("{"):
        raw = "{" + raw
    if not raw.endswith("}"):
        raw = raw + "}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def model_output_to_sql(raw: str, headers: list[str]) -> str:
    raw = raw.strip()
    if raw.upper().startswith("SELECT"):
        return raw
    struct = _parse_wikisql_struct(raw)
    if struct is not None:
        return wikisql_to_sql(struct, headers)
    return raw


def _infer_col_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "real"
    return "text"


def _sample_values(series: pd.Series, n: int = 3) -> list[str]:
    return [str(v) for v in series.dropna().unique()[:n]]


def extract_schema_from_csv(csv_path) -> dict:
    path = Path(csv_path)
    df = pd.read_csv(path)
    return {
        "source": str(path),
        "tables": [{
            "table_name": path.stem,
            "columns": [
                {"name": col, "type": _infer_col_type(df[col]), "samples": _sample_values(df[col])}
                for col in df.columns
            ],
            "row_count": len(df),
            "_df": df,
        }],
    }


def extract_schema_from_sqlite(db_path) -> dict:
    path = Path(db_path)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = []
    for (tname,) in cursor.fetchall():
        cursor.execute(f'PRAGMA table_info("{tname}")')
        pragma = cursor.fetchall()
        df = pd.read_sql_query(f'SELECT * FROM "{tname}" LIMIT 1000', conn)
        tables.append({
            "table_name": tname,
            "columns": [
                {
                    "name": r[1],
                    "type": (r[2] or "text").lower(),
                    "samples": _sample_values(df[r[1]]) if r[1] in df.columns else [],
                }
                for r in pragma
            ],
            "row_count": len(df),
        })
    conn.close()
    return {"source": str(path), "tables": tables, "_db_path": str(path)}


def load_user_data(file_path) -> dict:
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        return extract_schema_from_csv(path)
    if path.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        return extract_schema_from_sqlite(path)
    raise ValueError(f"Unsupported: {path.suffix} — use .csv or .db/.sqlite")


def build_prompt(
    question: str,
    schema: dict,
    *,
    use_rag: bool = False,
    examples: list[RetrievedExample] | None = None,
) -> str:
    primary_table = schema["tables"][0]
    if use_rag:
        return build_rag_prompt(
            question,
            columns=primary_table["columns"],
            examples=examples,
        )
    return build_plain_prompt(question, columns=primary_table["columns"])


def _execute_on_sqlite(db_path: str, sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def _execute_on_csv(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    agg_m = re.search(r"SELECT\s+(MAX|MIN|COUNT|SUM|AVG)\s*\((.+?)\)", sql, re.I)
    plain_m = re.search(r"SELECT\s+(.+?)\s*(?:WHERE|$)", sql, re.I)

    agg_func = agg_m.group(1).upper() if agg_m else None
    sel_col = agg_m.group(2) if agg_m else plain_m.group(1) if plain_m else None
    if sel_col:
        sel_col = sel_col.strip()

    filtered = df.copy()
    where_m = re.search(r"WHERE\s+(.+)$", sql, re.I)
    if where_m:
        for cond in re.split(r"\s+AND\s+", where_m.group(1), flags=re.I):
            m = re.match(r"(.+?)\s*(=|!=|>=|<=|>|<)\s*'?([^']+)'?", cond.strip())
            if not m:
                continue
            col, op, val = m.group(1).strip(), m.group(2), m.group(3).strip()
            if col not in filtered.columns:
                continue
            try:
                n = float(val)
                masks = {
                    "=": filtered[col] == n,
                    "!=": filtered[col] != n,
                    ">": filtered[col] > n,
                    "<": filtered[col] < n,
                    ">=": filtered[col] >= n,
                    "<=": filtered[col] <= n,
                }
                filtered = filtered[masks.get(op, filtered[col] == n)]
            except ValueError:
                sc = filtered[col].astype(str).str.upper()
                filtered = filtered[sc == val.upper() if op == "=" else sc != val.upper()]

    if sel_col and sel_col in filtered.columns:
        s = filtered[sel_col]
        if agg_func == "COUNT":
            return pd.DataFrame({agg_func: [len(s)]})
        if agg_func == "SUM":
            return pd.DataFrame({agg_func: [pd.to_numeric(s, errors="coerce").sum()]})
        if agg_func == "AVG":
            return pd.DataFrame({agg_func: [pd.to_numeric(s, errors="coerce").mean()]})
        if agg_func == "MAX":
            return pd.DataFrame({agg_func: [s.max()]})
        if agg_func == "MIN":
            return pd.DataFrame({agg_func: [s.min()]})
        return filtered[[sel_col]]
    return filtered


def execute_sql(sql: str, schema: dict):
    if not sql.strip().upper().startswith("SELECT"):
        return f"Model did not generate valid SQL: {sql}"
    try:
        if "_db_path" in schema:
            return _execute_on_sqlite(schema["_db_path"], sql)
        t = schema["tables"][0]
        if "_df" in t:
            return _execute_on_csv(t["_df"], sql)
        return "No executable data source in schema."
    except Exception as e:
        return f"Execution error: {e}"


class NL2SQLPipeline:
    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        device: str | None = None,
        use_rag: bool = False,
        retrieval_index_dir: Path | None = None,
        rag_top_k: int = RAG_TOP_K,
    ):
        self.checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k
        self.max_source_len = MAX_SOURCE_LEN_RAG if use_rag else MAX_SOURCE_LEN
        self.retriever: WikiSQLRetriever | None = None

        if self.use_rag:
            index_dir = Path(retrieval_index_dir or DEFAULT_INDEX_DIR)
            if not index_dir.exists():
                raise FileNotFoundError(
                    f"RAG index not found at {index_dir}. "
                    "Run: python build_retrieval_index.py"
                )
            self.retriever = WikiSQLRetriever.load(index_dir)

        self.tokenizer = T5TokenizerFast.from_pretrained(BASE_MODEL)
        base_model = T5ForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, self.checkpoint_dir)
        self.model = peft_model.merge_and_unload()
        self.model.eval()
        self.model.to(self.device)

    def retrieve_examples(self, question: str) -> list[RetrievedExample]:
        if not self.use_rag or self.retriever is None:
            return []
        return self.retriever.retrieve(
            question,
            k=self.rag_top_k,
            exclude_questions={question},
        )

    @torch.inference_mode()
    def generate_sql(self, question: str, schema: dict) -> tuple[str, str, list[RetrievedExample]]:
        """Return (raw_model_output, executable_sql, retrieved_examples)."""
        headers = [c["name"] for c in schema["tables"][0]["columns"]]
        examples = self.retrieve_examples(question)
        prompt = build_prompt(
            question,
            schema,
            use_rag=self.use_rag,
            examples=examples,
        )
        enc = self.tokenizer(
            prompt,
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=MAX_TARGET_LEN,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        raw = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return raw, model_output_to_sql(raw, headers), examples

    def query(self, question: str, schema: dict, execute: bool = True) -> dict:
        raw, sql, examples = self.generate_sql(question, schema)
        result = execute_sql(sql, schema) if execute else None
        return {
            "question": question,
            "raw": raw,
            "sql": sql,
            "result": result,
            "retrieved_examples": examples,
        }


def load_schema_from_upload(uploaded_name: str, uploaded_bytes: bytes) -> dict:
    """Save an uploaded file to a temp path and load schema."""
    suffix = Path(uploaded_name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_bytes)
        tmp_path = tmp.name
    return load_user_data(tmp_path)
