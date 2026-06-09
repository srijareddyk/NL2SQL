from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "wikisql_retrieval"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class RetrievedExample:
    question: str
    sql: str
    table_id: str
    score: float


def _get_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def format_schema_block(columns: list[dict]) -> str:
    parts = []
    for col in columns:
        samples = ", ".join(col.get("samples", [])[:3])
        sample_str = f" e.g. {samples}" if samples else ""
        parts.append(f"{col['name']}:{col['type']}{sample_str}")
    return " | ".join(parts)


def format_schema_from_wikisql_table(table: dict) -> str:
    columns = [
        {"name": header, "type": col_type, "samples": []}
        for header, col_type in zip(table["header"], table["types"])
    ]
    return format_schema_block(columns)


def format_table_header(columns: list[dict] | None = None, table: dict | None = None) -> str:
    if columns is not None:
        return " | ".join(f"{c['name']}:{c['type']}" for c in columns)
    if table is not None:
        return " | ".join(f"{h}:{t}" for h, t in zip(table["header"], table["types"]))
    raise ValueError("Provide columns or a WikiSQL table dict.")


def format_examples_block(examples: list[RetrievedExample]) -> str:
    if not examples:
        return ""
    chunks = [f"Q: {ex.question} SQL: {ex.sql}" for ex in examples]
    return " ;; ".join(chunks)


def build_rag_prompt(
    question: str,
    *,
    columns: list[dict] | None = None,
    table: dict | None = None,
    examples: list[RetrievedExample] | None = None,
) -> str:
    """Build a RAG-augmented prompt matching the training format."""
    header = format_table_header(columns=columns, table=table)
    if columns is not None:
        schema_block = format_schema_block(columns)
    elif table is not None:
        schema_block = format_schema_from_wikisql_table(table)
    else:
        raise ValueError("Provide columns or a WikiSQL table dict.")

    parts = [
        f"generate sql: {question}",
        f"table: {header}",
        f"[SCHEMA] {schema_block}",
    ]
    example_text = format_examples_block(examples or [])
    if example_text:
        parts.append(f"[EXAMPLES] {example_text}")
    return " [SEP] ".join(parts)


def build_plain_prompt(question: str, *, columns: list[dict] | None = None, table: dict | None = None) -> str:
    header = format_table_header(columns=columns, table=table)
    return f"generate sql: {question} [SEP] table: {header}"


class WikiSQLRetriever:
    def __init__(
        self,
        index: faiss.Index,
        questions: list[str],
        sqls: list[str],
        table_ids: list[str],
        encoder=None,
    ):
        self.index = index
        self.questions = questions
        self.sqls = sqls
        self.table_ids = table_ids
        self.encoder = encoder or _get_encoder()

    @property
    def size(self) -> int:
        return len(self.questions)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def retrieve(
        self,
        question: str,
        k: int = DEFAULT_TOP_K,
        *,
        prefer_table_id: str | None = None,
        exclude_questions: set[str] | None = None,
        search_pool: int = 24,
    ) -> list[RetrievedExample]:
        if self.size == 0:
            return []

        query = self._encode([question])
        pool = min(max(k * 4, search_pool), self.size)
        scores, indices = self.index.search(query, pool)

        exclude = {q.strip().lower() for q in (exclude_questions or set())}
        candidates: list[RetrievedExample] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            q = self.questions[idx]
            if q.strip().lower() in exclude:
                continue
            candidates.append(
                RetrievedExample(
                    question=q,
                    sql=self.sqls[idx],
                    table_id=self.table_ids[idx],
                    score=float(score),
                )
            )

        if prefer_table_id:
            same_table = [c for c in candidates if c.table_id == prefer_table_id]
            other_table = [c for c in candidates if c.table_id != prefer_table_id]
            candidates = same_table + other_table

        return candidates[:k]

    @classmethod
    def from_examples(cls, examples: list[dict], encoder=None) -> WikiSQLRetriever:
        encoder = encoder or _get_encoder()
        questions = [ex["question"] for ex in examples]
        sqls = [ex["sql"] for ex in examples]
        table_ids = [ex["table_id"] for ex in examples]

        vectors = encoder.encode(questions, convert_to_numpy=True, show_progress_bar=True)
        vectors = _normalize(np.asarray(vectors, dtype=np.float32))

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        return cls(index, questions, sqls, table_ids, encoder=encoder)

    def save(self, index_dir: Path | str) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        metadata = {
            "embedding_model": EMBEDDING_MODEL,
            "questions": self.questions,
            "sqls": self.sqls,
            "table_ids": self.table_ids,
        }
        with (index_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, index_dir: Path | str, encoder=None) -> WikiSQLRetriever:
        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / "index.faiss"))
        with (index_dir / "metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        return cls(
            index=index,
            questions=metadata["questions"],
            sqls=metadata["sqls"],
            table_ids=metadata["table_ids"],
            encoder=encoder,
        )
