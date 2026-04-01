from __future__ import annotations

import json
import re
import sys as _sys
from html import unescape
from pathlib import Path as _Path
from typing import Iterable

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter

from config.config import CHUNK_OVERLAP, CHUNK_SIZE


RAW_PATH = Path("data/raw/train.jsonl")
FALLBACK_RAW_PATH = Path("data/raw/pubmed_qa_train.jsonl")
OUT_PATH = Path("data/processed/pubmed_qa_train.jsonl")

HTML_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_html(text: str) -> str:
    text = unescape(text or "")
    text = HTML_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_source_path() -> Path:
    if RAW_PATH.exists():
        return RAW_PATH
    if FALLBACK_RAW_PATH.exists():
        return FALLBACK_RAW_PATH
    raise FileNotFoundError("No PubMed QA raw training file found under data/raw/.")


def _extract_abstract(record: dict) -> str:
    contexts = record.get("context") or record.get("context_docs") or []

    if isinstance(contexts, dict):
        nested_contexts = contexts.get("contexts") or []
        if isinstance(nested_contexts, list):
            return clean_html(" ".join(str(item) for item in nested_contexts))
        return clean_html(str(nested_contexts))

    if isinstance(contexts, list):
        return clean_html(" ".join(str(item) for item in contexts))

    return clean_html(str(contexts))


def _build_base_record(record: dict) -> dict | None:
    abstract = _extract_abstract(record)
    if len(abstract.split()) < 50:
        return None

    question = clean_html(str(record.get("question", "")))
    long_answer = clean_html(str(record.get("long_answer") or record.get("ground_truth") or ""))
    final_decision = clean_html(str(record.get("final_decision", "")))
    pubmed_id = clean_html(str(record.get("pubid") or record.get("pubmed_id") or ""))
    return {
        "question": question,
        "abstract": abstract,
        "long_answer": long_answer,
        "final_decision": final_decision,
        "pubmed_id": pubmed_id,
    }


def preprocess_records(records: Iterable[dict]) -> list[dict]:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    processed: list[dict] = []

    for record in records:
        base = _build_base_record(record)
        if base is None:
            continue

        chunks = splitter.split_text(base["abstract"])
        if not chunks:
            continue

        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            processed.append(
                {
                    **base,
                    "chunk_id": f"{base['pubmed_id'] or 'pubmed'}_{chunk_idx:03d}",
                    "text": chunk_text,
                    "chunk_index": chunk_idx,
                }
            )

    return processed


def main() -> None:
    source_path = _resolve_source_path()
    records = _load_jsonl(source_path)
    processed = preprocess_records(records)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for row in processed:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Preprocessed {len(processed)} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()
