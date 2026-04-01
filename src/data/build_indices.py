from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys as _sys
from pathlib import Path as _Path
from typing import cast
from typing import Iterable

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path

import faiss
import torch
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode, Document, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore

from config import config as app_config


PROCESSED_PATH = _ROOT / "data/processed/pubmed_qa_train.jsonl"
INDICES_DIR = _ROOT / "data/indices"
BM25_DIR = INDICES_DIR / "bm25"
MD5_PATH = INDICES_DIR / "corpus.md5"
BM25_CACHE_PATH = BM25_DIR / "retriever.pkl"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _load_processed_rows() -> list[dict]:
    with PROCESSED_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row_metadata(row: dict) -> dict:
    return {
        "chunk_id": row["chunk_id"],
        "pubmed_id": row.get("pubmed_id"),
        "question": row.get("question"),
        "final_decision": row.get("final_decision"),
        "long_answer": row.get("long_answer"),
        "chunk_index": row.get("chunk_index"),
    }


def _rows_to_documents(rows: Iterable[dict]) -> list[Document]:
    docs: list[Document] = []
    for row in rows:
        metadata = _row_metadata(row)
        metadata_keys = list(metadata.keys())
        docs.append(
            Document(
                text=row["text"],
                doc_id=row["chunk_id"],
                metadata=metadata,
                excluded_embed_metadata_keys=metadata_keys,
                excluded_llm_metadata_keys=metadata_keys,
            )
        )
    return docs


def _rows_to_nodes(rows: Iterable[dict]) -> list[TextNode]:
    nodes: list[TextNode] = []
    for row in rows:
        metadata = _row_metadata(row)
        metadata_keys = list(metadata.keys())
        nodes.append(
            TextNode(
                text=row["text"],
                id_=row["chunk_id"],
                metadata=metadata,
                excluded_embed_metadata_keys=metadata_keys,
                excluded_llm_metadata_keys=metadata_keys,
            )
        )
    return nodes


def compute_corpus_md5() -> str:
    digest = hashlib.md5()
    with PROCESSED_PATH.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_is_current() -> bool:
    return MD5_PATH.exists() and MD5_PATH.read_text(encoding="utf-8").strip() == compute_corpus_md5()


def _build_embed_model() -> HuggingFaceEmbedding:
    configured_device = getattr(app_config, "EMBEDDING_DEVICE", "cpu")
    if configured_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = configured_device
    return HuggingFaceEmbedding(model_name=EMBED_MODEL, device=device)


def _embedding_dimension(embed_model: HuggingFaceEmbedding) -> int:
    return len(embed_model.get_text_embedding("dimension probe"))


def _clear_persisted_indices() -> None:
    if INDICES_DIR.exists():
        shutil.rmtree(INDICES_DIR)


def build_indices(force: bool = False) -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Missing processed corpus: {PROCESSED_PATH}")

    vector_ready = (INDICES_DIR / "index_store.json").exists()
    bm25_ready = (BM25_DIR / "retriever.pkl").exists()

    if not force and corpus_is_current() and vector_ready and bm25_ready:
        print("Corpus hash unchanged; skipping rebuild.")
        return

    if not force and corpus_is_current() and vector_ready and not bm25_ready:
        rows = _load_processed_rows()
        BM25_DIR.mkdir(parents=True, exist_ok=True)
        with BM25_CACHE_PATH.open("wb") as handle:
            pickle.dump(rows, handle)
        MD5_PATH.write_text(compute_corpus_md5(), encoding="utf-8")
        print(f"Built BM25 cache in {BM25_DIR}")
        return

    embed_model = _build_embed_model()
    Settings.embed_model = embed_model

    if not force and corpus_is_current() and vector_ready:
        index = load_index()
    else:
        # Persisting into an existing LlamaIndex directory can leave stale
        # nodes/docstore entries behind, so a real rebuild starts from a clean
        # directory whenever the corpus changed or the caller forces a rebuild.
        _clear_persisted_indices()
        rows = _load_processed_rows()
        documents = _rows_to_documents(rows)
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(_embedding_dimension(embed_model)))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
        INDICES_DIR.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(INDICES_DIR))

    bm25 = BM25Retriever.from_defaults(nodes=list(index.docstore.docs.values()), similarity_top_k=5)
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    with BM25_CACHE_PATH.open("wb") as handle:
        pickle.dump(_load_processed_rows(), handle)

    MD5_PATH.write_text(compute_corpus_md5(), encoding="utf-8")
    print(f"Built vector + BM25 indices in {INDICES_DIR}")


def load_index() -> VectorStoreIndex:
    embed_model = _build_embed_model()
    Settings.embed_model = embed_model
    vector_store = FaissVectorStore.from_persist_dir(str(INDICES_DIR))
    storage_context = StorageContext.from_defaults(
        persist_dir=str(INDICES_DIR),
        vector_store=vector_store,
    )
    return cast(VectorStoreIndex, load_index_from_storage(storage_context, embed_model=embed_model))


def load_bm25_retriever() -> BM25Retriever:
    if BM25_CACHE_PATH.exists() and BM25_CACHE_PATH.stat().st_size > 0:
        try:
            with BM25_CACHE_PATH.open("rb") as handle:
                rows = pickle.load(handle)
            nodes = cast(list[BaseNode], _rows_to_nodes(rows))
            return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
        except Exception:
            # Fallback to the persisted BM25 artifact format produced by
            # BM25Retriever.persist()/from_persist_dir.
            pass
    return BM25Retriever.from_persist_dir(str(BM25_DIR), similarity_top_k=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete persisted index artifacts and rebuild from the processed corpus.",
    )
    args = parser.parse_args()
    build_indices(force=args.force)


if __name__ == "__main__":
    main()
