from __future__ import annotations

import sys
import types
import unittest


def _install_fake_build_indices_deps() -> None:
    if "faiss" not in sys.modules:
        fake_faiss = types.ModuleType("faiss")

        class IndexFlatL2:
            def __init__(self, dim: int) -> None:
                self.dim = dim

        fake_faiss.IndexFlatL2 = IndexFlatL2
        sys.modules["faiss"] = fake_faiss

    if "torch" not in sys.modules:
        fake_torch = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        fake_torch.cuda = _Cuda()
        sys.modules["torch"] = fake_torch

    if "llama_index" in sys.modules:
        return

    llama_index = types.ModuleType("llama_index")
    core = types.ModuleType("llama_index.core")
    schema = types.ModuleType("llama_index.core.schema")
    embeddings = types.ModuleType("llama_index.embeddings")
    embeddings_hf = types.ModuleType("llama_index.embeddings.huggingface")
    retrievers = types.ModuleType("llama_index.retrievers")
    retrievers_bm25 = types.ModuleType("llama_index.retrievers.bm25")
    vector_stores = types.ModuleType("llama_index.vector_stores")
    vector_stores_faiss = types.ModuleType("llama_index.vector_stores.faiss")

    class Settings:
        embed_model = None

    class StorageContext:
        @staticmethod
        def from_defaults(*args, **kwargs):
            return object()

    class VectorStoreIndex:
        @staticmethod
        def from_documents(*args, **kwargs):
            return object()

    def load_index_from_storage(*args, **kwargs):
        return object()

    class BaseNode:
        pass

    class Document:
        def __init__(
            self,
            text: str,
            doc_id: str,
            metadata: dict | None = None,
            excluded_embed_metadata_keys: list[str] | None = None,
            excluded_llm_metadata_keys: list[str] | None = None,
        ) -> None:
            self.text = text
            self.doc_id = doc_id
            self.metadata = metadata or {}
            self.excluded_embed_metadata_keys = excluded_embed_metadata_keys or []
            self.excluded_llm_metadata_keys = excluded_llm_metadata_keys or []

    class TextNode:
        def __init__(
            self,
            text: str,
            id_: str,
            metadata: dict | None = None,
            excluded_embed_metadata_keys: list[str] | None = None,
            excluded_llm_metadata_keys: list[str] | None = None,
        ) -> None:
            self.text = text
            self.id_ = id_
            self.metadata = metadata or {}
            self.excluded_embed_metadata_keys = excluded_embed_metadata_keys or []
            self.excluded_llm_metadata_keys = excluded_llm_metadata_keys or []

    class HuggingFaceEmbedding:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_text_embedding(self, text: str) -> list[float]:
            return [0.0]

    class BM25Retriever:
        @staticmethod
        def from_defaults(*args, **kwargs):
            return object()

        @staticmethod
        def from_persist_dir(*args, **kwargs):
            return object()

    class FaissVectorStore:
        def __init__(self, *args, **kwargs) -> None:
            pass

        @staticmethod
        def from_persist_dir(*args, **kwargs):
            return object()

    core.Settings = Settings
    core.StorageContext = StorageContext
    core.VectorStoreIndex = VectorStoreIndex
    core.load_index_from_storage = load_index_from_storage
    schema.BaseNode = BaseNode
    schema.Document = Document
    schema.TextNode = TextNode
    embeddings_hf.HuggingFaceEmbedding = HuggingFaceEmbedding
    retrievers_bm25.BM25Retriever = BM25Retriever
    vector_stores_faiss.FaissVectorStore = FaissVectorStore

    core.schema = schema
    llama_index.core = core
    embeddings.huggingface = embeddings_hf
    retrievers.bm25 = retrievers_bm25
    vector_stores.faiss = vector_stores_faiss

    sys.modules["llama_index"] = llama_index
    sys.modules["llama_index.core"] = core
    sys.modules["llama_index.core.schema"] = schema
    sys.modules["llama_index.embeddings"] = embeddings
    sys.modules["llama_index.embeddings.huggingface"] = embeddings_hf
    sys.modules["llama_index.retrievers"] = retrievers
    sys.modules["llama_index.retrievers.bm25"] = retrievers_bm25
    sys.modules["llama_index.vector_stores"] = vector_stores
    sys.modules["llama_index.vector_stores.faiss"] = vector_stores_faiss


_install_fake_build_indices_deps()

from src.data import build_indices as build_indices_mod


class TestBuildIndicesNodes(unittest.TestCase):
    def test_documents_and_nodes_exclude_all_metadata_from_content(self) -> None:
        row = {
            "chunk_id": "123_001",
            "text": "clean text only",
            "pubmed_id": "123",
            "question": "question text",
            "final_decision": "yes",
            "long_answer": "long answer",
            "chunk_index": 1,
        }

        doc = build_indices_mod._rows_to_documents([row])[0]
        node = build_indices_mod._rows_to_nodes([row])[0]
        expected_keys = [
            "chunk_id",
            "pubmed_id",
            "question",
            "final_decision",
            "long_answer",
            "chunk_index",
        ]

        self.assertEqual(doc.metadata, node.metadata)
        self.assertEqual(doc.excluded_embed_metadata_keys, expected_keys)
        self.assertEqual(doc.excluded_llm_metadata_keys, expected_keys)
        self.assertEqual(node.excluded_embed_metadata_keys, expected_keys)
        self.assertEqual(node.excluded_llm_metadata_keys, expected_keys)


if __name__ == "__main__":
    unittest.main()
