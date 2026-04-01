from __future__ import annotations

import sys
import types
import unittest


def _install_fake_retriever_deps() -> None:
    llama_index = sys.modules.get("llama_index")
    core = sys.modules.get("llama_index.core")
    schema = sys.modules.get("llama_index.core.schema")
    embeddings = sys.modules.get("llama_index.embeddings")
    embeddings_hf = sys.modules.get("llama_index.embeddings.huggingface")
    retrievers = sys.modules.get("llama_index.retrievers")
    retrievers_bm25 = sys.modules.get("llama_index.retrievers.bm25")
    vector_stores = sys.modules.get("llama_index.vector_stores")
    vector_stores_faiss = sys.modules.get("llama_index.vector_stores.faiss")

    if llama_index is None:
        llama_index = types.ModuleType("llama_index")
        sys.modules["llama_index"] = llama_index
    if core is None:
        core = types.ModuleType("llama_index.core")
        sys.modules["llama_index.core"] = core
    if schema is None:
        schema = types.ModuleType("llama_index.core.schema")
        sys.modules["llama_index.core.schema"] = schema
    if embeddings is None:
        embeddings = types.ModuleType("llama_index.embeddings")
        sys.modules["llama_index.embeddings"] = embeddings
    if embeddings_hf is None:
        embeddings_hf = types.ModuleType("llama_index.embeddings.huggingface")
        sys.modules["llama_index.embeddings.huggingface"] = embeddings_hf
    if retrievers is None:
        retrievers = types.ModuleType("llama_index.retrievers")
        sys.modules["llama_index.retrievers"] = retrievers
    if retrievers_bm25 is None:
        retrievers_bm25 = types.ModuleType("llama_index.retrievers.bm25")
        sys.modules["llama_index.retrievers.bm25"] = retrievers_bm25
    if vector_stores is None:
        vector_stores = types.ModuleType("llama_index.vector_stores")
        sys.modules["llama_index.vector_stores"] = vector_stores
    if vector_stores_faiss is None:
        vector_stores_faiss = types.ModuleType("llama_index.vector_stores.faiss")
        sys.modules["llama_index.vector_stores.faiss"] = vector_stores_faiss

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
            text: str = "",
            doc_id: str = "",
            metadata: dict | None = None,
            excluded_embed_metadata_keys: list[str] | None = None,
            excluded_llm_metadata_keys: list[str] | None = None,
            **kwargs,
        ) -> None:
            self.text = text
            self.doc_id = doc_id
            self.metadata = metadata or {}
            self.excluded_embed_metadata_keys = excluded_embed_metadata_keys or []
            self.excluded_llm_metadata_keys = excluded_llm_metadata_keys or []

    class NodeWithScore:
        def __init__(self, text: str = "", metadata: dict | None = None, score: float = 0.0) -> None:
            self._text = text
            self.metadata = metadata or {}
            self.score = score
            self.node_id = self.metadata.get("chunk_id")

        def get_content(self) -> str:
            return self._text

    class TextNode(NodeWithScore):
        def __init__(
            self,
            text: str = "",
            id_: str = "",
            metadata: dict | None = None,
            excluded_embed_metadata_keys: list[str] | None = None,
            excluded_llm_metadata_keys: list[str] | None = None,
            **kwargs,
        ) -> None:
            super().__init__(text=text, metadata=metadata)
            self.id_ = id_
            self.excluded_embed_metadata_keys = excluded_embed_metadata_keys or []
            self.excluded_llm_metadata_keys = excluded_llm_metadata_keys or []

    class QueryBundle:
        def __init__(self, query_str: str) -> None:
            self.query_str = query_str

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
    schema.NodeWithScore = NodeWithScore
    schema.TextNode = TextNode
    schema.QueryBundle = QueryBundle
    embeddings_hf.HuggingFaceEmbedding = HuggingFaceEmbedding
    retrievers_bm25.BM25Retriever = BM25Retriever
    vector_stores_faiss.FaissVectorStore = FaissVectorStore
    core.schema = schema
    llama_index.core = core
    embeddings.huggingface = embeddings_hf
    retrievers.bm25 = retrievers_bm25
    vector_stores.faiss = vector_stores_faiss

    if "src.retrieval.hybrid" not in sys.modules:
        hybrid = types.ModuleType("src.retrieval.hybrid")
        hybrid.load_hybrid_retriever = lambda *args, **kwargs: None
        sys.modules["src.retrieval.hybrid"] = hybrid

    if "src.retrieval.reranker" not in sys.modules:
        reranker = types.ModuleType("src.retrieval.reranker")
        reranker.build_reranker = lambda *args, **kwargs: None
        sys.modules["src.retrieval.reranker"] = reranker


_install_fake_retriever_deps()

from llama_index.core.schema import NodeWithScore

from src.graph.nodes import retriever as retriever_mod


class TestRetrieverRelevance(unittest.TestCase):
    def test_domain_gate_rejects_single_generic_overlap(self) -> None:
        query_terms = retriever_mod._keywords(
            "Does partial expander deflation exacerbate the adverse effects of radiotherapy in two-stage breast reconstruction?"
        )
        node = NodeWithScore(
            text=(
                "Radiotherapy reduces local recurrence rates after rectal cancer treatment "
                "and affects pelvic exenteration outcomes."
            ),
            metadata={"question": "Does radiotherapy affect rectal cancer recurrence?"},
        )

        self.assertFalse(retriever_mod._is_domain_relevant(node, query_terms, min_overlap=0.08))

    def test_domain_gate_accepts_multiple_specific_overlaps(self) -> None:
        query_terms = retriever_mod._keywords(
            "Does partial expander deflation exacerbate the adverse effects of radiotherapy in two-stage breast reconstruction?"
        )
        node = NodeWithScore(
            text=(
                "Two-stage breast reconstruction with a tissue expander was evaluated "
                "during radiotherapy."
            ),
            metadata={"question": "Does radiotherapy affect breast reconstruction outcomes?"},
        )

        self.assertTrue(retriever_mod._is_domain_relevant(node, query_terms, min_overlap=0.08))


if __name__ == "__main__":
    unittest.main()
