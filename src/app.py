from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure absolute imports like "src.*" resolve even if Streamlit changes cwd.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import config as app_config
from src.graph.graph import compile_graph
from src.graph.nodes.retriever import clear_retriever_cache
from src.model.llm_wrapper import QuantizedHFLLM, register_llm
from src.model.loader import load_model_and_tokenizer
from src.model.prompts import classify_query_mode
from src.utils.answer_cleaning import is_abstention


def _build_and_register_llm(*, finetuned: bool = True) -> QuantizedHFLLM:
    loaded = load_model_and_tokenizer(finetuned=finetuned)
    llm = QuantizedHFLLM(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        max_new_tokens=getattr(app_config, "GENERATION_MAX_NEW_TOKENS", None),
        min_new_tokens=getattr(app_config, "GENERATION_MIN_NEW_TOKENS", 16),
        temperature=getattr(app_config, "GENERATION_TEMPERATURE", 0.0),
        context_window=getattr(app_config, "INFERENCE_CONTEXT_WINDOW", 2048),
        do_sample=False,
        repetition_penalty=1.05,
        top_p=1.0,
        top_k=1,
        num_beams=1,
    )
    register_llm(llm)
    return llm


def _index_fingerprint() -> str:
    md5_path = _ROOT / "data" / "indices" / "corpus.md5"
    if md5_path.exists():
        return md5_path.read_text(encoding="utf-8").strip()

    processed_path = _ROOT / "data" / "processed" / "pubmed_qa_train.jsonl"
    if processed_path.exists():
        stat = processed_path.stat()
        return f"{int(stat.st_mtime)}:{stat.st_size}"

    return "missing-index"


def _runtime_info() -> dict[str, str]:
    app_path = Path(__file__).resolve()
    index_fp = _index_fingerprint()
    return {
        "app_path": str(app_path),
        "workspace_root": str(_ROOT),
        "index_fingerprint": index_fp,
        "build": f"{app_path.stat().st_mtime_ns}:{index_fp[:8]}",
    }


@st.cache_resource
def _get_app(index_fingerprint: str, finetuned: bool):
    # Recreate retrieval resources whenever the persisted index fingerprint
    # changes so the app cannot keep serving stale in-memory chunks after a
    # manual rebuild on the same instance.
    clear_retriever_cache()
    # Build and register local LLM so LlamaIndex does not default to OpenAI.
    _build_and_register_llm(finetuned=finetuned)
    return compile_graph()


def main() -> None:
    st.set_page_config(page_title="Medical RAG Agent", layout="wide")
    st.title("Medical RAG Agent")
    runtime = _runtime_info()
    st.caption(f"Build: `{runtime['build']}`")

    query = st.text_input("Query")
    run = st.button("Run")

    if run and query:
        with st.spinner("Running retrieval and generation..."):
            result = _get_app(
                _index_fingerprint(),
                bool(getattr(app_config, "USE_FINETUNED", True)),
            ).invoke({"query": query, "retry_count": 0})
        retrieved_docs = result.get("retrieved_docs", [])
        citations = result.get("citations", [])
        final_answer = str(result.get("final_answer", "No answer generated.") or "")
        is_withheld = is_abstention(final_answer)

        if not retrieved_docs:
            abstention_reason = "No grounded documents were retrieved."
        elif is_withheld:
            abstention_reason = "The model returned an abstention despite having retrieved context."
        elif float(result.get("faithfulness_score", 0.0) or 0.0) < float(getattr(app_config, "FAITHFULNESS_THRESHOLD", 0.4)):
            abstention_reason = "The critic judged the answer weakly supported."
        else:
            abstention_reason = "The pipeline did not abstain."

        # Disclaimer is shown once here; synthesizer no longer prepends it to final_answer.
        if result.get("safety_filter_triggered"):
            st.warning(result.get("disclaimer", "Medical disclaimer: This output is informational only and must not be used as a substitute for licensed clinical judgment.").strip())

        st.markdown(final_answer)
        if citations and not is_withheld:
            rendered = " ".join(f"[{citation}]" for citation in citations)
            st.caption(f"Citations: {rendered}")
        else:
            st.caption("No grounded citations available for this answer.")

        with st.expander("Sources"):
            if is_withheld:
                st.info("Sources are hidden for abstentions because the system did not return a grounded answer.")
            elif not retrieved_docs:
                st.info("No sources were retrieved for this query.")
            else:
                for idx, node in enumerate(retrieved_docs, start=1):
                    meta = node.metadata
                    st.markdown(f"**[{idx}] {meta.get('chunk_id', getattr(node, 'node_id', 'chunk'))}**")
                    st.write(node.get_content())
                    st.caption(f"PubMed: {meta.get('pubmed_id', 'n/a')} | Decision: {meta.get('final_decision', 'n/a')}")

        st.sidebar.metric("Faithfulness", f"{result.get('faithfulness_score', 0.0):.3f}")
        st.sidebar.metric("Retries", int(result.get("retry_count", 0)))

        with st.sidebar.expander("Debug"):
            st.write(
                {
                    **runtime,
                    "query_mode": classify_query_mode(query),
                    "abstained": is_withheld,
                    "retrieved_doc_count": len(retrieved_docs),
                    "faithfulness_threshold": float(getattr(app_config, "FAITHFULNESS_THRESHOLD", 0.4)),
                    "faithfulness_score": float(result.get("faithfulness_score", 0.0) or 0.0),
                    "retry_count": int(result.get("retry_count", 0) or 0),
                }
            )
            st.caption(abstention_reason)
            critic_feedback = str(result.get("critic_feedback", "") or "").strip()
            if critic_feedback:
                st.markdown("**Critic feedback**")
                st.code(critic_feedback)
            draft_answer = str(result.get("draft_answer", "") or "").strip()
            if draft_answer:
                st.markdown("**Draft answer**")
                st.code(draft_answer)

        for idx, node in enumerate(retrieved_docs, start=1):
            with st.sidebar.expander(f"Chunk {idx}"):
                st.write(node.get_content())
                st.write(
                    {
                        "chunk_id": node.metadata.get("chunk_id", getattr(node, "node_id", "unknown_chunk")),
                        "bm25_or_dense_score": getattr(node, "score", None),
                        "rerank_score": getattr(node, "score", None),
                    }
                )


if __name__ == "__main__":
    main()
