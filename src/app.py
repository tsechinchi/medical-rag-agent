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
from src.model.llm_wrapper import QuantizedHFLLM, register_llm
from src.model.loader import load_model_and_tokenizer


def _build_and_register_llm(*, finetuned: bool = True) -> QuantizedHFLLM:
    loaded = load_model_and_tokenizer(finetuned=finetuned)
    llm = QuantizedHFLLM(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        max_new_tokens=getattr(app_config, "GENERATION_MAX_NEW_TOKENS", 96),
        min_new_tokens=getattr(app_config, "GENERATION_MIN_NEW_TOKENS", 16),
        temperature=getattr(app_config, "GENERATION_TEMPERATURE", 0.0),
        context_window=getattr(app_config, "INFERENCE_CONTEXT_WINDOW", 512),
        do_sample=False,
        repetition_penalty=1.05,
        top_p=1.0,
        top_k=1,
        num_beams=1,
    )
    register_llm(llm)
    return llm


@st.cache_resource
def _get_app():
    # Build and register local LLM so LlamaIndex does not default to OpenAI.
    _build_and_register_llm(finetuned=getattr(app_config, "USE_FINETUNED", True))
    return compile_graph()


def main() -> None:
    st.set_page_config(page_title="Medical RAG Agent", layout="wide")
    st.title("Medical RAG Agent")

    query = st.text_input("Query")
    run = st.button("Run")

    if run and query:
        with st.spinner("Running retrieval and generation..."):
            result = _get_app().invoke({"query": query, "retry_count": 0})

        # Disclaimer is shown once here; synthesizer no longer prepends it to final_answer.
        if result.get("safety_filter_triggered"):
            st.warning(result.get("disclaimer", "Medical disclaimer: This output is informational only and must not be used as a substitute for licensed clinical judgment.").strip())

        st.markdown(result.get("final_answer", "No answer generated."))

        with st.expander("Sources"):
            for idx, node in enumerate(result.get("retrieved_docs", []), start=1):
                meta = node.metadata
                st.markdown(f"**[{idx}] {meta.get('chunk_id', getattr(node, 'node_id', 'chunk'))}**")
                st.write(node.get_content())
                st.caption(f"PubMed: {meta.get('pubmed_id', 'n/a')} | Decision: {meta.get('final_decision', 'n/a')}")

        st.sidebar.metric("Faithfulness", f"{result.get('faithfulness_score', 0.0):.3f}")
        st.sidebar.metric("Retries", int(result.get("retry_count", 0)))

        for idx, node in enumerate(result.get("retrieved_docs", []), start=1):
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