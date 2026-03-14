from __future__ import annotations

import streamlit as st

from src.graph.graph import compile_graph


@st.cache_resource
def _get_app():
    return compile_graph()


def main() -> None:
    st.set_page_config(page_title="Medical RAG Agent", layout="wide")
    st.title("Medical RAG Agent")

    query = st.text_input("Query")
    run = st.button("Run")

    if run and query:
        with st.spinner("Running retrieval and generation..."):
            result = _get_app().invoke({"query": query, "retry_count": 0})

        if result.get("safety_filter_triggered"):
            st.warning(result.get("disclaimer", "Medical disclaimer triggered."))

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