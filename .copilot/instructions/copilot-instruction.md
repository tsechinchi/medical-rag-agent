---
applyTo: "**"
---

# GitHub Copilot Instructions — Self-Correcting Medical RAG Agent

## Project Overview
This is a medical question-answering system using a self-correcting RAG pipeline.
The agent retrieves PubMed abstracts, generates answers with BioMistral-7B (4-bit
quantized), and iteratively verifies faithfulness via a LangGraph orchestration loop.

## Stack
- Orchestration: LangGraph (StateGraph)
- Retrieval: LlamaIndex (QueryFusionRetriever = FAISS + BM25)
- Reranker: sentence-transformers CrossEncoder (CPU)
- Generator: BioMistral/BioMistral-7B via BitsAndBytes NF4 4-bit
- Embeddings: BAAI/bge-small-en-v1.5
- Finetuning: TRL SFTTrainer + PEFT QLoRA
- Evaluation: RAGAS (local LLM judge) + BERTScore
- Interface: Streamlit
- Package manager: uv (Python 3.11)
- Hardware target: single T4 GPU (16GB VRAM)

---

## Project Structure
```
medical-rag-agent/
├── config/config.py          # all constants — import from here, never hardcode
├── src/
│   ├── data/
│   │   ├── download.py       # PubMed QA download + train/eval split
│   │   ├── preprocess.py     # SentenceSplitter chunking
│   │   └── build_indices.py  # FAISS + BM25 index builder with MD5 cache
│   ├── model/
│   │   ├── loader.py         # BitsAndBytes config + model loading + QLoRA toggle
│   │   ├── llm_wrapper.py    # LlamaIndex CustomLLM wrapper for BioMistral
│   │   └── prompts.py        # prompt templates
│   ├── retrieval/
│   │   ├── hybrid.py         # QueryFusionRetriever setup
│   │   └── reranker.py       # CrossEncoder reranker (CPU)
│   ├── graph/
│   │   ├── state.py          # AgentState TypedDict
│   │   ├── graph.py          # StateGraph assembly + conditional edges
│   │   └── nodes/
│   │       ├── planner.py    # query decomposition node
│   │       ├── retriever.py  # hybrid retrieval node
│   │       ├── generator.py  # BioMistral generation node
│   │       ├── critic.py     # NLI faithfulness scoring node
│   │       └── synthesizer.py # final answer + safety filter node
│   ├── evaluation/
│   │   ├── ragas_eval.py     # RAGAS with local LLM judge
│   │   ├── bertscore_eval.py # BERTScore F1
│   │   ├── ablations.py      # ablation experiment runner
│   │   └── plot_results.py   # result visualisations
│   ├── finetune/
│   │   └── qlora_train.py    # TRL SFTTrainer + PEFT QLoRA
│   ├── utils/
│   │   ├── seed.py           # set_seed(42)
│   │   └── memory.py         # flush_gpu()
│   └── app.py                # Streamlit interface
├── data/
│   ├── raw/                  # train.jsonl (800) + eval.jsonl (200)
│   ├── indices/              # persisted FAISS + BM25 indices
│   └── eval/test_set.json    # 50-question RAGAS test set
└── experiments/              # CSV result files — always committed
```

---

## AgentState Schema
```python
# src/graph/state.py
class AgentState(TypedDict):
    query: str
    sub_queries: list[str]
    retrieved_docs: list        # LlamaIndex NodeWithScore objects
    draft_answer: str
    faithfulness_score: float   # 0.0 to 1.0
    retry_count: int
    final_answer: str
    citations: list[str]
```
Every node receives and returns AgentState. Nodes must never mutate state
in place — always return `{**state, "field": new_value}`.

---

## Critical Constants (always import from config/config.py)
```python
MODEL_ID = "BioMistral/BioMistral-7B"
REVISION = "6bf2f09471b6b8d0e50533a8e81ca60ec9c2a272"
SEED = 42
FAITHFULNESS_THRESHOLD = 0.6
MAX_RETRIES = 2
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
USE_FINETUNED = True
```

---

## Coding Rules

### General
- Always import constants from `config/config.py` — never hardcode model names,
  thresholds, or seeds inline
- Always call `set_seed(SEED)` from `src/utils/seed.py` at the top of any
  script that trains or evaluates
- Always call `flush_gpu()` from `src/utils/memory.py` after each BioMistral
  inference call to prevent VRAM flooding
- Use `use_async=False` on all LlamaIndex retrievers — async causes concurrent
  GPU calls that OOM on T4

### LangGraph nodes
- Every node function signature: `def node_name(state: AgentState) -> AgentState`
- Never mutate state dict in place — always return `{**state, "key": value}`
- Conditional edge logic lives in `src/graph/graph.py`, not inside nodes
- The retry counter must be incremented in the conditional edge, not in the
  critic node

### GPU / Memory
- BioMistral runs on GPU, CrossEncoder reranker runs on CPU — never move the
  CrossEncoder to GPU
- After every `model.generate()` call, call `flush_gpu()`
- Set `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"` in
  loader.py before model loading
- Max sequence length: 512 tokens for generation, 256 for finetuning

### LlamaIndex
- All integration packages are separate installs — use the pattern
  `llama-index-{category}-{name}` e.g. `llama-index-embeddings-huggingface`
- Persist indices with `vector_store.persist()` and load with
  `FaissVectorStore.from_persist_path()` — never use `faiss.read_index()`
  directly on a LlamaIndex-managed store
- `faiss.IndexFlatIP` Pylance warnings are false alarms — add `# type: ignore`

### Evaluation
- RAGAS must use a local LLM judge — never use default OpenAI config
- BERTScore model: `microsoft/deberta-xlarge-mnli`
- All eval runs use `--seed 42` and save results to `experiments/`
- Test set (`data/eval/eval.jsonl`) must never be added to the retrieval index

### Finetuning (QLoRA)
- Always use QLoRA (4-bit base + LoRA adapters) — full finetuning will OOM
- LoRA config: `r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]`
- SFTConfig: `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`,
  `gradient_checkpointing=True`, `max_seq_length=512`, `bf16=True`
- Load finetuned model via `PeftModel.from_pretrained()` — toggled by
  `USE_FINETUNED` flag in config

### Package management
- Use `uv add <package>` — never `pip install`
- Python version is pinned to 3.11 (`requires-python = ">=3.11, <3.13"`)
- CPU extra: `faiss-cpu`, GPU extra: `faiss-gpu-cu12; sys_platform == 'linux'`
- Always run `uv lock` after adding packages and commit `uv.lock`

---

## Common Mistakes to Avoid
- Do not use `faiss.read_index()` for LlamaIndex-managed FAISS stores
- Do not use `use_async=True` on any retriever
- Do not load CrossEncoder on GPU
- Do not call OpenAI API anywhere — this is a fully offline pipeline
- Do not add eval.jsonl rows to the FAISS or BM25 index
- Do not hardcode `"BioMistral/BioMistral-7B"` inline — use `MODEL_ID`
- Do not create new top-level files — follow the existing directory structure
- Do not use `pip install` — use `uv add`
- Do not mutate AgentState in place inside nodes
