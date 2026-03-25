import os
from pathlib import Path


MODEL_ID = "BioMistral/BioMistral-7B"
# Use a moving ref by default to avoid broken pinned snapshot SHAs.
REVISION = "main"
MODEL_CACHE_DIR = str(Path("models") / "biomistral-7b")
FINETUNED_ADAPTER_PATH = "data/qlora_checkpoints/final"
USE_FINETUNED = False

# Recovery options when USE_FINETUNED=True but FINETUNED_ADAPTER_PATH is missing.
# 1) Reuse latest local checkpoint with adapter files.
AUTO_RESUME_LATEST_CHECKPOINT_ADAPTER = True
# 2) Optionally download adapter snapshot from Hugging Face Hub.
AUTO_DOWNLOAD_FINETUNED_ADAPTER = True
FINETUNED_ADAPTER_REPO_ID = ""
FINETUNED_ADAPTER_REVISION = "main"

# Inference defaults stay conservative so the repo can run with minimal setup.
INFERENCE_MODEL_ID = MODEL_ID
INFERENCE_REVISION = REVISION
INFERENCE_CONTEXT_WINDOW = 1536
GENERATION_MAX_NEW_TOKENS = 64
GENERATION_MIN_NEW_TOKENS = 16
GENERATION_TEMPERATURE = 0.0
MODEL_LOAD_TIMEOUT_SECONDS = 300
ANSWER_TIMEOUT_SECONDS = 45

# Device selection for eval components: "auto", "cuda", or "cpu".
BERTSCORE_DEVICE = "auto"
CRITIC_DEVICE = "auto"
RERANK_DEVICE = "auto"
EMBEDDING_DEVICE = "auto"

# Retrieval / reranking defaults are intentionally light for notebook runs.
RETRIEVAL_SIMILARITY_TOP_K = 8
BM25_SIMILARITY_TOP_K = 6
RERANK_TOP_N = 3
RETRIEVAL_FUSION_NUM_QUERIES = 1
PLANNER_MAX_SUBQUERIES = 1
DRUG_LOOKUP_MAX_SUBQUERIES = 2
MAX_CONTEXT_DOCS = 2
MIN_CONTEXT_DOCS = 2

SEED = 42

FAITHFULNESS_THRESHOLD = 0.4
MAX_RETRIES = 0

# Sentence-level entailment floor used by the critic to mark individual claims
# as unsupported; values below this are considered weak evidence.
CRITIC_SENTENCE_SUPPORT_THRESHOLD = 0.65
CRITIC_BATCH_SIZE = 12

# If True, any unsupported claim found by the critic can trigger a retry even
# when aggregate faithfulness is near threshold.
RETRY_ON_UNSUPPORTED_CLAIMS = False

# Cross-encoder rerank score gate: chunks scoring below this value are dropped
# before reaching the generator.
RERANK_MIN_SCORE = -999
# Keep only reranked chunks close to the best score to reduce context noise.
RERANK_SCORE_MARGIN = 1.0

# Hard cap on context chunks passed to generation after rerank filtering.
MAX_CONTEXT_DOCS = 2

# Soft floor to keep enough evidence chunks when rerank scores are noisy.
MIN_CONTEXT_DOCS = 2

# If the top retrieved chunk score falls below this floor, return a concise
# insufficient-evidence answer instead of attempting speculative generation.
LOW_EVIDENCE_SCORE_FLOOR = -999

# Lightweight lexical relevance gate to suppress clearly off-topic chunks that
# occasionally survive reranking.
ENABLE_DOMAIN_RELEVANCE_GATE = True

# Minimum overlap ratio between query keywords and candidate text/metadata.
DOMAIN_RELEVANCE_MIN_OVERLAP = 0.08

CHUNK_SIZE = 256
CHUNK_OVERLAP = 32

FAITHFULNESS_THRESHOLD_GRID = [0.5, 0.6, 0.7, 0.8]
CHUNK_SIZE_GRID = [128, 256, 512]

# Quantization for a single-GPU notebook setup.
LOAD_IN_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True

# Evaluation defaults.
BERTSCORE_MODEL_TYPE = "microsoft/deberta-v3-base"
EVAL_TIME_BUDGET_SECONDS = 10800
EVAL_JUDGE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EVAL_JUDGE_MAX_NEW_TOKENS = 128
EVAL_JUDGE_BATCH_SIZE = 4
EVAL_JUDGE_TIMEOUT_SECONDS = 180
