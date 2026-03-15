MODEL_ID = "BioMistral/BioMistral-7B"
REVISION = "6bf2f09471b6b8d0e50533a8e81ca60ec9c2a272"
FINETUNED_ADAPTER_PATH = "data/qlora_checkpoints/final"
USE_FINETUNED = True

# Inference defaults must follow the project baseline model config.
INFERENCE_MODEL_ID = MODEL_ID
INFERENCE_REVISION = REVISION
INFERENCE_CONTEXT_WINDOW = 512
# None means "use the full remaining model context budget".
GENERATION_MAX_NEW_TOKENS = None
GENERATION_MIN_NEW_TOKENS = 16
GENERATION_TEMPERATURE = 0.0
MODEL_LOAD_TIMEOUT_SECONDS = 300
ANSWER_TIMEOUT_SECONDS = 120
RETRIEVAL_SIMILARITY_TOP_K = 12
BM25_SIMILARITY_TOP_K = 8
RERANK_TOP_N = 5
RETRIEVAL_FUSION_NUM_QUERIES = 3
RERANK_DEVICE = "cpu"
EMBEDDING_DEVICE = "auto"
PLANNER_MAX_SUBQUERIES = 1
DRUG_LOOKUP_MAX_SUBQUERIES = 3

SEED = 42

FAITHFULNESS_THRESHOLD = 0.7
MAX_RETRIES = 2

# Sentence-level entailment floor used by the critic to mark individual claims
# as unsupported; values below this are considered weak evidence.
CRITIC_SENTENCE_SUPPORT_THRESHOLD = 0.7

# If True, any unsupported claim found by the critic can trigger a retry even
# when aggregate faithfulness is near threshold.
RETRY_ON_UNSUPPORTED_CLAIMS = True

# Cross-encoder rerank score gate: chunks scoring below this value are dropped
# before reaching the generator.  ms-marco-MiniLM-L-6-v2 emits raw logits;
# 0.0 ≈ 50% relevance — anything below is considered off-topic.
RERANK_MIN_SCORE = 0.1

# Keep only reranked chunks close to the best score to reduce context noise.
RERANK_SCORE_MARGIN = 1.5

# Hard cap on context chunks passed to generation after rerank filtering.
MAX_CONTEXT_DOCS = 5

# Soft floor to keep enough evidence chunks when rerank scores are noisy.
MIN_CONTEXT_DOCS = 3

# If the top retrieved chunk score falls below this floor, return a concise
# insufficient-evidence answer instead of attempting speculative generation.
LOW_EVIDENCE_SCORE_FLOOR = 0.2

# Lightweight lexical relevance gate to suppress clearly off-topic chunks that
# occasionally survive reranking.
ENABLE_DOMAIN_RELEVANCE_GATE = True

# Minimum overlap ratio between query keywords and candidate text/metadata.
DOMAIN_RELEVANCE_MIN_OVERLAP = 0.08

CHUNK_SIZE = 256
CHUNK_OVERLAP = 32

FAITHFULNESS_THRESHOLD_GRID = [0.5, 0.6, 0.7, 0.8]
CHUNK_SIZE_GRID = [128, 256, 512]

# ── Quantization (T4 / 16 GB VRAM) ───────────────────────────────────────
# BioMistral-7B at fp16 ≈ 14 GB — too tight for a T4 once activations and
# the KV cache are added.  4-bit NF4 via bitsandbytes brings VRAM down to
# ~3.5 GB, leaving plenty of headroom for the retrieval pipeline.
LOAD_IN_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"   # bf16 math inside 4-bit layers
BNB_4BIT_QUANT_TYPE = "nf4"           # NormalFloat4 — best quality for LLMs
BNB_4BIT_USE_DOUBLE_QUANT = True      # nested quantisation saves ~0.4 GB extra
