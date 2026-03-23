from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.graph.graph import compile_graph
from src.evaluation.run_metadata import annotate_with_run_metadata
from src.utils.answer_cleaning import clean_for_scoring, is_abstention, is_corrupted_output


TEST_SET_PATH = Path("data/eval/test_set.json")
OUT_PATH = Path("experiments/model_free_eval_results.csv")
_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_MAX_CONTEXT_DOCS = 3


def _load_test_set(test_set: list[dict] | None = None) -> list[dict]:
    if test_set is not None:
        return test_set
    return json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))


def _build_ragas_llm(llm):
    """Smaller instruction-tuned judge optimized for stable high-throughput scoring."""
    import json as _json
    import torch
    from langchain_core.outputs import Generation, LLMResult
    from langchain_core.prompt_values import PromptValue
    from ragas.llms import BaseRagasLLM
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Smaller judge model to keep end-to-end eval within wall-clock budget on a single L4.
    judge_model_id = "Qwen/Qwen2.5-3B-Instruct"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_id)
    judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_tokenizer.padding_side = "left"  # required for correct batched generation
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        quantization_config=bnb,
        device_map={"": 0},  # force all layers onto cuda:0; bnb-4bit cannot run on CPU
    )

    _SYSTEM = (
        "You are a JSON-only evaluation assistant. "
        "Always respond with a single valid JSON object or array. "
        "Do not include any explanation, markdown fences, or text outside the JSON."
    )

    def _extract_json(text: str) -> str:
        try:
            _json.loads(text)
            return text
        except Exception:
            pass
        for fence in ("```json", "```"):
            if fence in text:
                inner = text.split(fence, 1)[1].split("```", 1)[0].strip()
                try:
                    _json.loads(inner)
                    return inner
                except Exception:
                    text = inner
                    break
        for open_ch, close_ch in [('{', '}'), ('[', ']')]:
            start = text.find(open_ch)
            if start == -1:
                continue
            depth = 0
            for i, ch in enumerate(text[start:], start=start):
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            _json.loads(candidate)
                            return candidate
                        except Exception:
                            break
        return text

    def _generate_batch(texts: list[str]) -> list[str]:
        """Tokenize and run a batch of prompts in one model.generate() call."""
        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in texts
        ]
        inputs = judge_tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3072,
        ).to(judge_model.device)
        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=384,  # shorter JSON outputs improve throughput
                do_sample=False,
                pad_token_id=judge_tokenizer.eos_token_id,
            )
        n_input = inputs["input_ids"].shape[1]
        return [
            _extract_json(judge_tokenizer.decode(seq[n_input:], skip_special_tokens=True).strip())
            for seq in outputs
        ]

    class MistralJudgeLLM(BaseRagasLLM):
        def _render(self, prompt: PromptValue | str) -> str:
            if isinstance(prompt, str):
                return prompt
            if hasattr(prompt, "to_string"):
                return prompt.to_string()
            return str(prompt)

        def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            text = _generate_batch([self._render(prompt)])[0]
            if stop:
                for s in stop:
                    idx = text.find(s)
                    if idx != -1:
                        text = text[:idx]
            return LLMResult(generations=[[Generation(text=text)] * n])

        async def agenerate_text(self, prompt, n=1, temperature: float | None = 0.01, stop=None, callbacks=None):
            # Delegate async calls to the same deterministic single-call path.
            return self.generate_text(prompt=prompt, n=n, temperature=temperature or 0.01, stop=stop, callbacks=callbacks)

        def is_finished(self, response) -> bool:
            return True

    return MistralJudgeLLM()


def _build_ragas_embeddings():
    """Local embedding model for RAGAS — reuses project's BAAI/bge-small-en-v1.5."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=_EMBED_MODEL_NAME))


def run_model_free_evaluation(llm=None, test_set: list[dict] | None = None) -> pd.DataFrame:
    rows = _load_test_set(test_set)
    app = compile_graph()

    ragas_rows: list[dict] = []
    latencies: list[float] = []
    retry_counts: list[int] = []
    nli_faithfulness: list[float] = []

    unsupported_claims_counts: list[int] = []
    evidence_scores: list[float] = []
    citation_counts: list[int] = []
    corrupted_outputs: list[int] = []

    for row in rows:
        start = time.perf_counter()
        result = app.invoke({"query": row["question"], "retry_count": 0})
        latency = time.perf_counter() - start

        raw_answer = result.get("final_answer", result.get("draft_answer", ""))
        answer = clean_for_scoring(raw_answer)
        contexts = [node.get_content() for node in result.get("retrieved_docs", [])]
        contexts = contexts[:_MAX_CONTEXT_DOCS]

        # Extract safety metrics from state
        unsupported_count = result.get("unsupported_claims_count", 0)
        citation_count = raw_answer.count("[") if "[" in raw_answer else 0
        retrieved_docs = result.get("retrieved_docs", [])
        evidence_score = 0.0
        if retrieved_docs:
            top_doc = retrieved_docs[0]
            evidence_score = float(getattr(top_doc, "score", 0.0) or 0.0)

        ragas_rows.append({
            "user_input": row["question"],
            "response": answer,
            "retrieved_contexts": contexts if contexts else [""],
            "reference": row["ground_truth"],
        })
        latencies.append(round(latency, 4))
        retry_counts.append(result.get("retry_count", 0))
        nli_faithfulness.append(round(result.get("faithfulness_score", 0.0), 4))
        unsupported_claims_counts.append(unsupported_count)
        evidence_scores.append(evidence_score) #type: ignore
        citation_counts.append(citation_count)
        corrupted_outputs.append(1 if is_corrupted_output(raw_answer) else 0)

    # Always build a normalized base frame so downstream evaluators can reuse predictions.
    df = pd.DataFrame(
        {
            "question": [r["user_input"] for r in ragas_rows],
            "answer": [r["response"] for r in ragas_rows],
            "ground_truth": [r["reference"] for r in ragas_rows],
            "faithfulness": nli_faithfulness,
            "faithfulness_nli": nli_faithfulness,
            "faithfulness_ragas": [None] * len(ragas_rows),
            "answer_relevancy": [None] * len(ragas_rows),
            "context_precision": [None] * len(ragas_rows),
            "context_recall": [None] * len(ragas_rows),
            "latency_per_query_s": latencies,
            "avg_retries": retry_counts,
            # New safety metrics
            "abstention_detected": [1 if is_abstention(r["response"]) else 0 for r in ragas_rows],
            "unsupported_claims_count": unsupported_claims_counts,
            "citation_count": citation_counts,
            "retry_count": retry_counts,
            "evidence_score": evidence_scores,
            "corrupted_output_detected": corrupted_outputs,
        }
    )

    if llm is not None:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RAGAS is not installed. Install the optional dependency with "
                "`pip install ragas` or rerun with --skip-model-free."
            ) from exc

        ragas_dataset = Dataset.from_list(ragas_rows)
        from ragas.dataset_schema import EvaluationResult
        from ragas.run_config import RunConfig
        scores = evaluate(
            dataset=ragas_dataset,
            # Keep NLI faithfulness as default `faithfulness`, but also collect RAGAS faithfulness.
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=_build_ragas_llm(llm),
            embeddings=_build_ragas_embeddings(),
            run_config=RunConfig(
                timeout=300,
                max_retries=1,
                max_workers=2,
            ),
            raise_exceptions=False,  # return NaN for failed rows instead of aborting
            batch_size=4,
        )
        assert isinstance(scores, EvaluationResult)
        ragas_df = scores.to_pandas()

        # Copy judged metrics into the normalized output by row order.
        for col in ["answer_relevancy", "context_precision", "context_recall"]:
            if col in ragas_df.columns:
                vals = ragas_df[col].tolist()
                n = min(len(vals), len(df))
                df.loc[: n - 1, col] = vals[:n]

        # Keep RAGAS faithfulness separate if present; do not overwrite NLI faithfulness.
        if "faithfulness" in ragas_df.columns:
            vals = ragas_df["faithfulness"].tolist()
            n = min(len(vals), len(df))
            df.loc[: n - 1, "faithfulness_ragas"] = vals[:n]

    df = annotate_with_run_metadata(df, [r["user_input"] for r in ragas_rows])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    df.to_csv(Path("experiments/ragas_results.csv"), index=False)
    return df
