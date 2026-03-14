from __future__ import annotations

from typing import Any

from config import config as app_config

import torch
from llama_index.core import Settings
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer

from src.utils.memory import flush_gpu


class QuantizedHFLLM(CustomLLM):
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    max_new_tokens: int = 96
    min_new_tokens: int = 64
    temperature: float = 0.0
    context_window: int = 512
    do_sample: bool = False
    repetition_penalty: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    num_beams: int = 1

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=getattr(self.model.config, "_name_or_path", self.model.__class__.__name__),
            is_chat_model=False,
        )

    def _build_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_window,
        )
        return {name: tensor.to(self.model.device) for name, tensor in inputs.items()}

    def _generate_ids(self, prompt: str, **kwargs: Any) -> torch.Tensor:
        inputs = self._build_inputs(prompt)
        timeout_s = kwargs.get(
            "max_time",
            getattr(app_config, "ANSWER_TIMEOUT_SECONDS", 120),
        )
        do_sample = kwargs.get("do_sample", self.do_sample)
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "min_new_tokens": kwargs.get("min_new_tokens", self.min_new_tokens),
            "do_sample": do_sample,
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
            "num_beams": kwargs.get("num_beams", self.num_beams),
            "max_time": float(timeout_s),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = kwargs.get("temperature", max(self.temperature, 1e-5))
            gen_kwargs["top_p"] = kwargs.get("top_p", self.top_p)
            gen_kwargs["top_k"] = kwargs.get("top_k", self.top_k)
        try:
            with torch.inference_mode():
                return self.model.generate(**inputs, **gen_kwargs)
        finally:
            flush_gpu()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        output = self._generate_ids(prompt, **kwargs)
        prompt_ids = self._build_inputs(prompt)["input_ids"]
        new_ids = output[0][prompt_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        inputs = self._build_inputs(prompt)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        do_sample = kwargs.get("do_sample", self.do_sample)
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "min_new_tokens": kwargs.get("min_new_tokens", self.min_new_tokens),
            "do_sample": do_sample,
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
            "num_beams": kwargs.get("num_beams", self.num_beams),
            "max_time": float(
                kwargs.get(
                    "max_time",
                    getattr(app_config, "ANSWER_TIMEOUT_SECONDS", 120),
                )
            ),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = kwargs.get("temperature", max(self.temperature, 1e-5))
            gen_kwargs["top_p"] = kwargs.get("top_p", self.top_p)
            gen_kwargs["top_k"] = kwargs.get("top_k", self.top_k)

        import threading

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        partial = ""
        for token in streamer:
            partial += token
            yield CompletionResponse(text=partial, delta=token)
        thread.join()
        flush_gpu()


def register_llm(llm: QuantizedHFLLM) -> None:
    Settings.llm = llm
