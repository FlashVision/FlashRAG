"""
LLM-based answer generation with context injection.

Wraps HuggingFace causal LMs for RAG-style generation: retrieves contexts
are formatted via a prompt template and fed to the LLM for grounded output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from flashrag.generation.prompt_templates import PromptTemplate, get_template
from flashrag.registry import GENERATORS
from flashrag.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    answer: str
    prompt: str
    contexts: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@GENERATORS.register("huggingface")
class RAGGenerator:
    """
    Generate answers grounded in retrieved context using a HuggingFace LLM.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"gpt2"`` or ``"mistralai/Mistral-7B-v0.1"``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling probability.
    prompt_template : str or PromptTemplate
        Template name or instance for formatting the RAG prompt.
    load_in_4bit : bool
        Load model with 4-bit quantization (requires bitsandbytes).
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        prompt_template: str | PromptTemplate = "default",
        load_in_4bit: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample

        if isinstance(prompt_template, str):
            self.template = get_template(prompt_template)
        else:
            self.template = prompt_template

        logger.info(f"Loading generator model '{model_name}'...")

        load_kwargs: dict[str, Any] = {}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["device_map"] = "auto"
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if not load_in_4bit:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, **load_kwargs
            ).to(device)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )

        self._model.eval()
        logger.info(f"Generator ready: {model_name} (device={device})")

    def generate(
        self,
        question: str,
        contexts: list[str] | None = None,
        sources: list[str] | None = None,
        search_results: list[SearchResult] | None = None,
        template: PromptTemplate | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """
        Generate a grounded answer for the question using provided contexts.
        """
        import torch

        tmpl = template or self.template

        ctx_texts = contexts or []
        src_labels = sources or []
        src_metadata: list[dict[str, Any]] = []

        if search_results:
            ctx_texts = [r.text for r in search_results]
            src_labels = [r.metadata.get("source", f"doc_{r.vector_id}") for r in search_results]
            src_metadata = [r.metadata for r in search_results]

        prompt = tmpl.format(question=question, contexts=ctx_texts, sources=src_labels)

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_params = {
            "max_new_tokens": gen_kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": gen_kwargs.get("temperature", self.temperature),
            "top_p": gen_kwargs.get("top_p", self.top_p),
            "top_k": gen_kwargs.get("top_k", self.top_k),
            "do_sample": gen_kwargs.get("do_sample", self.do_sample),
            "pad_token_id": self._tokenizer.pad_token_id,
        }

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_params)

        generated_ids = output_ids[0][input_len:]
        answer = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return GenerationResult(
            answer=answer,
            prompt=prompt,
            contexts=ctx_texts,
            sources=[{"source": s, **m} for s, m in zip(src_labels, src_metadata)]
            if src_metadata
            else [{"source": s} for s in src_labels],
            metadata={"model": self.model_name, "gen_params": gen_params},
        )

    def generate_simple(self, prompt: str, **gen_kwargs: Any) -> str:
        """Generate text from a raw prompt without RAG context formatting."""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_params = {
            "max_new_tokens": gen_kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": gen_kwargs.get("temperature", self.temperature),
            "do_sample": gen_kwargs.get("do_sample", self.do_sample),
            "pad_token_id": self._tokenizer.pad_token_id,
        }

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_params)

        generated_ids = output_ids[0][input_len:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
