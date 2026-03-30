"""
EGX Predictor — Layer 7.

High-level inference and generation API.
Handles batched prediction, text generation with sampling strategies,
and model output formatting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from egx.core.device import get_default_device

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

logger = logging.getLogger("egx.api.predictor")


class EGXPredictor:
    """
    Standalone predictor for inference and text generation.

    Supports:
    - Batched forward pass
    - Autoregressive text generation with configurable decoding
    - Temperature, top-k, top-p sampling
    - Beam search (via HuggingFace generate())
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_default_device()

    def __repr__(self) -> str:
        return f"EGXPredictor(device='{self.device}')"

    def predict(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
    ) -> Any:
        """
        Run a single forward pass and return raw model outputs.

        Args:
            model: The model to run inference on.
            inputs: Dictionary with input tensors (input_ids, attention_mask, etc.).

        Returns:
            Model outputs (logits, hidden states, etc.).
        """
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            input_batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type=device_type):
                outputs = model(**input_batch)

        return outputs

    def generate(
        self,
        model: nn.Module,
        prompts: Union[str, List[str]],
        tokenizer: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> List[str]:
        """
        Generate text from prompts using the model.

        This supports both HuggingFace models (with .generate()) and
        custom models (with manual autoregressive decoding).

        Args:
            model: Language model.
            prompts: Single prompt string or list of prompts.
            tokenizer: Tokenizer for encoding/decoding.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            do_sample: Whether to use sampling (vs greedy).
            num_beams: Beam search width.
            repetition_penalty: Penalty for repeated tokens.

        Returns:
            List of generated text strings.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        model.to(self.device)
        model.eval()

        # Tokenize inputs
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Use HuggingFace .generate() if available
        if hasattr(model, "generate"):
            with torch.no_grad():
                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "do_sample": do_sample,
                    "num_beams": num_beams,
                    "repetition_penalty": repetition_penalty,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                }
                gen_kwargs.update(kwargs)
                output_ids = model.generate(**gen_kwargs)

            # Decode only the newly generated tokens
            generated_texts = []
            for i, ids in enumerate(output_ids):
                new_tokens = ids[input_ids.shape[1] :]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                generated_texts.append(text)

            return generated_texts

        # Manual autoregressive fallback for custom models
        return self._manual_generate(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            max_new_tokens,
            temperature,
            top_k,
            do_sample,
        )

    def _manual_generate(
        self,
        model: nn.Module,
        input_ids: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"],
        tokenizer: Any,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        do_sample: bool,
    ) -> List[str]:
        """Autoregressive generation loop for models without .generate()."""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=generated, attention_mask=attention_mask)

            # Get logits for the last token
            if hasattr(outputs, "logits"):
                logits = outputs.logits[:, -1, :]
            elif isinstance(outputs, torch.Tensor):
                logits = outputs[:, -1, :]
            else:
                break

            # Temperature scaling
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                indices_to_remove = (
                    logits < torch.topk(logits, top_k_val)[0][..., -1, None]
                )
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=-1
                )

            # Stop on EOS
            if tokenizer.eos_token_id is not None:
                if (next_token == tokenizer.eos_token_id).all():
                    break

        results = []
        for ids in generated:
            new_tokens = ids[input_ids.shape[1] :]
            results.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

        return results
