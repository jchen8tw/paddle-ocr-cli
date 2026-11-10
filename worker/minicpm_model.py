"""MiniCPM-OCR-1.0 on the GPU: native transformers (>= 5.21), bf16, no remote code.

The main process decides the prompt and the token limit (minicpm_reader.py); this class only turns
"crop + prompt" into text. The raw text is returned: all clean-up happens in the main process.
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # e.g. the Qwen3.5 linear-attention fallback warning on every start


class MiniCPMReader:
    def __init__(self, model_dir, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()
        # The prefill would otherwise compute vocabulary logits (248k) for every prompt position, which for a
        # large table costs more than a gigabyte; generation only ever needs the last position's.
        self.model.lm_head.register_forward_pre_hook(lambda module, args: (args[0][:, -1:, :],))

    def read(self, image_path: str, prompt: str, max_new_tokens: int, decode: dict | None = None) -> list[dict]:
        """Greedy by default. `decode` {"mode": "sample", "temperature", "top_k", "top_p", "n", "seed"} draws n seeded
        samples: the seed is set right before the one generate call, so the same seed on the same GPU and library
        versions gives the same texts, in any process."""
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        # The image comes before the text; thinking is off.
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=False).to(self.device)
        if decode is None or decode.get("mode") == "greedy":
            options, n_samples = {"do_sample": False}, 1
        elif decode.get("mode") == "sample":
            n_samples = int(decode["n"])
            options = {"do_sample": True, "temperature": float(decode["temperature"]), "top_k": int(decode["top_k"]),
                       "top_p": float(decode["top_p"]), "num_return_sequences": n_samples}
            torch.manual_seed(int(decode["seed"]))
        else:
            raise ValueError(f"unknown decode mode {decode.get('mode')!r}")
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, no_repeat_ngram_size=100, **options)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        readings = []
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad = getattr(tokenizer, "pad_token_id", None)
        for row in out:
            new_tokens = row[inputs["input_ids"].shape[1]:]
            if pad is not None:  # the shorter samples of a batch are padded to the longest one
                while new_tokens.shape[0] and int(new_tokens[-1]) == pad:
                    new_tokens = new_tokens[:-1]
            n = int(new_tokens.shape[0])
            readings.append({"text": self.processor.decode(new_tokens, skip_special_tokens=True), "tokens": n,
                             "finish": "length" if n >= max_new_tokens else "stop"})
        return readings
