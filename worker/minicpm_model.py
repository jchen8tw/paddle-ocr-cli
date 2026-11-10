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

    def read(self, image_path: str, prompt: str, max_new_tokens: int) -> list[dict]:
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        # The image comes before the text; thinking is off.
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=False).to(self.device)
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, no_repeat_ngram_size=100)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        n = int(new_tokens.shape[0])
        return [{"text": text, "tokens": n, "finish": "length" if n >= max_new_tokens else "stop"}]
