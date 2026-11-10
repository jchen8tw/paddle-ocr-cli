"""MiniCPM-OCR Jev on the GPU: the Judge's model (docs/adr/0002).

Follows the official transformers adapter of the MiniCPM-OCR SDK (scripts/minicpm-ocr-jev/sdk/jev_score/
backends/transformers_backend.py, Apache-2.0): load as `Qwen3_5ForSequenceClassification` (the class named
in config.json would initialise the score head randomly), bf16, SDPA attention, the official raw prompt
string with no chat template, the original crop first and the render of the Reading second, and
Score = softmax(logits)[1]. One pair of images per forward: Scores drift a little with the batch size.
"""
from __future__ import annotations

import hashlib
import math

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForSequenceClassification
from transformers.utils import logging as hf_logging

from judging import EncodeError

hf_logging.set_verbosity_error()

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
INSTRUCTION = (
    "第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
    "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
    "这是一个二分类任务：一致标为 1，不一致标为 0。"
)
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPT_TEMPLATE_MD5 = "9aab68e8ba8ab4da17ba9752bddb199c"  # md5 of build_prompt("{text}"), from the SDK
POOLED_TOKEN_ID = 271  # the value head pools the last non-pad token: the final "\n" of the empty think block
MAX_ASPECT_RATIO = 199.0


def build_prompt(text: str) -> str:
    """The official prompt. `text` goes in exactly as given."""
    return (
        f"<|im_start|>user\n{IMAGE_PLACEHOLDER}\nOCR文本：\n{text}\n{IMAGE_PLACEHOLDER}\n"
        f"{INSTRUCTION}{ASSISTANT_SUFFIX}"
    )


assert hashlib.md5(build_prompt("{text}").encode()).hexdigest() == PROMPT_TEMPLATE_MD5, "prompt drifted from the SDK's"


def pad_extreme_aspect(img: Image.Image) -> Image.Image:
    """The SDK pads (white, centred) an image whose aspect ratio is above 199:1; everything else is untouched."""
    w, h = img.size
    if max(w, h) / max(min(w, h), 1) <= MAX_ASPECT_RATIO:
        return img
    if w >= h:
        new_w, new_h = w, max(h, int(w / MAX_ASPECT_RATIO + 0.999999))
    else:
        new_h, new_w = h, max(w, int(h / MAX_ASPECT_RATIO + 0.999999))
    canvas = Image.new("RGB", (new_w, new_h), "white")
    canvas.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
    return canvas


class JevJudge:
    def __init__(self, model_dir, device: str = "cuda"):
        self.device = device
        self.parked = False
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.processor.tokenizer.padding_side = "right"
        model, info = Qwen3_5ForSequenceClassification.from_pretrained(
            model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
        if info["missing_keys"] or info["unexpected_keys"]:
            raise RuntimeError(f"Jev did not load cleanly: missing {sorted(info['missing_keys'])}, "
                               f"unexpected {sorted(info['unexpected_keys'])}")
        if model.config.num_labels != 2:
            raise RuntimeError(f"Jev has {model.config.num_labels} labels, expected 2")
        self.model = model.to(device).eval()

    def score(self, original_path: str, render_path: str, text: str) -> float:
        """Probability that `text` and its render agree with the original. Raises on any failure."""
        try:
            with Image.open(original_path) as a, Image.open(render_path) as b:
                images = [pad_extreme_aspect(a.convert("RGB")), pad_extreme_aspect(b.convert("RGB"))]
            inputs = self.processor(text=[build_prompt(text)], images=images, return_tensors="pt")
            ids = inputs["input_ids"][0]
            pooled = int(ids[(ids != self.model.config.pad_token_id).nonzero().max()])
        except Exception as e:
            raise EncodeError(f"{type(e).__name__}: {e}") from e
        if pooled != POOLED_TOKEN_ID:
            raise EncodeError(f"the prompt ends in token {pooled}, expected {POOLED_TOKEN_ID}")
        with torch.inference_mode():
            a_logit, b_logit = self.model(**inputs.to(self.device)).logits.float()[0].tolist()
        top = max(a_logit, b_logit)
        return math.exp(b_logit - top) / (math.exp(a_logit - top) + math.exp(b_logit - top))

    def free_cache(self) -> None:
        torch.cuda.empty_cache()

    def park(self) -> None:
        """Move the weights to the CPU and free the GPU (for a Reader that needs the memory; see serve.py)."""
        if not self.parked:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self.parked = True

    def unpark(self) -> None:
        if self.parked:
            self.model.to(self.device)
            self.parked = False
