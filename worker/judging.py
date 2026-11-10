"""The worker's Judge: render each Reading, score it against the Block image, report an official verdict.

No torch in here: the model and the renderer are passed in, so this is tested with fakes (and the real
ones are in jev_model.py and render_env.py). Verdicts follow the MiniCPM-OCR SDK's meanings:

    scored                  a real Score
    precheck_reject         the Node KaTeX screen says the LaTeX cannot be parsed: the Reading is unusable
    render_error            the renderer failed: the Reading is unusable
    missing_original_image  the renderer got no readable original: the Reading is unusable
    encode_error            the images could not be encoded: scoring never happened
    forward_error           the model failed: scoring never happened

One item failing never fails the others.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable

RENDER_REASONS = ("precheck_reject", "render_error", "missing_original_image")


class EncodeError(Exception):
    """An image could not be read or turned into model input."""


@dataclass
class Candidate:
    case_id: str
    image_path: str
    text: str
    category: str


class RenderedJudge:
    """`render(candidates, work_dir)` returns one result per candidate (case_id, ok, render_image_path,
    reason, detail); `model.score(original_path, render_path, text)` returns a probability."""

    def __init__(self, model, render: Callable, *, on_forward_error: Callable[[Exception], None] | None = None):
        self.model, self.render, self.on_forward_error = model, render, on_forward_error

    def park(self) -> None:
        """Free the model's GPU memory for a Reader that needs it, if the model can."""
        if hasattr(self.model, "park"):
            self.model.park()

    def unpark(self) -> None:
        if hasattr(self.model, "unpark"):
            self.model.unpark()

    def score_many(self, items: list[dict]) -> list[dict]:
        """items: [{"key", "image", "text", "category"}] -> [{"key", "verdict", "score", "detail"}], in order."""
        if not items:
            return []
        candidates = [Candidate(i["key"], i["image"], i["text"], i["category"]) for i in items]
        work_dir = tempfile.mkdtemp(prefix="pdf_ocr_render_")
        try:
            try:
                renders = self.render(candidates, work_dir)
            except Exception as e:  # the renderer could not run at all (Chromium crashed, node vanished, ...)
                return [self._result(c.case_id, "render_error", None, f"{type(e).__name__}: {e}") for c in candidates]
            return [self._score_one(c, r) for c, r in zip(candidates, renders)]
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)  # the renders are only needed while scoring

    def _score_one(self, c: Candidate, r) -> dict:
        if not r.ok:
            return self._result(c.case_id, r.reason if r.reason in RENDER_REASONS else "render_error", None, r.detail)
        try:
            score = float(self.model.score(c.image_path, r.render_image_path, c.text))
        except EncodeError as e:
            return self._result(c.case_id, "encode_error", None, str(e))
        except Exception as e:
            if self.on_forward_error is not None:
                self.on_forward_error(e)
            return self._result(c.case_id, "forward_error", None, f"{type(e).__name__}: {e}")
        if not (math.isfinite(score) and 0.0 <= score <= 1.0):
            return self._result(c.case_id, "forward_error", None, f"the model returned {score!r}")
        return self._result(c.case_id, "scored", score, None)

    @staticmethod
    def _result(key: str, verdict: str, score: float | None, detail: str | None) -> dict:
        return {"key": key, "verdict": verdict, "score": score, "detail": detail}
