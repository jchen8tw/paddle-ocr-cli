"""Test doubles shared by the bridge tests."""
from pathlib import Path

from worker_client import WorkerError


class FakeClient:
    """Stands in for WorkerClient: records what the bridge asks the worker for."""

    def __init__(self, texts=None, fail_on=None, scores=None, fail_score=None):
        self.calls, self.texts, self.fail_on = [], texts or {}, fail_on
        self.scores, self.fail_score = scores or {}, fail_score  # scores: text -> (verdict, score)
        self.score_calls, self.events = [], []
        self.started = 0

    def ensure_started(self):
        self.started += 1

    def read(self, image, prompt, max_new_tokens, *, timeout=None):
        assert Path(image).is_file(), "the crop must exist on disk while the worker reads it"
        self.calls.append((prompt, max_new_tokens))
        self.events.append("read")
        if self.fail_on == prompt:
            raise WorkerError("the worker stopped (exited with code 3)")
        return [{"text": self.texts.get(prompt, f"text for {prompt}"), "tokens": 5, "finish": "stop"}]

    def score(self, items, *, timeout=None):
        for i in items:
            assert Path(i["image"]).is_file(), "the crop must exist on disk while the worker scores"
        self.score_calls.append(items)
        self.events.append("score")
        if self.fail_score:
            raise self.fail_score
        out = []
        for i in items:
            verdict, score = self.scores.get(i["text"], ("scored", 0.99))
            out.append({"key": i["key"], "verdict": verdict, "score": score, "detail": None if verdict == "scored" else "why"})
        return out

    def park(self):
        self.events.append("park")
