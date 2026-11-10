"""Test doubles shared by the bridge tests."""
from pathlib import Path

from worker_client import WorkerError


class FakeClient:
    """Stands in for WorkerClient: records what the bridge asks the worker for."""

    def __init__(self, texts=None, fail_on=None):
        self.calls, self.texts, self.fail_on = [], texts or {}, fail_on
        self.started = 0

    def ensure_started(self):
        self.started += 1

    def read(self, image, prompt, max_new_tokens, *, timeout=None):
        assert Path(image).is_file(), "the crop must exist on disk while the worker reads it"
        self.calls.append((prompt, max_new_tokens))
        if self.fail_on == prompt:
            raise WorkerError("the worker stopped (exited with code 3)")
        return [{"text": self.texts.get(prompt, f"text for {prompt}"), "tokens": 5, "finish": "stop"}]
