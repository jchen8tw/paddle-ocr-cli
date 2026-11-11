"""The main-process side of the torch worker: start it, talk JSON Lines to it, never hang on it.

The worker (worker/serve.py) lives in its own uv project because torch and paddle cannot share a
venv (docs/adr/0001). One request is in flight at a time. A worker that crashes or stops answering
turns into a WorkerError for the request in progress, never into a hang: every wait has a timeout and
checks that the process is still there, and a worker that timed out is killed (its output stream can
no longer be trusted) and may be started again with ensure_started().
"""
from __future__ import annotations

import collections
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time


class WorkerError(RuntimeError):
    """The worker could not be started, crashed, timed out or reported an error."""


def read_timeout(max_new_tokens: int, samples: int = 1) -> float:
    """Seconds to wait for one Reading (or `samples` sampled ones, which decode together): a generous floor plus
    time for the token limit at 10 tokens/s (MiniCPM decodes at 50+ tokens/s on the reference GPU), scaled by the
    number of samples."""
    return 120.0 + max_new_tokens * max(1, samples) / 10.0


def score_timeout(n_items: int) -> float:
    """Seconds to wait for a batch of Judge Scores: Chromium's launch, then ~0.5 s per Reading (0.35 s for the
    forward and ~0.15 s to render), with a wide margin."""
    return 120.0 + 5.0 * n_items


class WorkerClient:
    quit_timeout = 5.0  # seconds close() waits for the quit request to be written before it kills the worker

    def __init__(self, cmd: list[str], *, env: dict | None = None, start_timeout: float = 600.0,
                 max_restarts: int = 3):
        self.cmd, self.env = list(cmd), env
        self.start_timeout, self.max_restarts = start_timeout, max_restarts
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._stderr_tail: collections.deque = collections.deque(maxlen=15)
        self._stderr_thread: threading.Thread | None = None
        self._next_id = 0
        self._starts = 0
        self.ready: dict | None = None

    # ------------------------------------------------------------------ process lifecycle

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> dict:
        """Start the worker and wait for its ready event, which is returned."""
        self._starts += 1
        self._lines = queue.Queue()
        self._stderr_tail.clear()
        try:
            self._proc = subprocess.Popen(
                self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1, env=self.env,
                start_new_session=True,  # so that a kill reaches uv's child as well
            )
        except OSError as e:
            raise WorkerError(f"cannot start the worker ({self.cmd[0]}): {e}") from e
        threading.Thread(target=self._pump_stdout, args=(self._proc, self._lines), daemon=True).start()
        self._stderr_thread = threading.Thread(target=self._pump_stderr, args=(self._proc,), daemon=True)
        self._stderr_thread.start()
        try:
            while True:
                msg = self._next_message(self.start_timeout, what=f"did not become ready within {self.start_timeout:g}s")
                if msg.get("event") == "ready":
                    self.ready = msg
                    return msg
                if msg.get("event") == "error":
                    raise WorkerError(f"worker start-up failed ({msg.get('what', '?')}): {msg.get('message', '')}")
        except WorkerError:
            self._kill()
            raise

    def ensure_started(self) -> None:
        """Start the worker if it is not running (again, after a crash), within the restart budget."""
        if self.is_alive():
            return
        if self._starts and self._starts - 1 >= self.max_restarts:
            raise WorkerError(f"the worker has crashed {self._starts} times; not restarting it again "
                              f"(restart limit {self.max_restarts})")
        self.start()

    def close(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            self._send(json.dumps({"op": "quit"}) + "\n", self.quit_timeout)
            proc.stdin.close()
            proc.wait(timeout=15)
        except (WorkerError, OSError, ValueError, subprocess.TimeoutExpired):
            self._kill()

    def _kill(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    # ------------------------------------------------------------------ requests

    def read(self, image: str, prompt: str, max_new_tokens: int, *, decode: dict | None = None,
             timeout: float | None = None) -> list[dict]:
        """One Reading of a crop (a PNG path) for `prompt`; returns [{"text", "tokens", "finish"}]. `decode` is
        None for greedy decoding, or {"mode": "sample", "temperature", "top_k", "top_p", "n", "seed"}, which returns
        `n` Readings (same seed, same texts)."""
        req = {"op": "read", "image": str(image), "prompt": prompt, "max_new_tokens": max_new_tokens}
        if decode is not None:
            req["decode"] = decode
        n = (decode or {}).get("n", 1)
        return self._request(req, read_timeout(max_new_tokens, n) if timeout is None else timeout)["readings"]

    def score(self, items: list[dict], *, timeout: float | None = None) -> list[dict]:
        """Score a batch of Readings: items are {"key", "image" (a PNG path), "text", "category"}; returns one
        {"key", "verdict", "score", "detail"} per item, in order."""
        reply = self._request({"op": "score", "items": items},
                              score_timeout(len(items)) if timeout is None else timeout)
        return reply["results"]

    def park(self) -> None:
        """Ask the worker to move the Judge off the GPU (it comes back by itself on the next score)."""
        self._request({"op": "park"}, 120.0)

    def _request(self, req: dict, timeout: float) -> dict:
        if not self.is_alive():
            raise WorkerError("the worker is not running" + self._exit_note())
        self._next_id += 1
        req = {"id": self._next_id, **req}
        deadline = time.monotonic() + timeout
        self._send(json.dumps(req, ensure_ascii=False) + "\n", timeout)
        try:
            reply = self._next_message(max(deadline - time.monotonic(), 0.0), what=f"no answer within {timeout:g}s")
            if reply.get("id") != req["id"]:
                raise WorkerError(f"protocol error: expected an answer to request {req['id']}, got {reply!r:.200}")
        except WorkerError:
            self._kill()  # timed out or out of step: the stream cannot be trusted any more
            raise
        if not reply.get("ok"):
            raise WorkerError(f"worker error: {reply.get('error', 'unknown')}")
        return reply

    # ------------------------------------------------------------------ plumbing

    def _send(self, line: str, timeout: float) -> None:
        """Write one request line. A worker that does not read its input fills the pipe and would block this write
        for ever, so it happens on a helper thread that is waited for no longer than `timeout` (the worker is killed
        then, which also ends the thread's write)."""
        proc, failure = self._proc, []

        def write():
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as e:
                failure.append(e)

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        writer.join(timeout)
        if writer.is_alive():
            self._kill()
            writer.join(5)
            raise WorkerError(f"the request could not be sent within {timeout:g}s: the worker does not read its input"
                              + self._exit_note())
        if failure:
            raise WorkerError("the worker closed its input" + self._exit_note(wait=True))

    def _next_message(self, timeout: float, *, what: str) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerError(f"worker {what}")
            try:
                line = self._lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue  # the process liveness is decided by the end-of-output marker below
            if line is None:  # the worker closed its output: it is gone
                raise WorkerError("the worker stopped" + self._exit_note(wait=True))
            try:
                msg = json.loads(line)
            except ValueError:
                print(f"worker: {line.rstrip()}", file=sys.stderr)  # stray output; not protocol
                continue
            if isinstance(msg, dict):
                return msg

    def _exit_note(self, *, wait: bool = False) -> str:
        proc = self._proc
        if proc is None:
            return ""
        if wait:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._kill()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)
        code = proc.poll()
        note = f" (exited with code {code})" if code is not None else ""
        tail = "\n".join(self._stderr_tail)
        return note + (f"; last worker output:\n{tail}" if tail else "")

    @staticmethod
    def _pump_stdout(proc, lines: queue.Queue) -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _pump_stderr(self, proc) -> None:
        for line in proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            self._stderr_tail.append(line.rstrip("\n"))
