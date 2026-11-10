"""The worker's request loop (worker/serve.py), driven in-process with a fake model: no torch, no GPU."""
import importlib.util
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_serve():
    spec = importlib.util.spec_from_file_location("worker_serve", ROOT / "worker" / "serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not import torch at module level
    return mod


class FakeReader:
    def __init__(self):
        self.calls = []

    def read(self, image, prompt, max_new_tokens, decode=None):
        self.calls.append((image, prompt, max_new_tokens) if decode is None else (image, prompt, max_new_tokens, decode))
        if prompt == "BOOM":
            raise RuntimeError("CUDA out of memory")
        return [{"text": f"read {Path(image).name}", "tokens": 3, "finish": "stop"}]


def run(requests, reader=None):
    serve = load_serve()
    stdin = io.StringIO("".join(r if isinstance(r, str) else json.dumps(r) + "\n" for r in requests))
    out = io.StringIO()
    serve.serve(reader or FakeReader(), stdin, out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_a_read_request_is_answered_with_its_id_and_the_readings():
    reader = FakeReader()
    replies = run([{"id": 7, "op": "read", "image": "/tmp/x/b12.png", "prompt": "Text Recognition:",
                    "max_new_tokens": 512}], reader)
    assert replies == [{"id": 7, "ok": True, "readings": [{"text": "read b12.png", "tokens": 3, "finish": "stop"}]}]
    assert reader.calls == [("/tmp/x/b12.png", "Text Recognition:", 512)]


def test_a_sampled_read_passes_its_decoding_settings_on_and_a_greedy_one_passes_none():
    reader = FakeReader()
    decode = {"mode": "sample", "temperature": 0.3, "top_k": 20, "top_p": 0.8, "n": 4, "seed": 99}
    run([{"id": 1, "op": "read", "image": "a.png", "prompt": "p", "max_new_tokens": 5, "decode": decode},
         {"id": 2, "op": "read", "image": "a.png", "prompt": "p", "max_new_tokens": 5}], reader)
    assert reader.calls == [("a.png", "p", 5, decode), ("a.png", "p", 5)]


def test_the_loop_ends_on_quit_and_ignores_what_follows():
    replies = run([{"op": "quit"}, {"id": 1, "op": "read", "image": "a.png", "prompt": "p", "max_new_tokens": 1}])
    assert replies == []


def test_the_loop_ends_at_end_of_input():
    assert run([]) == []


def test_a_failing_read_is_reported_and_the_worker_keeps_serving():
    replies = run([
        {"id": 1, "op": "read", "image": "a.png", "prompt": "BOOM", "max_new_tokens": 1},
        {"id": 2, "op": "read", "image": "b.png", "prompt": "ok", "max_new_tokens": 1},
    ])
    assert replies[0]["id"] == 1 and replies[0]["ok"] is False and "out of memory" in replies[0]["error"]
    assert replies[1]["ok"] is True


def test_unknown_ops_and_garbage_lines_get_an_error_reply():
    replies = run([{"id": 3, "op": "dance"}, "not json\n", {"id": 4, "op": "read"}])
    assert [r["ok"] for r in replies] == [False, False, False]
    assert replies[0]["id"] == 3
    assert replies[2]["id"] == 4  # a read without its fields is a request error, not a crash
