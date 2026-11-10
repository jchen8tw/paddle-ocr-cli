"""The worker's Judge ops (worker/serve.py): `score` and `park`, driven in-process with a fake Judge."""
import importlib.util
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_serve():
    spec = importlib.util.spec_from_file_location("worker_serve", ROOT / "worker" / "serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeJudge:
    def __init__(self):
        self.events = []

    def unpark(self):
        self.events.append("unpark")

    def park(self):
        self.events.append("park")

    def score_many(self, items):
        self.events.append(("score", [i["key"] for i in items]))
        if any(i["text"] == "BOOM" for i in items):
            raise RuntimeError("renderer exploded")
        return [{"key": i["key"], "verdict": "scored", "score": 0.5, "detail": None} for i in items]


def run(requests, judge=None, reader=None):
    serve = load_serve()
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    out = io.StringIO()
    serve.serve(reader, stdin, out, judge)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_a_score_request_is_answered_with_one_result_per_item_in_order():
    judge = FakeJudge()
    items = [{"key": "a", "image": "a.png", "text": "x", "category": "text"},
             {"key": "b", "image": "b.png", "text": "y", "category": "table"}]
    (reply,) = run([{"id": 9, "op": "score", "items": items}], judge)
    assert reply["id"] == 9 and reply["ok"] is True
    assert [r["key"] for r in reply["results"]] == ["a", "b"]
    assert judge.events == ["unpark", ("score", ["a", "b"])]  # the weights are on the GPU before scoring


def test_a_judge_that_crashes_on_a_request_gives_an_error_reply_and_the_worker_keeps_serving():
    judge = FakeJudge()
    boom = [{"key": "a", "image": "a.png", "text": "BOOM", "category": "text"}]
    fine = [{"key": "b", "image": "b.png", "text": "ok", "category": "text"}]
    replies = run([{"id": 1, "op": "score", "items": boom}, {"id": 2, "op": "score", "items": fine}], judge)
    assert replies[0]["ok"] is False and "renderer exploded" in replies[0]["error"]
    assert replies[1]["ok"] is True


def test_park_frees_the_judge_for_a_reader_that_needs_the_memory():
    judge = FakeJudge()
    (reply,) = run([{"id": 3, "op": "park"}], judge)
    assert reply == {"id": 3, "ok": True} and judge.events == ["park"]


def test_a_worker_without_a_judge_refuses_score_and_a_worker_without_a_reader_refuses_read():
    replies = run([{"id": 1, "op": "score", "items": []},
                   {"id": 2, "op": "read", "image": "a.png", "prompt": "p", "max_new_tokens": 1},
                   {"id": 3, "op": "park"}])
    assert [r["ok"] for r in replies] == [False, False, True]  # park with nothing to park is a no-op
    assert "Judge" in replies[0]["error"] and "Reader" in replies[1]["error"]
