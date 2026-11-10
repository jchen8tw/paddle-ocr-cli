"""The main-process side of the Judge ops, against tests/fake_worker.py (no torch)."""
import sys
from pathlib import Path

import pytest

from worker_client import WorkerClient, WorkerError, score_timeout

FAKE = str(Path(__file__).parent / "fake_worker.py")


@pytest.fixture
def client():
    c = WorkerClient([sys.executable, FAKE, "--mode", "ready"], start_timeout=20)
    c.start()
    yield c
    c.close()


def items(*texts):
    return [{"key": f"k{i}", "image": f"/tmp/c{i}.png", "text": t, "category": "text"} for i, t in enumerate(texts)]


def test_score_returns_one_result_per_item_in_order(client):
    results = client.score(items("a", "b"))
    assert [(r["key"], r["verdict"], r["score"]) for r in results] == [("k0", "scored", 0.5), ("k1", "scored", 0.5)]


def test_a_failing_score_request_is_a_worker_error_and_the_worker_stays_usable(client):
    with pytest.raises(WorkerError, match="renderer exploded"):
        client.score(items("BOOM"))
    assert client.is_alive()
    assert client.score(items("fine"))[0]["verdict"] == "scored"


def test_a_crash_while_scoring_raises_with_the_stderr_tail(client):
    with pytest.raises(WorkerError, match=r"(?s)exited with code 3.*CUDA error"):
        client.score(items("CRASH"))
    assert not client.is_alive()


def test_a_score_that_never_answers_times_out_and_kills_the_worker(client):
    with pytest.raises(WorkerError, match="no answer within 1"):
        client.score(items("HANG"), timeout=1)
    assert not client.is_alive()


def test_park_is_acknowledged(client):
    client.park()
    assert client.is_alive()


def test_the_score_timeout_grows_with_the_number_of_items():
    assert score_timeout(40) > score_timeout(1) >= 120
