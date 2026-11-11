"""The main-process side of the worker protocol: start-up handshake, requests, crashes and timeouts.

Runs against tests/fake_worker.py (the real request loop around a fake model), so no torch is needed.
"""
import sys
from pathlib import Path

import pytest

from worker_client import WorkerClient, WorkerError

FAKE = str(Path(__file__).parent / "fake_worker.py")


def make(mode="ready", **kw) -> WorkerClient:
    kw.setdefault("start_timeout", 20)
    return WorkerClient([sys.executable, FAKE, "--mode", mode], **kw)


@pytest.fixture
def client():
    c = make()
    yield c
    c.close()


def test_start_returns_the_ready_event_and_a_read_returns_the_readings(client):
    ready = client.start()
    assert ready["reader"] == "fake" and ready["versions"] == {"torch": "0.0"}
    readings = client.read("/tmp/x/b1.png", "Text Recognition:", 512)
    assert readings == [{"text": "b1.png|Text Recognition:|512", "tokens": 1, "finish": "stop"}]
    assert client.read("/tmp/x/b2.png", "Table Recognition:", 8192)[0]["text"] == "b2.png|Table Recognition:|8192"


def test_an_error_event_at_start_up_carries_the_fix_in_the_message():
    c = make("env-error")
    with pytest.raises(WorkerError, match="uv sync --project worker"):
        c.start()
    assert not c.is_alive()


def test_a_worker_that_dies_while_starting_reports_its_exit_code_and_last_stderr():
    c = make("die")
    with pytest.raises(WorkerError, match=r"(?s)exited with code 1.*cannot start"):
        c.start()


def test_a_worker_that_never_becomes_ready_is_killed_after_the_start_timeout():
    c = make("silent", start_timeout=1)
    with pytest.raises(WorkerError, match="did not become ready within 1"):
        c.start()
    assert not c.is_alive()


def test_a_missing_executable_is_a_clear_error():
    c = WorkerClient(["definitely-not-an-executable-xyz"], start_timeout=5)
    with pytest.raises(WorkerError, match="definitely-not-an-executable-xyz"):
        c.start()


def test_a_failed_read_is_an_error_but_the_worker_stays_usable(client):
    client.start()
    with pytest.raises(WorkerError, match="out of memory"):
        client.read("a.png", "FAIL", 1)
    assert client.is_alive()
    assert client.read("a.png", "ok", 1)


def test_a_crash_during_a_read_raises_with_the_stderr_tail_and_a_restart_recovers(client):
    client.start()
    with pytest.raises(WorkerError, match=r"(?s)exited with code 3.*CUDA error"):
        client.read("a.png", "CRASH", 1)
    assert not client.is_alive()
    with pytest.raises(WorkerError, match="not running"):
        client.read("a.png", "ok", 1)  # no silent restart inside a request
    client.ensure_started()
    assert client.read("a.png", "ok", 1)[0]["text"].startswith("a.png")


def test_a_read_that_never_answers_times_out_instead_of_hanging(client):
    client.start()
    with pytest.raises(WorkerError, match="no answer within 1"):
        client.read("a.png", "HANG", 1, timeout=1)
    assert not client.is_alive()  # the stream cannot be resynchronised, so the worker is killed


def test_a_worker_that_stops_reading_its_input_cannot_block_the_writer_forever():
    import time

    c = make("deaf")
    try:
        c.start()
        t0 = time.monotonic()
        with pytest.raises(WorkerError, match="could not be sent within 1"):
            c.read("a.png", "x" * 4_000_000, 1, timeout=1)  # far more than a pipe holds
        assert time.monotonic() - t0 < 10
        assert not c.is_alive()
    finally:
        c.close()


def test_restarts_are_capped():
    c = make(max_restarts=1)
    try:
        c.start()
        with pytest.raises(WorkerError):
            c.read("a.png", "CRASH", 1)
        c.ensure_started()
        with pytest.raises(WorkerError):
            c.read("a.png", "CRASH", 1)
        with pytest.raises(WorkerError, match="restart"):
            c.ensure_started()
    finally:
        c.close()


def test_close_stops_the_worker_and_is_safe_to_repeat(client):
    client.start()
    client.close()
    assert not client.is_alive()
    client.close()


def test_the_read_timeout_grows_with_the_token_limit():
    from worker_client import read_timeout

    assert read_timeout(8192) > read_timeout(256) >= 120
