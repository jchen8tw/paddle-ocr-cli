"""A worker crash fails the affected page, is recorded, and does not hang or take the run down.

Runs the real Layout model on the sample's first pages and the real request loop with a fake model
(tests/fake_worker.py) as the worker: no MiniCPM weights, but the real main-process path.
"""
import json
import re
import sys
from pathlib import Path

import pytest

import pdf_ocr

pytestmark = pytest.mark.gpu
FAKE = str(Path(__file__).parent / "fake_worker.py")


def run_with_fake_worker(monkeypatch, sample_pdf, layout_models_dir, out, extra_env=()):
    for k, v in extra_env:
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(pdf_ocr, "worker_env_problem", lambda: None)
    monkeypatch.setattr(pdf_ocr, "worker_command", lambda args, uv: [sys.executable, FAKE])
    code = pdf_ocr.main([str(sample_pdf), "-o", str(out), "--models-dir", str(layout_models_dir),
                         "--pages", "1-2", "--no-csv", "--reader", "minicpm", "--judge"])
    doc = json.loads((out / f"{sample_pdf.stem}.json").read_text(encoding="utf-8"))
    return code, doc


def test_a_run_through_the_worker_records_the_reader_in_the_json(monkeypatch, sample_pdf, layout_models_dir, tmp_path):
    code, doc = run_with_fake_worker(monkeypatch, sample_pdf, layout_models_dir, tmp_path)
    assert code == 0 and doc["source"]["pages_failed"] == []
    assert doc["settings"]["reader"] == "minicpm" and doc["engine"]["reader"] == "MiniCPM-OCR-1.0"
    assert doc["engine"]["torch"] == "0.0"
    assert doc["postprocessing"]["repetition_cuts"] == 0
    contents = [b["content"] for p in doc["pages"] for b in p["blocks"] if b["label"] not in ("image", "header_image")]
    # the stand-in worker echoes "<crop>|<prompt>|<token limit>"; the prompt depends on the label (Text, Table, Formula)
    assert contents and all(re.search(r"\|(Text|Table|Formula) Recognition:\|", c) for c in contents)


def test_a_worker_crash_fails_that_page_only_and_the_next_page_gets_a_restarted_worker(
        monkeypatch, sample_pdf, layout_models_dir, tmp_path):
    marker = tmp_path / "crashed"
    code, doc = run_with_fake_worker(monkeypatch, sample_pdf, layout_models_dir, tmp_path / "out",
                                     [("FAKE_CRASH_ONCE", str(marker))])
    assert marker.exists()
    assert code == 1
    assert doc["source"]["pages_failed"] == [1]
    failed, ok = doc["pages"]
    assert failed["blocks"] == [] and "exited with code 3" in failed["error"]
    assert ok["blocks"] and "error" not in ok
