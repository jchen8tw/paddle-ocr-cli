"""The CLI side of --reader: defaults, which model folders each option needs, the worker command line and
what the JSON records. No GPU, no models (a model folder is just an empty directory here)."""
import sys
from pathlib import Path

import pytest

import pdf_ocr


def args(*argv):
    return pdf_ocr.parse_args(["in.pdf", *argv])


def spec(*argv):
    return pdf_ocr.EngineSpec.from_options(args(*argv))


def test_paddle_is_the_default_reader_and_minicpm_can_be_chosen():
    assert args().reader == "paddle"
    assert args("--reader", "minicpm").reader == "minicpm"
    with pytest.raises(SystemExit):
        args("--reader", "tesseract")


def test_check_models_names_every_missing_folder(tmp_path):
    msg = pdf_ocr.missing_models_message(spec(), tmp_path)
    named = msg.split(". ")[0]  # the list of missing folders, before the hint about the other Reader
    assert "PP-DocLayoutV3" in named and "PaddleOCR-VL-1.6" in named
    assert "MiniCPM-OCR-1.0" not in named and "MiniCPM-OCR-Jev" not in named


def test_a_run_with_no_options_needs_only_the_layout_and_paddleocr_vl_folders(tmp_path):
    for name in ("PP-DocLayoutV3", "PaddleOCR-VL-1.6"):
        (tmp_path / name).mkdir()
    assert pdf_ocr.missing_models_message(spec(), tmp_path) is None
    assert "MiniCPM-OCR-1.0" in pdf_ocr.missing_models_message(spec("--reader", "minicpm"), tmp_path)


def test_worker_command_runs_serve_py_in_the_worker_project_without_syncing(tmp_path):
    cmd = pdf_ocr.worker_command(args("--reader", "minicpm", "--judge", "--models-dir", str(tmp_path)), uv="/usr/bin/uv")
    assert cmd[:2] == ["/usr/bin/uv", "run"]
    assert "--no-sync" in cmd and cmd[cmd.index("--project") + 1] == str(pdf_ocr.WORKER_DIR)
    assert cmd[-4:] == ["python", str(pdf_ocr.WORKER_DIR / "serve.py"), "--models-dir", str(tmp_path)]


def test_a_missing_worker_environment_is_reported_before_uv_can_create_an_empty_one(tmp_path, monkeypatch):
    monkeypatch.delenv("PDF_OCR_WORKER_ENV", raising=False)
    monkeypatch.setattr(pdf_ocr, "WORKER_DIR", tmp_path)  # no .venv in it
    msg = pdf_ocr.worker_env_problem()
    assert "uv sync --project worker" in msg and str(tmp_path / ".venv") in msg
    assert not (tmp_path / ".venv").exists()
    (tmp_path / ".venv").mkdir()
    assert pdf_ocr.worker_env_problem() is None


def test_the_worker_env_override_replaces_the_default_location(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_OCR_WORKER_ENV", str(tmp_path / "elsewhere"))
    assert "elsewhere" in pdf_ocr.worker_env_problem()
    env = pdf_ocr.worker_environment()
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "elsewhere")


def test_the_main_projects_uv_environment_does_not_leak_into_the_worker(monkeypatch):
    monkeypatch.delenv("PDF_OCR_WORKER_ENV", raising=False)
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/some/main/.venv")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/main/.venv")  # what an outer `uv run` exports
    env = pdf_ocr.worker_environment()
    assert "UV_PROJECT_ENVIRONMENT" not in env and "VIRTUAL_ENV" not in env
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"


def test_engine_names_the_layout_model_and_the_reader_with_torch_and_transformers_for_minicpm():
    ready = {"reader": "MiniCPM-OCR-1.0", "versions": {"torch": "2.15.0+cu132", "transformers": "5.22.2"}}
    e = pdf_ocr.engine_info(args("--reader", "minicpm"), "gpu:0", ready)
    assert e["layout_model"] == "PP-DocLayoutV3" and e["reader"] == "MiniCPM-OCR-1.0"
    assert e["torch"] == "2.15.0+cu132" and e["transformers"] == "5.22.2" and e["device"] == "gpu:0"
    assert "vl_model" not in e


def test_engine_for_the_paddle_reader_keeps_todays_keys_and_adds_the_reader():
    e = pdf_ocr.engine_info(args(), "gpu:0", None)
    assert e["pipeline"] == "PaddleOCR-VL-1.6" and e["vl_model"] == "PaddleOCR-VL-1.6"
    assert e["reader"] == "PaddleOCR-VL-1.6" and e["layout_model"] == "PP-DocLayoutV3"
    assert "torch" not in e
