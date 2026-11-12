"""What a Run would write to the JSON's `settings` and `engine` model names, known from the options alone: no worker, no
models. The "already done" check of a folder Run compares an existing JSON with exactly this."""
import importlib.util
import json
from pathlib import Path

import pytest

import pdf_ocr
from fakes import ENGINE_ARGV, FakeClient, blank_pdf, engine_with
from reader_bridge import MiniCPMBridge, PaddleBridge

ROOT = Path(__file__).resolve().parent.parent
READERS = ["minicpm", "paddle"]


def args(*argv):
    return pdf_ocr.parse_args(["in.pdf", *argv])


def bridge_for(reader, judge, tmp_path):
    """The live bridge a Run with these options would hold (the worker and PaddleOCR-VL faked)."""
    common = dict(to_html=lambda s: "", judge=judge, judge_threshold=0.98)
    if reader == "paddle":
        return PaddleBridge(object(), FakeClient(), tmp_path, **common)  # supports() never calls the model
    return MiniCPMBridge(FakeClient(), tmp_path, **common)


@pytest.mark.parametrize("judge", [True, False])
@pytest.mark.parametrize("reader", READERS)
def test_the_settings_from_the_options_alone_equal_those_of_a_live_bridge(reader, judge, tmp_path):
    a = args("--reader", reader, *(["--judge"] if judge else []))
    assert pdf_ocr.settings_info(a) == pdf_ocr.settings_info(a, bridge_for(reader, judge, tmp_path))


@pytest.mark.parametrize("reader, supported", [("minicpm", [True, True, True, True]), ("paddle", [True, True, False, False])])
def test_each_reader_says_which_rounds_it_can_do_without_a_bridge(reader, supported):
    schedule = pdf_ocr.settings_info(args("--reader", reader, "--judge"))["reread_schedule"]
    assert [r["supported"] for r in schedule] == supported


def test_the_options_the_fake_engine_stands_for_are_the_ones_it_runs_with(tmp_path):
    a = args(*ENGINE_ARGV)
    engine, _ = engine_with(tmp_path)
    assert isinstance(engine.bridge, MiniCPMBridge) and a.reader == "minicpm"
    assert (engine.bridge.judge, engine.bridge.threshold) == (a.judge, a.judge_threshold) == (True, 0.98)


def test_the_settings_a_run_writes_are_the_settings_from_the_options_alone(tmp_path):
    pdf = blank_pdf(tmp_path / "in.pdf")
    a = pdf_ocr.parse_args([str(pdf), *ENGINE_ARGV, "--script", "off", "--no-page-markers"])
    engine, _ = engine_with(tmp_path / "engine")
    pdf_ocr.transcribe_document(engine, a, pdf_ocr.open_document(pdf, None), tmp_path / "out")
    written = json.loads((tmp_path / "out" / "in.json").read_text(encoding="utf-8"))
    assert written["settings"] == pdf_ocr.settings_info(a)


MODELS = [
    (["--reader", "minicpm"], {"layout_model": "PP-DocLayoutV3", "reader": "MiniCPM-OCR-1.0"}),
    (["--reader", "minicpm", "--judge"],
     {"layout_model": "PP-DocLayoutV3", "reader": "MiniCPM-OCR-1.0", "judge": "MiniCPM-OCR-Jev"}),
    ([], {"layout_model": "PP-DocLayoutV3", "reader": "PaddleOCR-VL-1.6"}),
    (["--judge"], {"layout_model": "PP-DocLayoutV3", "reader": "PaddleOCR-VL-1.6", "judge": "MiniCPM-OCR-Jev"}),
]


@pytest.mark.parametrize("argv, models", MODELS)
def test_the_engines_model_names_come_from_the_options_alone(argv, models):
    assert pdf_ocr.engine_models(args(*argv)) == models


@pytest.mark.parametrize("argv, models", MODELS)
def test_the_engine_section_a_run_writes_holds_those_model_names_whatever_the_worker_reports(argv, models):
    ready = {"judge": "whatever-the-worker-says", "versions": {"torch": "2.15.0"}}
    engine = pdf_ocr.engine_info(args(*argv), "gpu:0", ready)
    assert {k: v for k, v in engine.items() if k in ("layout_model", "reader", "judge")} == models


def test_the_workers_model_folder_names_are_the_ones_the_check_assumes():
    """The worker names its models in its `ready` event; the check names them without a worker."""
    spec = importlib.util.spec_from_file_location("worker_serve", ROOT / "worker" / "serve.py")
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    models = pdf_ocr.engine_models(args("--reader", "minicpm", "--judge"))
    assert (serve.READER_DIR, serve.JUDGE_DIR) == (models["reader"], models["judge"])
