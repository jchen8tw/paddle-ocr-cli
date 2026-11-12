"""What a Run would write to the JSON's `settings` and `engine` model names, known from the options alone: no worker, no
models. The "already done" check of a folder Run compares an existing JSON with exactly this."""
import importlib.util
from pathlib import Path

import pytest

import pdf_ocr
from block_reader import SCHEDULE
from fakes import FakeClient
from reader_bridge import MiniCPMBridge, PaddleBridge, reader_supports

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
def test_for_every_round_a_live_bridge_answers_like_the_rule_of_the_reader_that_reader_names(reader, judge, tmp_path):
    rule = reader_supports(args("--reader", reader).reader)
    bridge = bridge_for(reader, judge, tmp_path)
    assert [bridge.supports(step) for step in SCHEDULE] == [rule(step) for step in SCHEDULE]


@pytest.mark.parametrize("reader, supported", [("minicpm", [True, True, True, True]), ("paddle", [True, True, False, False])])
def test_each_reader_says_which_rounds_it_can_do_without_a_bridge(reader, supported):
    schedule = pdf_ocr.run_fingerprint(args("--reader", reader, "--judge")).settings["reread_schedule"]
    assert [r["supported"] for r in schedule] == supported


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
