"""The Engine spec answers every Reader x Judge rule from the options alone: no worker, no models."""
from typing import NamedTuple

import pytest

import pdf_ocr
from fakes import SETUPS
from pdf_ocr import Adapter, EngineSpec

LAYOUT, VL, MINICPM, JEV = "PP-DocLayoutV3", "PaddleOCR-VL-1.6", "MiniCPM-OCR-1.0", "MiniCPM-OCR-Jev"


def spec(*argv):
    return EngineSpec.from_options(pdf_ocr.parse_args(["in.pdf", *argv]))


class Answers(NamedTuple):
    needs_worker: bool
    model_dirs: list[str]
    worker_flags: list[str]
    models: dict[str, str]
    ink_reread: bool
    adapter: Adapter
    memory_advice_says: str


ANSWERS = {
    "paddleocr-vl": Answers(
        needs_worker=False, model_dirs=[LAYOUT, VL], worker_flags=["--no-reader", "--no-judge"],
        models={"layout_model": LAYOUT, "reader": VL}, ink_reread=True, adapter=Adapter.PADDLEOCR_VL_ALONE,
        memory_advice_says="try --reader minicpm, which needs less GPU memory"),
    "paddleocr-vl+judge": Answers(
        needs_worker=True, model_dirs=[LAYOUT, VL, JEV], worker_flags=["--no-reader"],
        models={"layout_model": LAYOUT, "reader": VL, "judge": JEV}, ink_reread=False, adapter=Adapter.WORKER,
        memory_advice_says="PaddleOCR-VL with the Judge is the tightest combination"),
    "minicpm": Answers(
        needs_worker=True, model_dirs=[LAYOUT, MINICPM], worker_flags=["--no-judge"],
        models={"layout_model": LAYOUT, "reader": MINICPM}, ink_reread=True, adapter=Adapter.WORKER,
        memory_advice_says="MiniCPM-OCR-1.0 without the Judge is the smallest configuration"),
    "minicpm+judge": Answers(
        needs_worker=True, model_dirs=[LAYOUT, MINICPM, JEV], worker_flags=[],
        models={"layout_model": LAYOUT, "reader": MINICPM, "judge": JEV}, ink_reread=False, adapter=Adapter.WORKER,
        memory_advice_says="must fit side by side; close other GPU programs, or run without --judge"),
}


@pytest.mark.parametrize("setup", SETUPS, ids=lambda setup: setup.id)
def test_the_spec_answers_every_reader_and_judge_rule_of_the_setup(setup):
    expected, s = ANSWERS[setup.id], spec(*setup.options)
    assert s.needs_worker is expected.needs_worker
    assert s.model_dirs == expected.model_dirs
    assert s.worker_flags == expected.worker_flags
    assert s.models == expected.models
    assert s.ink_reread is expected.ink_reread
    assert s.adapter is expected.adapter
    assert expected.memory_advice_says in s.memory_advice


@pytest.mark.parametrize("setup", [s for s in SETUPS if not s.judge], ids=lambda setup: setup.id)
def test_no_recover_turns_the_ink_based_re_read_off(setup):
    assert not spec(*setup.options, "--no-recover").ink_reread


def test_the_spec_holds_what_the_engine_needs_to_load_and_read():
    assert spec("--reader", "minicpm", "--judge", "--judge-threshold", "0.7", "--dpi", "150", "--max-new-tokens", "99",
                "--layout-threshold", "0.3", "--charts", "--seals", "--ocr-images", "--no-merge-blocks", "--no-recover",
                "--script", "off") == EngineSpec(
        reader="minicpm", judge=True, judge_threshold=0.7, dpi=150.0, max_new_tokens=99, layout_threshold=0.3,
        charts=True, seals=True, ocr_images=True, merge_blocks=False, recover=False, script="off")
    assert spec() == EngineSpec(
        reader="paddle", judge=False, judge_threshold=None, dpi=216.0, max_new_tokens=8192, layout_threshold=0.2,
        charts=False, seals=False, ocr_images=False, merge_blocks=True, recover=True, script="auto")
