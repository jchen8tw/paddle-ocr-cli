"""tests/baseline.json still describes the runs the end-to-end suite makes (no GPU): a change of a default that
would leave a measured configuration pointing at a run nobody measured fails here, before anyone re-measures."""
import math

import pytest

import pdf_ocr
from conftest import BASELINE

CONFIGURATIONS = BASELINE["configurations"]


def parse(argv):
    return pdf_ocr.parse_args(["x.pdf", *argv])


def test_the_default_configuration_is_a_run_with_no_options():
    assert CONFIGURATIONS["default"]["argv"] == []


def test_both_readers_are_measured_with_the_judge_on():
    judged = [parse(c["argv"]) for c in CONFIGURATIONS.values() if parse(c["argv"]).judge]
    assert {a.reader for a in judged} == set(pdf_ocr.READER_MODEL_DIRS)


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_a_measured_configuration_still_runs_with_the_settings_it_was_measured_with(name):
    entry = CONFIGURATIONS[name]
    options = parse(entry["argv"])
    settings = pdf_ocr.run_fingerprint(pdf_ocr.EngineSpec.from_options(options), options).settings
    assert {k: settings[k] for k in entry["settings"]} == entry["settings"]


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_each_bound_is_the_measured_rate_plus_0_05_points_rounded_up_to_the_next_0_05_points(name):
    entry, measured = CONFIGURATIONS[name], CONFIGURATIONS[name]["measured"]
    assert measured["ref_chars"] == BASELINE["ref_chars"]
    assert measured["cer"] == round(measured["errors"] / measured["ref_chars"], 6)
    steps = math.ceil(round((measured["cer"] + 0.0005) / 0.0005, 6))
    assert entry["cer_bound"] == pytest.approx(steps * 0.0005)
