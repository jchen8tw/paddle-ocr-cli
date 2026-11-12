"""End-to-end: run the CLI on a few pages of a real-text-layer PDF (slow; GPU + models + PDF needed).

Skipped cleanly when PDF_OCR_SAMPLE_PDF is not set (see conftest.py). The structure, error-rate and determinism
tests run once per measured configuration of tests/baseline.json (no options, and each Reader with --judge), against
the same expectations: every configuration must produce the same structure. The Judge tests run for the
configurations with --judge, and the Judge-off tests once per Reader.
"""
import json
import warnings
from pathlib import Path

import pytest

import ocr_eval
import pdf_ocr
from conftest import BASELINE
from pdf_ocr import DEFAULT_JUDGE_THRESHOLD

pytestmark = pytest.mark.gpu


def load(out: Path, pdf: Path) -> tuple[str, dict]:
    md = (out / f"{pdf.stem}.md").read_text(encoding="utf-8")
    doc = json.loads((out / f"{pdf.stem}.json").read_text(encoding="utf-8"))
    return md, doc


def test_output_structure(config_runs, config, sample_pdf):
    out = config_runs[0]
    md, doc = load(out, sample_pdf)
    options = config.options

    assert md.strip()
    assert {"source", "engine", "settings", "postprocessing", "csv_tables", "elapsed_seconds", "pages"} == doc.keys()
    assert config.record["settings"].items() <= doc["settings"].items()
    assert doc["engine"]["layout_model"] == "PP-DocLayoutV3"
    assert doc["engine"]["reader"] == pdf_ocr.READER_MODEL_DIRS[options.reader]
    assert ("judge" in doc["engine"]) == options.judge
    assert ("torch" in doc["engine"]) == pdf_ocr.needs_worker(options)
    assert {"tables_with_line_breaks_fixed", "latex_symbols_unwrapped", "script_repair"} <= doc["postprocessing"].keys()
    assert doc["source"]["file"] == sample_pdf.name
    assert doc["source"]["pages_failed"] == []
    assert len(doc["pages"]) == len(doc["source"]["pages_processed"]) > 0
    for page in doc["pages"]:
        assert page["blocks"], f"page {page['page']} has no blocks"
        for b in page["blocks"]:
            assert {"id", "order", "label", "content", "bbox", "polygon", "in_markdown"} <= b.keys()

    has_table = any(b["label"] == "table" for p in doc["pages"] for b in p["blocks"])
    if has_table:
        assert doc["csv_tables"] is not None
        files = list((out / "csv").glob("p*_table*.csv"))
        assert len(files) == len(doc["csv_tables"]["exported"])
        for entry in doc["csv_tables"]["exported"]:
            assert (out / entry["file"]).is_file()
    else:
        assert not (out / "csv").exists()


def test_error_rate_against_the_text_layer_stays_within_the_recorded_bound(
        config_runs, config, sample_pdf, cer_bound):
    score = ocr_eval.evaluate(sample_pdf, config_runs[0] / f"{sample_pdf.stem}.json")
    print("\n" + ocr_eval.format_report(score))
    assert score.total.ref_chars > 0, "the sample PDF has no text layer on these pages"
    if BASELINE.get("bound_is_placeholder"):
        warnings.warn("tests/baseline.json holds a PLACEHOLDER error-rate bound, not a measured baseline",
                      stacklevel=1)
    assert score.total.cer <= cer_bound, (
        f"CER {score.total.cer:.2%} ({score.total.errors} errors) exceeds the recorded bound {cer_bound:.2%}; "
        f"{config.name} was measured with {config.record['measured']['errors']} errors")


def test_second_run_gives_identical_markdown_and_json(config_runs, sample_pdf):
    (md1, doc1), (md2, doc2) = (load(out, sample_pdf) for out in config_runs)
    for doc in (doc1, doc2):
        doc.pop("elapsed_seconds")  # the only timing field
    assert md1 == md2
    assert doc1 == doc2


JUDGE_KEYS = {"score", "verdict", "flagged"}
UNJUDGED_LABELS = {"chart", "seal", "image", "header_image", "footer_image"}


def test_with_the_judge_on_every_judged_block_has_a_score_a_verdict_and_a_flag(judged_runs, sample_pdf):
    _, doc = load(judged_runs[0], sample_pdf)
    judged = 0
    for page in doc["pages"]:
        for b in page["blocks"]:
            if b["label"] in UNJUDGED_LABELS:
                assert not JUDGE_KEYS & b.keys(), f"{b['label']} must not carry a Score"
                continue
            assert JUDGE_KEYS <= b.keys(), f"page {page['page']} {b['label']} was not judged"
            assert b["verdict"] in {"scored", "precheck_reject", "render_error", "missing_original_image",
                                    "encode_error", "forward_error"}
            assert (b["score"] is not None) == (b["verdict"] == "scored")
            assert b["score"] is None or 0.0 <= b["score"] <= 1.0
            assert b["flagged"] == ("readings" in b or b["score"] is None or b["score"] < DEFAULT_JUDGE_THRESHOLD)
            judged += 1
    assert judged > 0
    assert doc["postprocessing"]["judge"]["blocks_judged"] == judged


def test_stderr_reports_the_flagged_blocks_per_page_and_the_markdown_carries_no_scores(
        judged_runs, sample_pdf):
    stderr = (judged_runs[0] / "stderr.txt").read_text(encoding="utf-8")
    md, doc = load(judged_runs[0], sample_pdf)
    page_lines = [ln for ln in stderr.splitlines() if ln.startswith("[") and "page" in ln]
    assert len(page_lines) == len(doc["pages"])
    assert all("judged flagged" in ln for ln in page_lines)
    for page in doc["pages"]:
        for b in page["blocks"]:
            if b.get("score") is not None:
                assert str(b["score"]) not in md, "a Score leaked into the Markdown"
    for key in ('"flagged"', "judged_before_script_decision", '"verdict"'):
        assert key not in md


def test_a_reader_without_judge_gives_no_scores_and_records_that_the_judge_was_off(judge_off_run, sample_pdf, reader):
    md, doc = load(judge_off_run, sample_pdf)
    assert doc["settings"]["reader"] == reader
    assert doc["settings"]["judge"] is False and doc["settings"]["judge_threshold"] is None
    assert doc["settings"]["reread_short_blocks"] is True  # the ink-based re-read runs instead
    assert "judge" not in doc["engine"] and "judge" not in doc["postprocessing"]
    assert "latex_symbols_unwrapped" in doc["postprocessing"]  # for --reader paddle, counted without a bridge
    assert doc["source"]["pages_failed"] == []
    for page in doc["pages"]:
        for b in page["blocks"]:
            assert not JUDGE_KEYS & b.keys()
    stderr = (judge_off_run / "stderr.txt").read_text(encoding="utf-8")
    assert "judged flagged" not in stderr


def test_the_markdown_is_the_same_whether_or_not_the_judge_ran_unless_the_ink_reread_changed_a_block(
        judge_without_rereads_run, judge_off_run, sample_pdf):
    """The Judge alone changes nothing about the text. (With a threshold that triggers re-reads the Markdown can
    legitimately differ, which is why this run uses --judge-threshold 0.)"""
    md_judged, doc_judged = load(judge_without_rereads_run, sample_pdf)
    md_plain, doc_plain = load(judge_off_run, sample_pdf)
    assert not any("readings" in b for p in doc_judged["pages"] for b in p["blocks"])
    if any("content_before_reread" in b for p in doc_plain["pages"] for b in p["blocks"]):
        pytest.skip("the ink-based re-read changed a block, so the two Markdowns legitimately differ")
    assert md_judged == md_plain
