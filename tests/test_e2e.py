"""End-to-end: run the CLI on a few pages of a real-text-layer PDF (slow; GPU + models + PDF needed).

Skipped cleanly when PDF_OCR_SAMPLE_PDF is not set (see conftest.py). Every test runs once per Reader
(MiniCPM-OCR-1.0 through the torch worker, and PaddleOCR-VL-1.6), against the same expectations: the
two Readers must produce the same structure.
"""
import json
import warnings
from pathlib import Path

import pytest

import ocr_eval
from conftest import BASELINE

pytestmark = pytest.mark.gpu


def load(out: Path, pdf: Path) -> tuple[str, dict]:
    md = (out / f"{pdf.stem}.md").read_text(encoding="utf-8")
    doc = json.loads((out / f"{pdf.stem}.json").read_text(encoding="utf-8"))
    return md, doc


def test_output_structure(cli_runs, sample_pdf, pages, reader):
    out = cli_runs[0]
    md, doc = load(out, sample_pdf)

    assert md.strip()
    assert {"source", "engine", "settings", "postprocessing", "csv_tables", "elapsed_seconds", "pages"} == doc.keys()
    assert doc["settings"]["reader"] == reader
    assert doc["engine"]["layout_model"] == "PP-DocLayoutV3"
    assert doc["engine"]["reader"] == {"minicpm": "MiniCPM-OCR-1.0", "paddle": "PaddleOCR-VL-1.6"}[reader]
    assert doc["engine"]["judge"] == "MiniCPM-OCR-Jev"
    assert "torch" in doc["engine"] and "transformers" in doc["engine"]  # the worker runs the Judge for either Reader
    assert doc["settings"]["judge"] is True and doc["settings"]["judge_threshold"] == 0.98
    assert {"tables_with_line_breaks_fixed", "script_repair"} <= doc["postprocessing"].keys()
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
        cli_runs, sample_pdf, cer_bound):
    score = ocr_eval.evaluate(sample_pdf, cli_runs[0] / f"{sample_pdf.stem}.json")
    print("\n" + ocr_eval.format_report(score))
    assert score.total.ref_chars > 0, "the sample PDF has no text layer on these pages"
    if BASELINE.get("bound_is_placeholder"):
        warnings.warn("tests/baseline.json holds a PLACEHOLDER error-rate bound, not a measured baseline",
                      stacklevel=1)
    assert score.total.cer <= cer_bound, (
        f"CER {score.total.cer:.2%} exceeds the recorded bound {cer_bound:.2%}")


def test_second_run_gives_identical_markdown_and_json(cli_runs, sample_pdf):
    (md1, doc1), (md2, doc2) = (load(out, sample_pdf) for out in cli_runs)
    for doc in (doc1, doc2):
        doc.pop("elapsed_seconds")  # the only timing field
    assert md1 == md2
    assert doc1 == doc2


JUDGE_KEYS = {"score", "verdict", "below_threshold"}
UNJUDGED_LABELS = {"chart", "seal", "image", "header_image", "footer_image"}


def test_with_the_judge_on_every_judged_block_has_a_score_a_verdict_and_a_flag(cli_runs, sample_pdf):
    _, doc = load(cli_runs[0], sample_pdf)
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
            assert b["below_threshold"] == (b["score"] is None or b["score"] < 0.98)
            judged += 1
    assert judged > 0
    assert doc["postprocessing"]["judge"]["blocks_judged"] == judged


def test_stderr_reports_the_blocks_below_the_threshold_per_page_and_the_markdown_carries_no_scores(cli_runs, sample_pdf):
    stderr = (cli_runs[0] / "stderr.txt").read_text(encoding="utf-8")
    md, doc = load(cli_runs[0], sample_pdf)
    page_lines = [ln for ln in stderr.splitlines() if ln.startswith("[") and "page" in ln]
    assert len(page_lines) == len(doc["pages"])
    assert all("judged below the threshold" in ln for ln in page_lines)
    for page in doc["pages"]:
        for b in page["blocks"]:
            if b.get("score") is not None:
                assert str(b["score"]) not in md, "a Score leaked into the Markdown"
    for key in ("below_threshold", "judged_before_script_decision", '"verdict"'):
        assert key not in md


def test_no_judge_gives_no_scores_and_records_that_the_judge_was_off(cli_run_no_judge, sample_pdf, reader):
    md, doc = load(cli_run_no_judge, sample_pdf)
    assert doc["settings"]["judge"] is False and doc["settings"]["judge_threshold"] is None
    assert doc["settings"]["reread_short_blocks"] is True  # the ink-based re-read runs instead
    assert "judge" not in doc["engine"] and "judge" not in doc["postprocessing"]
    assert doc["source"]["pages_failed"] == []
    for page in doc["pages"]:
        for b in page["blocks"]:
            assert not JUDGE_KEYS & b.keys()
    stderr = (cli_run_no_judge / "stderr.txt").read_text(encoding="utf-8")
    assert "below the threshold" not in stderr


def test_the_markdown_is_the_same_whether_or_not_the_judge_ran_unless_the_ink_reread_changed_a_block(
        cli_run_judge_without_rereads, cli_run_no_judge, sample_pdf):
    """The Judge alone changes nothing about the text. (With a threshold that triggers re-reads the Markdown can
    legitimately differ, which is why this run uses --judge-threshold 0.)"""
    md_judged, doc_judged = load(cli_run_judge_without_rereads, sample_pdf)
    md_plain, doc_plain = load(cli_run_no_judge, sample_pdf)
    assert not any("readings" in b for p in doc_judged["pages"] for b in p["blocks"])
    if any("content_before_reread" in b for p in doc_plain["pages"] for b in p["blocks"]):
        pytest.skip("the ink-based re-read changed a block, so the two Markdowns legitimately differ")
    assert md_judged == md_plain
