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
    assert ("torch" in doc["engine"] and "transformers" in doc["engine"]) == (reader == "minicpm")
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
