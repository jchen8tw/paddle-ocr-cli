"""End-to-end re-reading (slow; GPU + models + PDF needed): a threshold of 0.9999 forces Blocks through the
re-read schedule, with either Reader (see conftest.py: PDF_OCR_REREAD_PAGES)."""
import json

import pytest

pytestmark = pytest.mark.gpu

STOP_REASONS = {"passed", "exhausted", "no_scored", "judge_error", "reader_error"}


def load(out, pdf):
    return json.loads((out / f"{pdf.stem}.json").read_text(encoding="utf-8"))


def reread_blocks(doc):
    return [b for p in doc["pages"] for b in p["blocks"] if "readings" in b]


def test_two_runs_with_re_reads_give_identical_json_and_markdown(reread_runs, sample_pdf):
    docs = [load(out, sample_pdf) for out in reread_runs]
    assert reread_blocks(docs[0]), "the threshold should have forced re-reads"
    for doc in docs:
        doc.pop("elapsed_seconds")
    assert docs[0] == docs[1]
    mds = [(out / f"{sample_pdf.stem}.md").read_text(encoding="utf-8") for out in reread_runs]
    assert mds[0] == mds[1]


def test_a_re_read_block_has_readings_a_stop_reason_and_the_winners_content_and_score(reread_runs, sample_pdf, reader):
    doc = load(reread_runs[0], sample_pdf)
    for b in reread_blocks(doc):
        assert b["stop_reason"] in STOP_REASONS
        readings = b["readings"]
        assert 1 <= len(readings) and sum(1 + r.get("duplicates", 0) for r in readings) <= 10
        assert len({r["content"] for r in readings}) == len(readings)  # each text listed (and scored) once
        for r in readings:
            assert {"content", "score", "verdict", "round", "decode", "temperature", "scale"} <= r.keys()
            assert (r["score"] is not None) == (r["verdict"] == "scored")
        scored = [r["score"] for r in readings if r["score"] is not None]
        if scored:
            assert b["score"] == max(scored)  # the highest Score wins, not the latest
            assert b["stop_reason"] != "no_scored"
        assert b["flagged"] is True  # a re-read never clears the flag
        if reader == "paddle":  # the local PaddleOCR-VL cannot sample: rounds 3 and 4 are recorded as skipped
            assert [s["round"] for s in b["skipped_rounds"]] == [3, 4]
            assert {r["round"] for r in readings} <= {1, 2}


def test_blocks_that_were_not_re_read_have_no_readings_or_stop_reason(reread_runs, sample_pdf):
    doc = load(reread_runs[0], sample_pdf)
    for p in doc["pages"]:
        for b in p["blocks"]:
            if "readings" not in b:
                assert not {"stop_reason", "skipped_rounds"} & b.keys()


def test_the_postprocessing_and_stderr_report_re_read_blocks_and_flagged_blocks(
        reread_runs, sample_pdf):
    doc = load(reread_runs[0], sample_pdf)
    judge = doc["postprocessing"]["judge"]
    judged = [b for p in doc["pages"] for b in p["blocks"] if "verdict" in b]
    assert judge["blocks_reread"] == len(reread_blocks(doc)) > 0
    assert judge["blocks_flagged"] == sum(1 for b in judged if b["flagged"])
    assert sum(judge["reread_stop_reasons"].values()) == judge["blocks_reread"]
    stderr = (reread_runs[0] / "stderr.txt").read_text(encoding="utf-8")
    page_lines = [ln for ln in stderr.splitlines() if ln.startswith("[") and "page" in ln]
    assert len(page_lines) == len(doc["pages"])
    assert all("judged flagged" in ln and " re-read" in ln for ln in page_lines)
    assert sum(int(ln.split("flagged, ")[1].split(" re-read")[0]) for ln in page_lines) == judge["blocks_reread"]
