"""The real Judge in the real worker, against the MiniCPM-OCR authors' own reference examples (slow; GPU).

worker/render/examples/ holds three Blocks with the Scores the authors observed (`observed_prob_yes`). If the
renderer, its fonts, Chromium or the model loading are wrong, the Scores drift away from them: the authors
expect about 5e-3, and a missing font or a randomly initialised score head is far off. This needs no sample
PDF, only the worker environment, the MiniCPM-OCR-Jev folder, Chromium, node and Noto Sans CJK.
"""
import json
import os
import sys
from pathlib import Path

import pytest

import pdf_ocr
from worker_client import WorkerClient

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "worker" / "render" / "examples"
TOLERANCE = 0.015  # measured here: at most 0.009 (a Score of 0.032 against the authors' 0.041)


@pytest.fixture(scope="module")
def client():
    models = Path(os.environ.get("PDF_OCR_MODELS_DIR", ROOT / "models"))
    if not (models / "MiniCPM-OCR-Jev").is_dir():
        pytest.skip(f"model folder missing: {models / 'MiniCPM-OCR-Jev'}")
    if problem := pdf_ocr.worker_env_problem():
        pytest.skip(problem)
    if problems := pdf_ocr.judge_setup_problems():
        pytest.skip("; ".join(problems))
    args = pdf_ocr.build_parser().parse_args(["x.pdf", "--reader", "paddle", "--models-dir", str(models)])
    c = WorkerClient(pdf_ocr.worker_command(args, "uv"), env=pdf_ocr.worker_environment())
    try:
        c.start()
    except Exception as e:
        pytest.skip(f"the worker could not start (set PLAYWRIGHT_BROWSERS_PATH?): {e}")
    yield c
    c.close()


def cases():
    out = []
    for d in sorted(EXAMPLES.iterdir()):
        case = json.loads((d / "case.json").read_text(encoding="utf-8"))
        for cand in case["candidates"]:
            out.append((case, d, cand))
    return out


def test_the_scores_match_the_authors_observed_scores(client):
    items = [{"key": f"{c['case_id']}/{cand['role']}", "image": str(d / c["block_image"]),
              "text": cand["text"], "category": c["category"]} for c, d, cand in cases()]
    results = client.score(items)
    assert len(results) == len(items)
    for (c, _, cand), r in zip(cases(), results):
        assert r["verdict"] == "scored", r
        observed = c["observed_prob_yes"][cand["role"]]
        assert abs(r["score"] - observed) < TOLERANCE, f"{r['key']}: {r['score']:.4f} vs observed {observed:.4f}"


def test_unparsable_latex_is_rejected_by_the_precheck_and_the_rest_of_the_batch_is_scored(client):
    block = str(EXAMPLES / "02_formula_putnam" / "block.png")
    results = client.score([
        {"key": "bad", "image": block, "text": r"\frac{1", "category": "formula"},
        {"key": "good", "image": block, "text": r"x \in S", "category": "formula"},
        {"key": "gone", "image": str(EXAMPLES / "nope.png"), "text": "x", "category": "text"},
    ])
    verdicts = {r["key"]: r["verdict"] for r in results}
    assert verdicts == {"bad": "precheck_reject", "good": "scored", "gone": "missing_original_image"}
    assert all(r["score"] is None for r in results if r["verdict"] != "scored")
