import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import document_record as record

ROOT = Path(__file__).resolve().parent.parent

OPTIONS = dict(reader="minicpm", judge=True, judge_threshold=0.98, dpi=216.0, layout_threshold=0.2, max_new_tokens=8192,
               charts=False, seals=False, ocr_images=False, no_merge_blocks=False, no_recover=False,
               no_merge_tables=False, md_ignore_labels=["header", "footer"], script="auto", no_page_markers=False,
               no_csv=False)
WITHOUT_JUDGE = dict(judge=False, judge_threshold=None)
MODELS = {"layout_model": "PP-DocLayoutV3", "reader": "MiniCPM-OCR-1.0", "judge": "MiniCPM-OCR-Jev"}
MODELS_WITHOUT_JUDGE = {"layout_model": "PP-DocLayoutV3", "reader": "MiniCPM-OCR-1.0"}
ENGINE = {"layout_model": "PP-DocLayoutV3", "reader": "MiniCPM-OCR-1.0", "paddleocr": "3.7.0", "paddlex": "3.7.2",
          "judge": "MiniCPM-OCR-Jev", "torch": "2.15.0", "device": "gpu:0"}
REPAIR = {"mode": "auto", "applied": False}
REPAIR_SKIPPED_BY_AN_OLD_RUN = {"mode": "auto", "applied": False, "skipped": "opencc not installed"}


def every_round(step):
    return True


def greedy_rounds_only(step):
    return step.decode == "greedy"


def fingerprint(models=MODELS, supports=every_round, **options):
    return record.fingerprint(argparse.Namespace(**{**OPTIONS, **options}), models, supports)


def transcription(pdf, pages=1, **results):
    return record.Transcription(**{
        "sha256": record.sha256(pdf), "page_count": pages, "pages_processed": list(range(1, pages + 1)),
        "pages_failed": [], "tables_with_line_breaks_fixed": 0, "latex_symbols_unwrapped": 0, "script_repair": REPAIR,
        "restructuring_failed": None, "csv_tables": None, "elapsed_seconds": 12.34,
        "pages": [{"page": n, "blocks": []} for n in range(1, pages + 1)], **results})


def written(tmp_path, name="a", run=None, engine=ENGINE, tally=record.EngineTally(), **results):
    pdf, folder = tmp_path / "docs" / f"{name}.pdf", tmp_path / "out" / name
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(f"%PDF-1.7 {name}".encode())
    folder.mkdir(parents=True)
    record.write(pdf, folder, run or fingerprint(), engine, tally, transcription(pdf, **results))
    return pdf, folder


def read(folder, name="a"):
    return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))


def edit(folder, change, name="a"):
    doc = read(folder, name)
    change(doc)
    (folder / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_the_json_is_named_after_the_pdf_and_its_sections_come_in_this_order(tmp_path):
    pdf, folder = written(tmp_path)
    doc = read(folder)
    assert list(doc) == ["source", "engine", "settings", "postprocessing", "csv_tables", "elapsed_seconds", "pages"]
    assert doc["source"] == {"file": "a.pdf", "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "page_count": 1,
                             "pages_processed": [1], "pages_failed": []}
    assert (doc["csv_tables"], doc["elapsed_seconds"], doc["pages"]) == (None, 12.3, [{"page": 1, "blocks": []}])


def test_the_postprocessing_of_a_run_without_the_worker_has_only_the_documents_own_steps(tmp_path):
    _, folder = written(tmp_path, tables_with_line_breaks_fixed=2, latex_symbols_unwrapped=4)
    assert read(folder)["postprocessing"] == {"tables_with_line_breaks_fixed": 2, "latex_symbols_unwrapped": 4,
                                              "script_repair": REPAIR}


def test_the_engine_adds_its_repetition_cuts_unwrapped_symbols_and_judge_summary_in_their_places(tmp_path):
    tally = record.EngineTally(repetition_cuts=5, latex_symbols_unwrapped=3, judge={"blocks_judged": 2})
    _, folder = written(tmp_path, tally=tally, latex_symbols_unwrapped=4, restructuring_failed="RuntimeError: x")
    post = read(folder)["postprocessing"]
    assert list(post) == ["tables_with_line_breaks_fixed", "repetition_cuts", "latex_symbols_unwrapped",
                          "script_repair", "judge", "restructuring_failed"]
    assert (post["repetition_cuts"], post["latex_symbols_unwrapped"], post["judge"]) == (5, 7, {"blocks_judged": 2})
    assert post["restructuring_failed"] == "RuntimeError: x"


def test_the_engine_section_keeps_its_order_and_names_the_models_of_the_fingerprint(tmp_path):
    _, folder = written(tmp_path, engine={**ENGINE, "reader": "whatever-the-worker-says"})
    engine = read(folder)["engine"]
    assert engine == ENGINE and list(engine) == list(ENGINE)


def test_the_engine_section_of_a_run_without_the_judge_names_no_judge(tmp_path):
    _, folder = written(tmp_path, run=fingerprint(MODELS_WITHOUT_JUDGE, **WITHOUT_JUDGE))
    assert "judge" not in read(folder)["engine"]


def test_the_settings_say_which_rounds_the_reader_can_do_and_only_with_the_judge():
    schedule = fingerprint(supports=greedy_rounds_only).settings["reread_schedule"]
    assert [(r["round"], r["decode"], r["supported"]) for r in schedule] == [
        (1, "greedy", True), (2, "greedy", True), (3, "temperature", False), (4, "temperature", False)]
    assert "reread_schedule" not in fingerprint(MODELS_WITHOUT_JUDGE, **WITHOUT_JUDGE).settings


READERS_AND_JUDGES = [
    dict(models={**MODELS_WITHOUT_JUDGE, "reader": "PaddleOCR-VL-1.6"}, supports=greedy_rounds_only, reader="paddle",
         **WITHOUT_JUDGE),
    dict(models={**MODELS, "reader": "PaddleOCR-VL-1.6"}, supports=greedy_rounds_only, reader="paddle"),
    dict(models=MODELS_WITHOUT_JUDGE, **WITHOUT_JUDGE),
    dict(models=MODELS),
]


@pytest.mark.parametrize("run", READERS_AND_JUDGES,
                         ids=["paddleocr-vl", "paddleocr-vl+judge", "minicpm", "minicpm+judge"])
def test_a_json_written_with_the_fingerprint_of_this_run_is_done(tmp_path, run):
    pdf, folder = written(tmp_path, run=fingerprint(**run))
    assert record.why_not_done(pdf, folder, fingerprint(**run)) is None


def test_a_document_without_a_json_is_not_done_and_the_others_still_are(tmp_path):
    a, a_out = written(tmp_path)
    b, b_out = written(tmp_path, "b")
    (a_out / "a.json").unlink()
    assert record.why_not_done(a, a_out, fingerprint()) == "no output yet"
    assert record.why_not_done(b, b_out, fingerprint()) is None


def test_a_pdf_that_changed_since_is_not_done(tmp_path):
    pdf, folder = written(tmp_path)
    pdf.write_bytes(b"%PDF-1.7 another")
    assert "changed" in record.why_not_done(pdf, folder, fingerprint())


def test_a_document_with_failed_pages_is_not_done(tmp_path):
    pdf, folder = written(tmp_path, pages=2, pages_failed=[2])
    assert "failed" in record.why_not_done(pdf, folder, fingerprint())


@pytest.mark.parametrize("processed, named", [([1], "1 of 3"), ([1, 3], "1, 3 of 3"), ([], "of 3")])
def test_a_json_that_covers_only_some_of_the_pages_is_not_done_and_its_reason_names_them(tmp_path, processed, named):
    pdf, folder = written(tmp_path, pages=3, pages_processed=processed)
    assert named in record.why_not_done(pdf, folder, fingerprint())


def test_a_json_that_records_a_skipped_script_repair_is_not_done(tmp_path):
    pdf, folder = written(tmp_path, script_repair=REPAIR_SKIPPED_BY_AN_OLD_RUN)
    assert "repair" in record.why_not_done(pdf, folder, fingerprint())


def test_a_json_that_records_a_failed_cross_page_restructuring_is_done(tmp_path):
    pdf, folder = written(tmp_path, restructuring_failed="RuntimeError: the merge broke")
    assert record.why_not_done(pdf, folder, fingerprint()) is None


@pytest.mark.parametrize("changed, key", [
    (dict(script="off"), "script"),
    (dict(no_page_markers=True), "page_markers"),
    (dict(judge_threshold=0.9), "judge_threshold"),
    (dict(reader="paddle"), "reader"),
    (dict(dpi=150.0), "dpi"),
    (dict(layout_threshold=0.3), "layout_threshold"),
    (dict(max_new_tokens=4096), "max_new_tokens"),
    (dict(charts=True), "chart_recognition"),
    (dict(seals=True), "seal_recognition"),
    (dict(ocr_images=True), "ocr_for_image_blocks"),
    (dict(no_merge_blocks=True), "merge_layout_blocks"),
    (dict(no_merge_tables=True), "markdown_merge_tables"),
    (dict(md_ignore_labels=["header"]), "markdown_ignore_labels"),
    (dict(no_csv=True), "csv_export"),
    (dict(supports=greedy_rounds_only), "reread_schedule"),
])
def test_a_document_written_with_another_setting_is_not_done(tmp_path, changed, key):
    pdf, folder = written(tmp_path)
    reason = record.why_not_done(pdf, folder, fingerprint(**changed))
    assert reason.startswith("settings differ") and key in reason


def test_a_document_written_with_the_judge_is_not_done_by_a_run_without_it(tmp_path):
    pdf, folder = written(tmp_path)
    reason = record.why_not_done(pdf, folder, fingerprint(MODELS_WITHOUT_JUDGE, **WITHOUT_JUDGE))
    assert reason.startswith("settings differ") and "judge" in reason


def test_options_that_do_not_shape_the_output_leave_the_document_done(tmp_path):
    pdf, folder = written(tmp_path)
    assert record.why_not_done(pdf, folder, fingerprint(verbose=True, device="cpu", models_dir="elsewhere")) is None


@pytest.mark.parametrize("key", ["layout_model", "reader", "judge"])
def test_a_document_read_by_other_models_is_not_done(tmp_path, key):
    pdf, folder = written(tmp_path)
    edit(folder, lambda doc: doc["engine"].update({key: "SomeOtherModel"}))
    assert record.why_not_done(pdf, folder, fingerprint()) == f"models differ: {key}"


def test_a_run_with_other_models_does_not_find_the_document_done(tmp_path):
    pdf, folder = written(tmp_path)
    another_reader = fingerprint({**MODELS, "reader": "MiniCPM-OCR-2.0"})
    assert record.why_not_done(pdf, folder, another_reader) == "models differ: reader"


def test_a_judge_that_was_not_recorded_while_this_run_uses_it_is_not_done(tmp_path):
    pdf, folder = written(tmp_path)
    edit(folder, lambda doc: doc["engine"].pop("judge"))
    assert record.why_not_done(pdf, folder, fingerprint()) == "models differ: judge"


def test_a_judge_that_was_recorded_while_this_run_has_none_is_not_done(tmp_path):
    run = fingerprint(MODELS_WITHOUT_JUDGE, **WITHOUT_JUDGE)
    pdf, folder = written(tmp_path, run=run)
    edit(folder, lambda doc: doc["engine"].update(judge="MiniCPM-OCR-Jev"))
    assert record.why_not_done(pdf, folder, run) == "models differ: judge"


def test_other_library_versions_and_another_device_leave_a_document_done(tmp_path):
    pdf, folder = written(tmp_path)

    def upgrade(doc):
        doc["engine"].update(paddleocr="9.9.9", paddlex="9.9.9", torch="9.9.9", transformers="9.9.9", device="gpu:3")
        doc["engine"]["new_library"] = "1.0"

    edit(folder, upgrade)
    assert record.why_not_done(pdf, folder, fingerprint()) is None


@pytest.mark.parametrize("content", [
    b"", b"{", b'{"source": {"sha256": "ab', b"not json at all", b"\xff\xfe\x00 binary", b"[]", b"null", b'"text"', b"42",
])
def test_a_json_that_cannot_be_read_is_not_done_and_never_an_error(tmp_path, content):
    pdf, folder = written(tmp_path)
    (folder / "a.json").write_bytes(content)
    assert record.why_not_done(pdf, folder, fingerprint())


def test_a_json_cut_off_in_the_middle_is_not_done(tmp_path):
    pdf, folder = written(tmp_path)
    text = (folder / "a.json").read_text(encoding="utf-8")
    (folder / "a.json").write_text(text[: len(text) // 2], encoding="utf-8")
    assert record.why_not_done(pdf, folder, fingerprint())


def test_a_folder_where_the_json_should_be_is_not_done_and_never_an_error(tmp_path):
    pdf, folder = written(tmp_path)
    (folder / "a.json").unlink()
    (folder / "a.json").mkdir()
    assert record.why_not_done(pdf, folder, fingerprint())


@pytest.mark.parametrize("remove", [
    lambda doc: doc.pop("source"),
    lambda doc: doc["source"].pop("sha256"),
    lambda doc: doc["source"].pop("pages_failed"),
    lambda doc: doc["source"].pop("page_count"),
    lambda doc: doc["source"].pop("pages_processed"),
    lambda doc: doc.pop("settings"),
    lambda doc: doc.pop("engine"),
    lambda doc: doc["settings"].pop("script"),
    lambda doc: doc["settings"].pop("reread_schedule"),
    lambda doc: doc["engine"].pop("layout_model"),
    lambda doc: doc["engine"].pop("reader"),
    lambda doc: doc.update(source=None),
    lambda doc: doc.update(settings=[]),
    lambda doc: doc.update(engine="MiniCPM"),
    lambda doc: doc["source"].update(pages_failed=None),
    lambda doc: doc["source"].update(sha256=None),
    lambda doc: doc["source"].update(page_count="1"),
    lambda doc: doc["source"].update(pages_processed=None),
    lambda doc: doc.pop("postprocessing"),
    lambda doc: doc["postprocessing"].pop("script_repair"),
    lambda doc: doc["postprocessing"].update(script_repair="auto"),
])
def test_a_json_missing_a_field_is_not_done_and_never_an_error(tmp_path, remove):
    pdf, folder = written(tmp_path)
    edit(folder, remove)
    assert record.why_not_done(pdf, folder, fingerprint())


def test_a_pdf_that_cannot_be_read_has_nothing_to_compare_with_and_is_not_done(tmp_path):
    pdf, folder = written(tmp_path)
    pdf.unlink()
    pdf.symlink_to(tmp_path / "nowhere.pdf")
    assert "cannot read the PDF" in record.why_not_done(pdf, folder, fingerprint())


def test_asking_whether_a_document_is_done_writes_nothing(tmp_path):
    pdf, folder = written(tmp_path)
    edit(folder, lambda doc: doc["source"].update(pages_failed=[1]))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert record.why_not_done(pdf, folder, fingerprint())
    assert record.why_not_done(pdf, folder, fingerprint(script="off"))
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_the_record_needs_nothing_from_paddle_the_bridge_or_the_engine():
    code = ("import sys, document_record\n"
            "loaded = {'paddle', 'paddlex', 'torch', 'reader_bridge', 'worker_client', 'pdf_ocr'} & set(sys.modules)\n"
            "assert not loaded, loaded")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert done.returncode == 0, done.stderr
