"""Planning a folder Run, the "already done" half: which Documents an earlier Run has already written, judged from the
JSON that is on disk and the options of this Run. Pure file-system work, no model, no GPU.

The JSONs under test are written by the real post-processing (an Engine built by `fakes.engine_for` reads the pages
with the PaddleX pipe and the worker faked), then damaged.
"""
import json
from pathlib import Path

import pytest

import pdf_ocr
from fakes import SETUPS, blank_pdf, engine_for

JUDGE_ON = ["--reader", "minicpm", "--judge", "--judge-threshold", "0.98"]  # the Run these tests start from
JUDGE_OFF = ["--reader", "minicpm", "--no-judge"]  # the same Reader without the Judge: a threshold would be a usage error


def options(folder, out, *argv, base=JUDGE_ON):
    return pdf_ocr.parse_args([str(folder), "-o", str(out), *base, *argv])


def written(tmp_path, names=("a.pdf", "sub/b.pdf"), *argv, fail_pages=False, base=JUDGE_ON):
    """A folder of blank PDFs, and the output root that a Run of them with `base` and `argv` left behind."""
    folder, out = tmp_path / "docs", tmp_path / "out"
    for name in names:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        blank_pdf(folder / name)
    args = options(folder, out, *argv, base=base)
    engine, fakes = engine_for(args, tmp_path / "engine")
    if fail_pages:
        fakes.worker.fail_on = "Text Recognition:"
    for rel in pdf_ocr.find_pdfs(folder):
        errors = pdf_ocr.transcribe_document(engine, args, pdf_ocr.open_document(folder / rel, None),
                                             out / rel.parent / rel.stem)
        assert bool(errors) == fail_pages
    return folder, out


def plan(folder, out, *argv, base=JUDGE_ON):
    return {d.relative.as_posix(): d for d in pdf_ocr.plan_run(folder, out, options(folder, out, *argv, base=base))}


def json_of(out, name="a"):
    return out / name / f"{name}.json"


def edit(path, change):
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_documents_whose_json_matches_the_pdf_and_the_options_are_done(tmp_path):
    folder, out = written(tmp_path)
    plan_ = plan(folder, out)
    assert [(d.done, d.reason) for d in plan_.values()] == [(True, ""), (True, "")]


@pytest.mark.parametrize("setup", SETUPS, ids=lambda setup: setup.id)
def test_documents_written_in_any_setup_are_done_for_a_run_with_the_same_options(tmp_path, setup):
    folder, out = written(tmp_path, base=setup.options)
    plan_ = plan(folder, out, base=setup.options)
    assert [(d.done, d.reason) for d in plan_.values()] == [(True, ""), (True, "")]


def test_a_document_without_a_json_is_not_done_and_the_others_still_are(tmp_path):
    folder, out = written(tmp_path)
    json_of(out).unlink()
    plan_ = plan(folder, out)
    assert (plan_["a.pdf"].done, plan_["sub/b.pdf"].done) == (False, True)
    assert plan_["a.pdf"].reason


def test_a_pdf_that_changed_since_is_not_done(tmp_path):
    folder, out = written(tmp_path)
    blank_pdf(folder / "a.pdf", pages=2)
    plan_ = plan(folder, out)
    assert not plan_["a.pdf"].done and "changed" in plan_["a.pdf"].reason
    assert plan_["sub/b.pdf"].done


def test_a_document_with_failed_pages_is_not_done(tmp_path):
    folder, out = written(tmp_path, fail_pages=True)
    assert json.loads(json_of(out).read_text(encoding="utf-8"))["source"]["pages_failed"] == [1]  # a real failure
    plan_ = plan(folder, out)
    assert [d.done for d in plan_.values()] == [False, False]
    assert "failed" in plan_["a.pdf"].reason


@pytest.mark.parametrize("changed, key", [
    (["--script", "off"], "script"),
    (["--no-page-markers"], "page_markers"),
    (["--judge-threshold", "0.9"], "judge_threshold"),
    (["--reader", "paddle"], "reader"),
    (["--dpi", "150"], "dpi"),
    (["--layout-threshold", "0.3"], "layout_threshold"),
    (["--max-new-tokens", "4096"], "max_new_tokens"),
    (["--charts"], "chart_recognition"),
    (["--no-merge-tables"], "markdown_merge_tables"),
    (["--md-ignore-labels", "header"], "markdown_ignore_labels"),
    (["--no-csv"], "csv_export"),
])
def test_a_document_written_with_another_setting_is_not_done(tmp_path, changed, key):
    folder, out = written(tmp_path)
    plan_ = plan(folder, out, *changed)
    assert [d.done for d in plan_.values()] == [False, False]
    assert key in plan_["a.pdf"].reason and "settings" in plan_["a.pdf"].reason


def test_a_document_written_with_the_judge_is_not_done_by_a_run_without_it(tmp_path):
    folder, out = written(tmp_path)
    plan_ = plan(folder, out, base=JUDGE_OFF)
    assert [d.done for d in plan_.values()] == [False, False]
    assert "judge" in plan_["a.pdf"].reason and "settings" in plan_["a.pdf"].reason


def test_options_that_do_not_shape_the_output_leave_the_documents_done(tmp_path):
    folder, out = written(tmp_path)
    plan_ = plan(folder, out, "-v", "--device", "cpu", "--models-dir", "elsewhere")
    assert [d.done for d in plan_.values()] == [True, True]


@pytest.mark.parametrize("section, key", [("engine", "layout_model"), ("engine", "reader"), ("engine", "judge")])
def test_a_document_read_by_other_models_is_not_done(tmp_path, section, key):
    folder, out = written(tmp_path)
    edit(json_of(out), lambda doc: doc[section].update({key: "SomeOtherModel"}))
    plan_ = plan(folder, out)
    assert not plan_["a.pdf"].done and key in plan_["a.pdf"].reason and "model" in plan_["a.pdf"].reason
    assert plan_["sub/b.pdf"].done


def test_a_judge_that_was_not_recorded_while_this_run_uses_it_is_not_done(tmp_path):
    folder, out = written(tmp_path)
    edit(json_of(out), lambda doc: doc["engine"].pop("judge"))
    assert not plan(folder, out)["a.pdf"].done


def test_a_judge_that_was_recorded_while_this_run_has_none_is_not_done(tmp_path):
    folder, out = written(tmp_path, ("a.pdf",), base=JUDGE_OFF)
    edit(json_of(out), lambda doc: doc["engine"].update(judge="MiniCPM-OCR-Jev"))
    assert not plan(folder, out, base=JUDGE_OFF)["a.pdf"].done


def test_other_library_versions_and_another_device_leave_a_document_done(tmp_path):
    folder, out = written(tmp_path)

    def upgrade(doc):
        doc["engine"].update(paddleocr="9.9.9", paddlex="9.9.9", torch="9.9.9", transformers="9.9.9", device="gpu:3")
        doc["engine"]["new_library"] = "1.0"

    edit(json_of(out), upgrade)
    assert plan(folder, out)["a.pdf"].done


@pytest.mark.parametrize("content", [
    b"", b"{", b'{"source": {"sha256": "ab', b"not json at all", b"\xff\xfe\x00 binary", b"[]", b"null", b'"text"', b"42",
])
def test_a_json_that_cannot_be_read_is_not_done_and_never_an_error(tmp_path, content):
    folder, out = written(tmp_path)
    json_of(out).write_bytes(content)
    plan_ = plan(folder, out)
    assert not plan_["a.pdf"].done and plan_["a.pdf"].reason
    assert plan_["sub/b.pdf"].done


def test_a_json_cut_off_in_the_middle_is_not_done(tmp_path):
    folder, out = written(tmp_path)
    text = json_of(out).read_text(encoding="utf-8")
    json_of(out).write_text(text[: len(text) // 2], encoding="utf-8")
    assert not plan(folder, out)["a.pdf"].done


def test_a_folder_where_the_json_should_be_is_not_done_and_never_an_error(tmp_path):
    folder, out = written(tmp_path)
    json_of(out).unlink()
    json_of(out).mkdir()
    assert not plan(folder, out)["a.pdf"].done


@pytest.mark.parametrize("remove", [
    lambda doc: doc.pop("source"),
    lambda doc: doc["source"].pop("sha256"),
    lambda doc: doc["source"].pop("pages_failed"),
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
])
def test_a_json_missing_a_field_is_not_done_and_never_an_error(tmp_path, remove):
    folder, out = written(tmp_path)
    edit(json_of(out), remove)
    plan_ = plan(folder, out)
    assert not plan_["a.pdf"].done and plan_["a.pdf"].reason
    assert plan_["sub/b.pdf"].done


def test_a_pdf_that_cannot_be_read_has_nothing_to_compare_with_and_is_not_done(tmp_path):
    folder, out = written(tmp_path)
    (folder / "a.pdf").unlink()
    (folder / "a.pdf").symlink_to(tmp_path / "nowhere.pdf")
    plan_ = plan(folder, out)
    assert not plan_["a.pdf"].done and plan_["sub/b.pdf"].done


def test_force_leaves_nothing_done(tmp_path):
    folder, out = written(tmp_path)
    plan_ = plan(folder, out, "--force")
    assert [d.done for d in plan_.values()] == [False, False]
    assert all("--force" in d.reason for d in plan_.values())


def test_a_planned_document_still_names_its_path_and_output_folder(tmp_path):
    folder, out = written(tmp_path)
    d = plan(folder, out)["sub/b.pdf"]
    assert (d.path, d.relative, d.out_dir) == (folder / "sub" / "b.pdf", Path("sub/b.pdf"), out / "sub" / "b")


def test_collisions_are_found_among_the_documents_that_are_done_too(tmp_path):
    folder, out = written(tmp_path, ("a.pdf",))
    blank_pdf(folder / "a.PDF")  # the same output folder as a.pdf, which is done
    with pytest.raises(pdf_ocr.PlanError) as e:
        plan(folder, out)
    assert "a.pdf" in str(e.value) and "a.PDF" in str(e.value)


def test_a_done_document_inside_another_documents_csv_folder_is_still_a_collision(tmp_path):
    folder, out = written(tmp_path, ("a.pdf", "a/csv/b.pdf"))
    with pytest.raises(pdf_ocr.PlanError, match="csv"):
        plan(folder, out)


def test_planning_the_done_check_writes_nothing(tmp_path):
    folder, out = written(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    plan(folder, out)
    plan(folder, out, "--force")
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
