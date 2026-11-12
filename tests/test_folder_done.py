from pathlib import Path

import pytest

import document_record as record
import pdf_ocr

ENGINE = {"paddleocr": "3.7.0", "paddlex": "3.7.2", "device": "cpu"}


def options(folder, out, *argv):
    return pdf_ocr.parse_args([str(folder), "-o", str(out), *argv])


def written(tmp_path, names=("a.pdf", "sub/b.pdf"), *argv):
    folder, out = tmp_path / "docs", tmp_path / "out"
    for name in names:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(f"%PDF-1.7 {name}".encode())
    run_options = options(folder, out, *argv)
    run = pdf_ocr.run_fingerprint(pdf_ocr.EngineSpec.from_options(run_options), run_options)
    for name in names:
        pdf, rel = folder / name, Path(name)
        (out / rel.parent / rel.stem).mkdir(parents=True)
        record.write(pdf, out / rel.parent / rel.stem, run, ENGINE, record.EngineTally(), record.Transcription(
            sha256=record.sha256(pdf), page_count=1, pages_processed=[1], pages_failed=[],
            tables_with_line_breaks_fixed=0, latex_symbols_unwrapped=0, script_repair={"mode": "auto", "applied": False},
            restructuring_failed=None, csv_tables=None, elapsed_seconds=1.0, pages=[{"page": 1, "blocks": []}]))
    return folder, out


def plan(folder, out, *argv):
    run_options = options(folder, out, *argv)
    planned = pdf_ocr.plan_run(folder, out, pdf_ocr.EngineSpec.from_options(run_options), run_options)
    return {d.relative.as_posix(): d for d in planned}


def test_documents_whose_json_matches_the_pdf_and_the_options_are_done(tmp_path):
    folder, out = written(tmp_path)
    assert [(d.done, d.reason) for d in plan(folder, out).values()] == [(True, ""), (True, "")]


def test_a_document_that_is_not_done_carries_the_reason_and_the_others_still_are_done(tmp_path):
    folder, out = written(tmp_path)
    (out / "a" / "a.json").unlink()
    plan_ = plan(folder, out)
    assert (plan_["a.pdf"].done, plan_["a.pdf"].reason) == (False, "no output yet")
    assert plan_["sub/b.pdf"].done


def test_a_run_with_another_option_that_shapes_the_output_finds_nothing_done(tmp_path):
    folder, out = written(tmp_path)
    plan_ = plan(folder, out, "--no-page-markers")
    assert [(d.done, d.reason) for d in plan_.values()] == [(False, "settings differ: page_markers")] * 2


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
    (folder / "a.PDF").write_bytes(b"%PDF-1.7 A")
    with pytest.raises(pdf_ocr.PlanError) as e:
        plan(folder, out)
    assert "a.pdf" in str(e.value) and "a.PDF" in str(e.value)


def test_a_done_document_inside_another_documents_csv_folder_is_still_a_collision(tmp_path):
    folder, out = written(tmp_path, ("a.pdf", "a/csv/b.pdf"))
    with pytest.raises(pdf_ocr.PlanError, match="csv"):
        plan(folder, out)


def test_planning_the_done_check_writes_nothing(tmp_path):
    folder, out = written(tmp_path)
    (out / "a" / "a.json").write_text("{", encoding="utf-8")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    plan(folder, out)
    plan(folder, out, "--force")
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
