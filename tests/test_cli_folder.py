"""A folder Run through main(argv), without models: what is refused before anything loads, and the loop over the
Documents with a faked Engine (the pages, the post-processing and the files are the real ones)."""
import json
import os
import time

import pytest

import pdf_ocr
from fakes import blank_pdf, engine_with


def run_refused(argv, capsys):
    with pytest.raises(SystemExit) as e:
        pdf_ocr.main(argv)
    return e.value.code, capsys.readouterr().err


def folder_of(tmp_path, *names, pages=1):
    folder = tmp_path / "docs"
    for name in names:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        blank_pdf(folder / name, pages)
    return folder


NO_MODELS = ["--models-dir", "does-not-exist"]  # reaching the setup checks would end in "model folder(s) not found"


def test_pages_with_a_folder_is_a_usage_error(tmp_path, capsys):
    folder = folder_of(tmp_path, "a.pdf")
    code, err = run_refused([str(folder), "-o", str(tmp_path / "out"), "--pages", "1", *NO_MODELS], capsys)
    assert code == 2 and "--pages" in err and "folder" in err
    assert not (tmp_path / "out").exists()


def test_a_folder_without_pdfs_is_an_error_before_any_setup_check(tmp_path, capsys):
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    (folder / "notes.txt").write_text("not a PDF")
    code, err = run_refused([str(folder), "-o", str(tmp_path / "out"), *NO_MODELS], capsys)
    assert code.startswith("error: no PDF found under") and err == ""
    assert not (tmp_path / "out").exists()


def test_two_documents_with_one_output_folder_are_an_error_before_any_setup_check(tmp_path, capsys):
    folder = folder_of(tmp_path, "a.pdf", "a.PDF", "other.pdf")
    code, err = run_refused([str(folder), "-o", str(tmp_path / "out"), *NO_MODELS], capsys)
    assert code.startswith("error: ") and "a.pdf" in code and "a.PDF" in code and "other.pdf" not in code
    assert err == "" and not (tmp_path / "out").exists()


def test_a_documents_output_folder_inside_anothers_csv_is_an_error_before_any_setup_check(tmp_path, capsys):
    folder = folder_of(tmp_path, "a.pdf", "a/csv/b.pdf")
    code, err = run_refused([str(folder), "-o", str(tmp_path / "out"), *NO_MODELS], capsys)
    assert code.startswith("error: ") and "a.pdf" in code and "a/csv/b.pdf" in code
    assert err == "" and not (tmp_path / "out").exists()


def test_the_setup_checks_follow_the_planning_and_nothing_is_written_when_they_fail(tmp_path, capsys):
    folder = folder_of(tmp_path, "a.pdf", "sub/b.pdf")
    code, err = run_refused([str(folder), "-o", str(tmp_path / "out"), *NO_MODELS], capsys)
    assert code.startswith("error: model folder(s) not found")
    assert err == f"found 2 Documents in {folder}\n"
    assert not (tmp_path / "out").exists()


class Loaded:
    """Replaces the setup checks and the model load of a Run: counts them and hands out the faked Engine."""

    def __init__(self, monkeypatch, tmp_path):
        self.engine, self.client = engine_with(tmp_path / "engine")
        self.checks = self.loads = 0
        monkeypatch.setattr(pdf_ocr, "check_setup", self.check_setup)
        monkeypatch.setattr(pdf_ocr, "load_engine", self.load_engine)

    def check_setup(self, args):
        self.checks += 1

    def load_engine(self, args, stack):
        self.loads += 1
        self.engine.load_started = time.time() - 10_000  # the model load, long ago: a Document must not count it
        return self.engine


@pytest.fixture
def loaded(monkeypatch, tmp_path):
    return Loaded(monkeypatch, tmp_path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_folder_run_writes_each_document_to_its_mirrored_folder_and_exits_0(tmp_path, capsys, loaded):
    folder = folder_of(tmp_path, "b.pdf", "sub/deeper/a.PDF")
    out = tmp_path / "out"
    assert pdf_ocr.main([str(folder), "-o", str(out), "--judge-threshold", "0.98"]) == 0
    assert (out / "b" / "b.md").is_file() and (out / "sub" / "deeper" / "a" / "a.md").is_file()
    doc = read_json(out / "sub" / "deeper" / "a" / "a.json")
    assert doc["source"]["file"] == "a.PDF"  # the bare file name, as in a single-PDF Run
    assert doc["elapsed_seconds"] < 100  # the Document's own time, not the model load
    err = capsys.readouterr().err
    assert err.splitlines()[0] == f"found 2 Documents in {folder}"
    assert err.splitlines()[-1] == "2 Documents: 2 written, 0 failed"
    assert loaded.checks == 1 and loaded.loads == 1


def test_the_default_output_root_is_output_slash_the_folders_name(tmp_path, monkeypatch, loaded):
    folder = folder_of(tmp_path, "a.pdf")
    monkeypatch.chdir(tmp_path)
    assert pdf_ocr.main([str(folder)]) == 0
    assert (tmp_path / "output" / "docs" / "a" / "a.json").is_file()


def test_the_stderr_of_a_folder_run_has_a_header_per_document_and_a_summary_with_the_failures(
        tmp_path, capsys, loaded):
    folder = folder_of(tmp_path, "a.pdf", "sub/b.pdf", pages=2)
    (folder / "bad.pdf").write_bytes(b"not a pdf")
    (folder / "gone.pdf").symlink_to(tmp_path / "nowhere.pdf")
    (folder / ".hidden.pdf").write_bytes(b"not a pdf either")
    out = tmp_path / "out"
    assert pdf_ocr.main([str(folder), "-o", str(out), "--judge-threshold", "0.98"]) == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == f"found 4 Documents in {folder}"
    assert lines[1] == "[1/4] a.pdf (2 pages)"
    assert lines[2].startswith("[  1/2] page 1:") and lines[3].startswith("[  2/2] page 2:")
    assert lines[4].startswith(f"wrote {out / 'a' / 'a.md'} and {out / 'a' / 'a.json'}")
    assert lines[5].startswith("[2/4] bad.pdf FAILED: cannot open ")  # reported where it happens, nothing else for it
    assert lines[6].startswith("[3/4] gone.pdf FAILED: cannot open ")
    assert lines[7] == "[4/4] sub/b.pdf (2 pages)"
    assert lines[-3] == "4 Documents: 2 written, 2 failed"
    assert lines[-2].startswith("  bad.pdf: cannot open ") and lines[-1].startswith("  gone.pdf: cannot open ")
    assert sorted(p.name for p in out.iterdir()) == ["a", "sub"]  # no output folder, no JSON, for a failed Document
    assert (out / "sub" / "b" / "b.json").is_file()


def test_a_failed_page_is_a_failed_document_run_exit_1_but_its_output_is_written(tmp_path, capsys, loaded):
    folder = folder_of(tmp_path, "a.pdf", "b.pdf")
    loaded.client.fail_on = "Text Recognition:"  # the worker dies on every read: every page fails
    out = tmp_path / "out"
    assert pdf_ocr.main([str(folder), "-o", str(out), "--judge-threshold", "0.98"]) == 1
    assert read_json(out / "a" / "a.json")["source"]["pages_failed"] == [1]
    assert read_json(out / "b" / "b.json")["source"]["pages_failed"] == [1]
    lines = capsys.readouterr().err.splitlines()
    assert lines[-3] == "2 Documents: 2 written, 0 failed"
    assert lines[-2:] == ["  a.pdf: page 1 failed (recorded in its JSON)", "  b.pdf: page 1 failed (recorded in its JSON)"]


def test_a_document_that_raises_is_reported_and_the_run_continues(tmp_path, capsys, loaded, monkeypatch):
    folder = folder_of(tmp_path, "a.pdf", "boom.pdf", "c.pdf")
    real = pdf_ocr.transcribe_document

    def flaky(engine, args, document, out_dir, started=None):
        if document.path.name == "boom.pdf":
            raise RuntimeError("the disk is full")
        return real(engine, args, document, out_dir, started)

    monkeypatch.setattr(pdf_ocr, "transcribe_document", flaky)
    out = tmp_path / "out"
    assert pdf_ocr.main([str(folder), "-o", str(out), "--judge-threshold", "0.98"]) == 1
    assert (out / "a" / "a.json").is_file() and (out / "c" / "c.json").is_file() and not (out / "boom").exists()
    err = capsys.readouterr().err
    assert "3 Documents: 2 written, 1 failed\n  boom.pdf: RuntimeError: the disk is full\n" in err


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_a_pdf_that_cannot_be_read_is_reported_and_the_run_continues(tmp_path, capsys, loaded):
    folder = folder_of(tmp_path, "a.pdf", "locked.pdf", "z.pdf")
    (folder / "locked.pdf").chmod(0o000)
    assert pdf_ocr.main([str(folder), "-o", str(tmp_path / "out"), "--judge-threshold", "0.98"]) == 1
    assert "  locked.pdf: cannot open " in capsys.readouterr().err
    assert (tmp_path / "out" / "z" / "z.json").is_file() and not (tmp_path / "out" / "locked").exists()
