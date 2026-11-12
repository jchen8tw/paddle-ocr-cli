"""Planning a folder Run: which Documents a folder holds, in which order, and where each one's output goes.

Pure file-system work on temp trees: no model, no GPU, no paddle.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pdf_ocr

ROOT = Path(__file__).resolve().parent.parent


def tree(base: Path, *names: str) -> Path:
    """Empty files at these paths under `base` (parents created)."""
    for name in names:
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return base


def plan(folder, output_root=None, *argv):
    args = pdf_ocr.parse_args([str(folder), *argv])
    return pdf_ocr.plan_run(Path(folder), output_root, pdf_ocr.EngineSpec.from_options(args), args)


def found(documents) -> list[str]:
    return [d.relative.as_posix() for d in documents]


def test_every_pdf_in_the_tree_is_a_document_and_other_files_are_not(tmp_path):
    folder = tree(tmp_path / "docs", "a.pdf", "sub/b.pdf", "sub/deeper/c.pdf", "notes.txt", "sub/d.pdf.bak", "pdf",
                  "sub/e.pdfx")
    assert found(plan(folder, tmp_path / "out")) == ["a.pdf", "sub/b.pdf", "sub/deeper/c.pdf"]


def test_the_extension_is_matched_ignoring_case(tmp_path):
    folder = tree(tmp_path, "SCAN.PDF", "report.Pdf", "plain.pdf")
    assert found(plan(folder, tmp_path / "out")) == ["SCAN.PDF", "plain.pdf", "report.Pdf"]


def test_files_and_folders_starting_with_a_dot_are_skipped_at_any_depth(tmp_path):
    folder = tree(tmp_path / "docs", "keep.pdf", ".hidden.pdf", "._mac.pdf", ".git/objects/x.pdf", "sub/.Trash/y.pdf",
                  "sub/._z.pdf", "sub/.cache.PDF", "sub/ok.pdf")
    assert found(plan(folder, tmp_path / "out")) == ["keep.pdf", "sub/ok.pdf"]


def test_the_folder_itself_may_be_hidden_or_lie_under_a_hidden_folder(tmp_path):
    folder = tree(tmp_path / ".hidden" / "docs", "a.pdf")
    assert found(plan(folder, tmp_path / "out")) == ["a.pdf"]


def test_a_symlinked_file_is_a_document_and_a_symlinked_folder_is_not_entered(tmp_path):
    outside = tree(tmp_path / "outside", "real.pdf", "far/other.pdf")
    folder = tree(tmp_path / "docs", "own.pdf")
    (folder / "link.pdf").symlink_to(outside / "real.pdf")
    (folder / "linked-folder").symlink_to(outside, target_is_directory=True)
    (folder / "folder.pdf").symlink_to(outside, target_is_directory=True)  # a folder, whatever its name says
    documents = plan(folder, tmp_path / "out")
    assert found(documents) == ["link.pdf", "own.pdf"]
    assert documents[0].path == folder / "link.pdf"  # the link itself, not where it points: it names its output


def test_a_link_back_to_a_parent_folder_does_not_make_the_search_endless(tmp_path):
    folder = tree(tmp_path / "docs", "a.pdf", "sub/b.pdf")
    (folder / "sub" / "loop").symlink_to("..", target_is_directory=True)
    (folder / "self").symlink_to(".", target_is_directory=True)
    assert found(plan(folder, tmp_path / "out")) == ["a.pdf", "sub/b.pdf"]


def test_a_symlink_to_nothing_is_still_a_document_so_that_opening_it_can_report_it(tmp_path):
    folder = tree(tmp_path / "docs", "a.pdf")
    (folder / "gone.pdf").symlink_to(tmp_path / "nowhere.pdf")
    assert found(plan(folder, tmp_path / "out")) == ["a.pdf", "gone.pdf"]


def test_the_same_bytes_at_two_paths_are_two_documents(tmp_path):
    folder = tree(tmp_path / "docs", "a.pdf")
    (folder / "copy.pdf").symlink_to(folder / "a.pdf")
    (folder / "sub").mkdir()
    (folder / "sub" / "a.pdf").hardlink_to(folder / "a.pdf")
    documents = plan(folder, tmp_path / "out")
    assert found(documents) == ["a.pdf", "copy.pdf", "sub/a.pdf"]
    assert len({d.out_dir for d in documents}) == 3


def test_documents_come_folder_by_folder_in_sorted_order_whatever_order_they_were_made_in(tmp_path):
    folder = tree(tmp_path / "docs", "b.pdf", "z/a.pdf", "a.pdf", "a-b.pdf", "a/z.pdf", "a/B.pdf", "a/a.pdf", "A.pdf")
    # path by path, like Path: the folder "a" comes before the file "a-b.pdf", which comes before "a.pdf"
    assert found(plan(folder, tmp_path / "out")) == [
        "A.pdf", "a/B.pdf", "a/a.pdf", "a/z.pdf", "a-b.pdf", "a.pdf", "b.pdf", "z/a.pdf"]


def test_a_document_writes_into_its_own_folder_below_the_output_root_mirroring_its_place(tmp_path):
    folder = tree(tmp_path / "docs", "top.pdf", "2024/q1/report.PDF", "v1.2.final.pdf")
    out = tmp_path / "out"
    assert {d.relative.as_posix(): d.out_dir for d in plan(folder, out)} == {
        "top.pdf": out / "top",
        "2024/q1/report.PDF": out / "2024" / "q1" / "report",
        "v1.2.final.pdf": out / "v1.2.final",
    }


def test_the_output_root_defaults_to_output_slash_the_name_of_the_resolved_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tree(tmp_path / "scans", "a.pdf")
    (tmp_path / "other").mkdir()
    assert [d.out_dir for d in plan(Path("scans"))] == [Path("output/scans/a")]
    assert [d.out_dir for d in plan(Path("other/../scans/"))] == [Path("output/scans/a")]  # the name after resolving
    (tmp_path / "alias").symlink_to("scans", target_is_directory=True)
    assert [d.out_dir for d in plan(Path("alias"))] == [Path("output/scans/a")]


def test_dot_as_the_folder_is_the_current_directory_and_names_the_default_output_root(tmp_path, monkeypatch):
    folder = tree(tmp_path / "archive", "a.pdf", "sub/b.pdf")
    monkeypatch.chdir(folder)
    documents = plan(Path("."))
    assert found(documents) == ["a.pdf", "sub/b.pdf"]
    assert [d.out_dir for d in documents] == [Path("output/archive/a"), Path("output/archive/sub/b")]
    assert [d.path for d in documents] == [Path("a.pdf"), Path("sub/b.pdf")]


def test_planning_writes_nothing(tmp_path):
    folder = tree(tmp_path / "docs", "a.pdf", "sub/b.pdf")
    before = sorted(p for p in tmp_path.rglob("*"))
    plan(folder, tmp_path / "out")
    assert sorted(p for p in tmp_path.rglob("*")) == before and not (tmp_path / "out").exists()


def test_two_documents_with_one_output_folder_are_refused_and_named(tmp_path):
    folder = tree(tmp_path / "docs", "a.pdf", "a.PDF", "fine.pdf")
    with pytest.raises(pdf_ocr.PlanError) as e:
        plan(folder, tmp_path / "out")
    message = str(e.value)
    assert "a.pdf" in message and "a.PDF" in message and "fine.pdf" not in message
    assert str(tmp_path / "out" / "a") in message


@pytest.mark.parametrize("inside", ["csv", "imgs"])
def test_a_documents_output_folder_inside_another_documents_csv_or_imgs_is_refused_and_both_are_named(tmp_path, inside):
    folder = tree(tmp_path / "docs", "a.pdf", f"a/{inside}/b.pdf", f"a/{inside}/deeper/c.pdf", "fine.pdf")
    with pytest.raises(pdf_ocr.PlanError) as e:
        plan(folder, tmp_path / "out")
    message = str(e.value)
    assert "a.pdf" in message and f"a/{inside}/b.pdf" in message and f"a/{inside}/deeper/c.pdf" in message
    assert "fine.pdf" not in message


def test_an_output_folder_that_is_the_csv_folder_itself_is_a_collision(tmp_path):
    folder = tree(tmp_path / "docs", "a.pdf", "a/csv.pdf")  # csv.pdf writes to <root>/a/csv/
    with pytest.raises(pdf_ocr.PlanError) as e:
        plan(folder, tmp_path / "out")
    assert "a.pdf" in str(e.value) and "a/csv.pdf" in str(e.value)


def test_a_document_inside_another_documents_folder_but_not_its_csv_or_imgs_is_fine(tmp_path):
    folder = tree(tmp_path / "docs", "a.pdf", "a/b.pdf", "a/csv2/c.pdf", "csv/d.pdf")
    assert found(plan(folder, tmp_path / "out")) == ["a/b.pdf", "a/csv2/c.pdf", "a.pdf", "csv/d.pdf"]


@pytest.mark.parametrize("names", [[], ["notes.txt", ".hidden.pdf", ".git/a.pdf", "sub/readme.md"]])
def test_a_folder_without_documents_is_refused(tmp_path, names):
    folder = tree(tmp_path / "docs", *names)
    folder.mkdir(exist_ok=True)
    with pytest.raises(pdf_ocr.PlanError, match="no PDF"):
        plan(folder, tmp_path / "out")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any folder")
def test_a_folder_that_cannot_be_read_is_refused_instead_of_being_skipped_silently(tmp_path):
    folder = tree(tmp_path / "docs", "a.pdf", "locked/b.pdf")
    (folder / "locked").chmod(0o000)
    try:
        with pytest.raises(pdf_ocr.PlanError, match="locked"):
            plan(folder, tmp_path / "out")
    finally:
        (folder / "locked").chmod(0o755)


def test_planning_needs_no_paddle(tmp_path):
    folder = tree(tmp_path / "docs", "a.pdf", "sub/b.pdf")
    code = ("import sys, pdf_ocr; from pathlib import Path\n"
            "args = pdf_ocr.parse_args(sys.argv[1:2])\n"
            "assert len(pdf_ocr.plan_run(Path(sys.argv[1]), None, pdf_ocr.EngineSpec.from_options(args), args)) == 2\n"
            "assert 'paddle' not in sys.modules and 'torch' not in sys.modules")
    done = subprocess.run([sys.executable, "-c", code, str(folder)], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert done.returncode == 0, done.stderr
