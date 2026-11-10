"""The Simplified to Traditional repair decided while reading, so the Judge can judge repaired text.

The document-level repair (pdf_ocr.repair_script) needs the whole document before it decides. The Judge
runs per Block, so the decision is made on the Readings seen so far and stays made: once the document
looks Traditional every later Reading is repaired before it is judged.
"""
from script_repair import ScriptRepair, StickyScriptRepair, is_traditional

TRAD = "國際學習體驗電腦關係開發設計價值義務織組條約證據隨時變數與認為對應條款歷計劃書據點"  # 39 characters that only Traditional uses
SIMP = "银行医院细胞"  # Simplified drift


def sticky(**kw):
    return StickyScriptRepair(ScriptRepair(), **kw)


def test_the_decision_rule_matches_the_document_level_repair():
    assert is_traditional(simp=0, trad=30)
    assert not is_traditional(simp=0, trad=29)      # too little evidence
    assert is_traditional(simp=15, trad=85)         # share 0.15 is still Traditional
    assert not is_traditional(simp=16, trad=84)     # a mixed or Simplified document is left alone


def test_readings_before_the_document_is_known_to_be_traditional_are_not_repaired():
    s = sticky()
    text, decided = s.apply(SIMP, "text")
    assert (text, decided) == (SIMP, False)
    assert not s.is_traditional


def test_once_the_document_is_decided_traditional_every_later_reading_is_repaired():
    s = sticky()
    s.apply(TRAD, "text")
    assert s.is_traditional
    text, decided = s.apply("银行", "text")
    assert text == "銀行" and decided


def test_the_reading_that_completes_the_evidence_is_itself_repaired():
    s = sticky()
    s.apply(TRAD[:20], "text")
    assert not s.is_traditional
    text, decided = s.apply(TRAD[20:] + "银行", "text")
    assert text.endswith("銀行") and decided


def test_the_decision_is_sticky_even_if_a_later_reading_is_mostly_simplified():
    s = sticky()
    s.apply(TRAD, "text")
    s.apply(SIMP * 100, "text")
    assert s.is_traditional
    assert s.apply("医院", "text") == ("醫院", True)


def test_formulas_and_algorithms_are_neither_surveyed_nor_repaired():
    s = sticky()
    text, _ = s.apply(TRAD * 3, "display_formula")
    assert not s.is_traditional                      # not counted as evidence
    s.apply(TRAD, "text")
    assert s.apply("银行", "inline_formula") == ("银行", True)  # decided, but this label is left alone


def test_the_same_readings_in_the_same_order_give_the_same_repairs():
    readings = [SIMP, TRAD, "银行医院", "细胞" + TRAD, "银行"]
    runs = []
    for _ in range(2):
        s = sticky()
        runs.append([s.apply(t, "text") for t in readings])
    assert runs[0] == runs[1]


def test_without_opencc_nothing_is_repaired_and_the_run_goes_on():
    s = StickyScriptRepair(None)
    assert s.apply(TRAD + SIMP, "text") == (TRAD + SIMP, False)
