"""The PaddleX post-processing a Reading gets, done before judging so that the Score describes the written text."""
import minicpm_reader as mr


def clean(raw, label="text", **kw):
    return mr.clean_reading(raw, label, to_html=lambda s: "", **kw)[0]


def test_inline_math_delimiters_are_rewritten_the_way_the_pipeline_does_it_later():
    assert clean(r"(a) \(0 \in S\);") == "(a)  $ 0 \\in S $;"


def test_display_math_delimiters_are_rewritten():
    assert clean(r"\[ E = mc^2 \]") == " $$  E = mc^2  $$ "


def test_dollar_signs_in_a_reading_with_latex_delimiters_are_dropped_as_the_pipeline_does():
    assert clean(r"costs $5 and \(x\)") == "costs 5 and  $ x $"


def test_a_reading_without_latex_delimiters_keeps_its_dollar_signs():
    assert clean("price $5 and $10") == "price $5 and $10"


def test_the_rewrite_is_idempotent_so_the_pipelines_own_pass_finds_nothing_to_do():
    once = mr.rewrite_latex_delimiters(r"if \(x \in S\) then \[y\]", "text")
    assert mr.rewrite_latex_delimiters(once, "text") == once


def test_a_formula_number_loses_its_dollar_signs_altogether():
    assert clean(r"\(3.1\)", "formula_number") == "  3.1 "


def test_strip_can_be_turned_off_for_a_reader_whose_text_the_pipeline_never_stripped():
    assert clean("  padded  ", strip=False) == "  padded  "
    assert clean("  padded  ") == "padded"
