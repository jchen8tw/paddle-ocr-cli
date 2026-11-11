"""The PaddleX post-processing a Reading gets, done before judging so that the Score describes the written text,
and the step that post-processing lacks: a lone symbol written as LaTeX becomes its character."""
import pytest

import minicpm_reader as mr


def clean(raw, label="text", **kw):
    return mr.clean_reading(raw, label, to_html=lambda s: "", **kw).text


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


# --- lone symbols written as LaTeX (issue #7) ------------------------------------------------------------

ISSUE_TABLE = {
    r"\geq": "≥", r"\ge": "≥", r"\geqslant": "≥",
    r"\leq": "≤", r"\le": "≤", r"\leqslant": "≤",
    r"\geqq": "≧", r"\leqq": "≦",
    r"\neq": "≠", r"\ne": "≠",
    r"\times": "×", r"\div": "÷", r"\pm": "±", r"\approx": "≈", r"\%": "%",
    r"^\circ": "°", r"^{\circ}": "°", r"\degree": "°",
    r"\rightarrow": "→", r"\to": "→",
}


def test_the_symbol_table_is_exactly_the_issues():
    assert dict(mr.LATEX_SYMBOLS) == ISSUE_TABLE


def test_a_lone_symbol_in_dollars_becomes_its_character():
    assert clean(r"年滿15足歲$\geq$者") == "年滿15足歲≥者"


def test_whitespace_inside_the_dollars_goes_with_them():
    assert clean(r"歲$ \geq $者") == "歲≥者"
    assert clean(r"$ \geq $") == "≥"


def test_a_rewritten_paren_span_adds_no_space():
    assert clean(r"年滿15足歲\(\geq\)者") == "年滿15足歲≥者"
    assert clean(r"歲 \(\geq\) 者") == "歲 ≥ 者"  # the Reader's own spaces stay


def test_a_bare_command_becomes_its_character():
    assert clean(r"年滿15足歲\geq者") == "年滿15足歲≥者"
    assert clean(r"\ge5") == "≥5"


@pytest.mark.parametrize("command,char", list(ISSUE_TABLE.items()))
def test_every_table_entry_maps_to_its_character(command, char):
    assert clean(f"甲{command}者") == f"甲{char}者"
    assert clean(f"甲${command}$者") == f"甲{char}者"


def test_longest_command_wins_and_a_following_letter_blocks_a_match():
    assert clean(r"\geqslant 0") == "≥ 0"
    for kept in (r"\left( x \right)", r"a \leftarrow b", r"\top", r"\circ", r"\\%"):
        assert clean(kept) == kept


def test_a_span_holding_more_than_one_symbol_is_kept_whole():
    for kept in (r"若 $x \geq 5$ 則", r"$\geq 5$", r"$50\%$"):
        assert clean(kept) == kept


def test_display_spans_are_not_unwrapped():
    assert clean(r"$$\geq$$") == r"$$\geq$$"
    assert clean(r"\[\geq\]") == r" $$ \geq $$ "  # what the delimiter rewrite alone makes of it


def test_an_unpaired_dollar_is_not_a_span():
    assert clean(r"a $ b \geq") == "a $ b ≥"


@pytest.mark.parametrize("label", ["paragraph_title", "doc_title", "header", "footnote", "reference_content", "ocr"])
def test_every_label_the_judge_rates_as_text_is_unwrapped(label):
    assert clean(r"年滿15足歲$\geq$者", label) == "年滿15足歲≥者"


@pytest.mark.parametrize("label", ["display_formula", "inline_formula", "formula_number", "algorithm", "table",
                                   "chart", "seal", "image"])
def test_formulas_code_tables_charts_seals_and_images_are_untouched(label):
    for raw in (r"年滿15足歲$\geq$者", r"年滿15足歲\geq者", r"$ \geq $"):
        assert clean(raw, label) == raw


def test_the_count_is_the_number_of_symbols_written():
    def count(raw, label="text"):
        return mr.clean_reading(raw, label, to_html=lambda s: "").symbols_unwrapped

    assert count(r"$\geq$ 且 \le 3 且 \(\neq\)") == 3
    assert count(r"$x \geq 5$ \left( \top") == 0
    assert count(r"$\geq$", "display_formula") == 0


CORPUS = [
    r"年滿15足歲$\geq$者", r"歲$ \geq $者", r"$ \geq $", r"年滿15足歲\(\geq\)者", r"歲 \(\geq\) 者",
    r"年滿15足歲\geq者", r"\ge5", r"\geqslant 0", r"\left( x \right)", r"a \leftarrow b", r"\top", r"\circ", r"\\%",
    r"若 $x \geq 5$ 則", r"$\geq 5$", r"$50\%$", r"$$\geq$$", r"\[\geq\]", r"a $ b \geq",
    r"X $ \geq $ Y", r"$a$\geq$b$", r"x$ $\geq$", r"\$\geq$", r"$\$$\geq$", r"costs $5 and \(\times\) 3",
    r"30^\circ C 與 30^{\circ}C", "  前後有空白 $\\leq$  ", *[f"甲{c}者" for c in ISSUE_TABLE],
]
# The delimiter rewrite pads the edges of this one and the next strip takes the padding off. That was so before the
# symbols were unwrapped (it holds for any Reading that starts with \( or \[) and the pipeline itself never strips.
EDGE_PADDED = {r"\[\geq\]"}


@pytest.mark.parametrize("raw", CORPUS)
def test_cleaning_twice_changes_nothing(raw):
    once = clean(raw, strip=False)
    assert clean(once, strip=False) == once
    if raw not in EDGE_PADDED:
        once = clean(raw)
        assert clean(once) == once


@pytest.mark.parametrize("raw", CORPUS)
def test_unwrapped_text_survives_the_pipelines_own_pass(raw):
    cleaned = clean(raw, strip=False)
    assert mr.rewrite_latex_delimiters(mr.truncate_repetitive_content(cleaned), "text") == cleaned
