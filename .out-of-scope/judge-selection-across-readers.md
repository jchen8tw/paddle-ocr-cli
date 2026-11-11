# Letting the Judge choose between two Readers' Readings

pdf_ocr reads each Block with one Reader. It does not read every Block with both MiniCPM-OCR-1.0 and
PaddleOCR-VL-1.6 and then let the Judge (MiniCPM-OCR Jev) keep whichever Reading it Scores higher, or keep one
Reader's Reading and swap in the other's where the first scores low. The official MiniCPM-OCR repository
documents this as "Selecting across models to improve parsing quality" (`docs/minicpm_ocr_jev_en.md`), and it is
the only use of the Judge the authors publish gains for. Spec #1 left it out of scope as a possible later
extension, and a prototype then measured it.

## Why this is out of scope

It costs a lot. Every Block is read twice, and the two Readers share only the Layout model. Both Readers plus
the Judge do not fit on the 12 GB card the defaults are tuned for, so the run would need two passes over the
document. The Judge also scores twice as many Readings.

On the documents this project reads, it buys nothing. The prototype ran both Readers with the Judge on (nothing
re-read) on `台灣人壽解約金 115.09.23.pdf` and scored 13 per-Block rules with `ocr_eval.py`'s own assignment and
edit distance. Pages 38-58 are the slice the README's tuning notes use. Pages 1-37 had never been measured
before. Their formula Blocks cannot be scored (LaTeX source against drawn glyphs), so only the Blocks without
LaTeX count there.

| rule | wrong characters, pages 38-58 | wrong characters, pages 1-37 without LaTeX |
|---|---:|---:|
| PaddleOCR-VL alone | **19** | **5** |
| MiniCPM alone | 35 | 33 |
| higher Score of the two | 25 | 36 |
| keep MiniCPM, re-pick from PaddleOCR-VL below 0.5 (the official repository's rule) | 31 | 36 |
| keep PaddleOCR-VL, re-pick from MiniCPM below 0.5 | 22 | 35 |
| keep PaddleOCR-VL, switch only if MiniCPM scores higher by 0.3 | 19 | 5 |
| best possible per-Block choice (ceiling) | 19 | 1 |

No rule beats PaddleOCR-VL alone. The only rules that tie it do so by almost never switching. On pages 38-58
PaddleOCR-VL is right or better in every Block where the Readers differ, so no choice could beat it there.

The reason is the Judge, not the idea. Where these Readers disagree, the disagreement is usually a rare
Traditional character, and the Judge cannot tell the right one from the wrong one. In the 14 disagreeing Blocks
without LaTeX, the higher Score points at the better Reading 6 times, while "always PaddleOCR-VL" is right 13
times. The Judge scores MiniCPM's wrong 蔓 or 薈 at or above the correct 躉:

| page | PaddleOCR-VL (correct) | MiniCPM (wrong) |
|---|---|---|
| 51 | 躉繳, Score 0.048 | 蔓繳, Score 0.242 |
| 54 | 躉繳, Score 0.022 | 蔓繳, Score 0.061 |
| 37 | 躉繳保險單, Score 0.724 | 蔓繳保險單, Score 0.785 |

This is not our renderer (ADR-0002): Noto Sans CJK, first in the renderer's font stack, has both 躉 (U+8E89) and
蔓 (U+8513). It is the same weakness that makes Scores weak error detectors in the README's "Judge Scores against
errors" (AUC 0.83). The official repository's gain (+0.25 Overall for PaddleOCR-VL on OmniDocBench, re-picking
from two other parsers) was measured on a different kind of document and does not carry over.

## What this does not cover

- **Using a second Reader only as a review flag**, where disagreement marks a Block for a person to check and the
  kept Reading never changes. That was the better flag on both page sets and is tracked in #15.
- **Re-reading with the same Reader**, the Judge-driven Re-read schedule, which the CLI already does.

## What would reopen it

- A different Judge, or a Judge release that ranks the correct 躉 above 蔓 and 薈 on the Blocks above.
- Documents where the Readers' errors don't fall on the same Blocks, so that the best per-Block choice beats the
  better Reader alone by more than a few characters. The prototype's `select_readings.py` reports that ceiling
  for any pair of runs.

Either one is a measurement to repeat, not an argument to have again. The script, the commands and the full
tables are in `prototypes/cross_reader_selection/` on branch `worktree-proto-cross-reader-selection` (commit
b6c318a).

## Prior requests

- #8: "Best-of-N across Readers, or flag Blocks where MiniCPM and PaddleOCR-VL disagree" (the selection half; the
  flag moved to #15)
