# PaddleOCR-VL-1.6 is the default Reader, and the Judge is off unless asked for

Until [#11](https://github.com/jchen8tw/paddle-ocr-cli/issues/11), a run with no options read every Block with
MiniCPM-OCR-1.0 in the torch worker and scored every Reading with the Judge at threshold 0.5. The original spec
chose that before anything was measured. A run with no options now reads with PaddleOCR-VL-1.6, leaves the Judge
off, and starts no worker. It needs only the main paddle environment and the `PP-DocLayoutV3/` and
`PaddleOCR-VL-1.6/` model folders. MiniCPM-OCR-1.0 and the Judge stay, behind `--reader minicpm` and `--judge`,
and the old default is `--reader minicpm --judge` with unchanged output. The maintainer decided on the evidence
from one document instead of waiting for more.

## Evidence

Everything was measured on `台灣人壽解約金 115.09.23.pdf`, the only document with a text layer, on one RTX 4070 Ti
(12 GB). Pages 38-58 are the runs in the README's "Tuning notes". Pages 1-37 come from the #8 prototype runs
(`--judge-threshold 0`, so nothing was re-read), reported in the comments on #11.

- **Text.** On pages 38-58 PaddleOCR-VL without the Judge has 19 wrong characters of 16,586, against 33 for
  `--reader minicpm --judge` and 35 for MiniCPM without the Judge. On the 335 Blocks of pages 1-37 that contain no
  LaTeX, PaddleOCR-VL has 5 and MiniCPM 33. 30 of MiniCPM's 33 come from one Block on page 5, where MiniCPM also read
  a line of the overlapping next Block and so wrote that sentence twice. Without that Block it is 3 against 5, and
  MiniCPM's 3 are the same kind of rare-character misread (躉, 寶) that it makes on pages 38-58.
- **LaTeX.** Of 356 Blocks with LaTeX on pages 1-37, PaddleOCR-VL is wrong in 55 and MiniCPM in 75, classified by
  hand against crops of the page. PaddleOCR-VL is better on inline LaTeX: of the 98 such Blocks that differ, only
  PaddleOCR-VL is right in 49 and only MiniCPM in 15. MiniCPM is better on display formulas: of 68, it is right in
  14 where PaddleOCR-VL is not, and never the reverse.
- **Tables.** Both Readers give the same structure. All 132 table Blocks have the same row, cell and span counts,
  and all 119 CSVs have the same shape. On pages 38-58 all 2,389 cells match.
- **Time and memory.** On pages 38-58 the new default takes 188 s against 375 s and 389 s for the old one, at a
  card peak of 9,657 MiB against 9,540 and 9,996 MiB. On the 4-page brochure without a text layer it took 85 s
  against 214 s, at 11,037 MiB against 10,778 MiB. Fresh runs on master before the change reproduced the README's
  pages 38-58 numbers: 33 wrong characters in 373 s for MiniCPM with the Judge, 19 in 306 s for PaddleOCR-VL with
  the Judge.
- **The Judge adds nothing measurable to PaddleOCR-VL.** At threshold 0.5 it re-read 46 Blocks on pages 38-58 and
  changed none. PaddleOCR-VL's 4 wrong Blocks scored 0.77 to 0.92, so none was flagged. The Judge took the run from
  188 s to 311 s and needs the worker environment, Chromium, Node and Noto Sans CJK.

**What the switch gives up.** The default loses MiniCPM's clean display formulas. In 8 of the 68 PaddleOCR-VL
typesets the right-hand side as a subscript of `=`, so the characters are there but the formula renders wrong. It
also loses MiniCPM's lack of Simplified drift. PaddleOCR-VL wrote 16 Simplified characters on pages 38-58, all
repaired by `--script auto`, where MiniCPM wrote none. And MiniCPM needs less memory when it reads alone, 5,408 MiB
against 9,657 MiB. All three are one flag away, `--reader minicpm`.

**What is still unmeasured.** A scanned document, ICD-style tables (where PaddleOCR-VL misread 腫 as 腗 on the
original sample), and a second document with a text layer. The formula counts also come from a document that
repeats a few formulas across about 25 product sections, so they measure a few habits of each Reader. A second
document could change the ranking.

## Considered Options

- **Keep `--reader minicpm --judge` as the default and measure more documents first**, as #11 first proposed.
  Rejected: on the one document measured, it loses on text and on LaTeX overall, ties on tables, and takes twice
  as long.
- **PaddleOCR-VL with the Judge on.** Rejected: the same 19 errors in 311 s instead of 188 s, and every plain run
  would still need the worker, Chromium, Node and Noto Sans CJK.
- **Keep the Reading with the higher Score of the two Readers, per Block.** Rejected for the default: worked out
  offline from the two Judge runs on pages 38-58, it gives 24 wrong characters in Blocks, more than PaddleOCR-VL's
  18 alone, and it needs both Readers and the Judge on every run.

## What this changes in ADR-0001

[ADR-0001](0001-torch-models-in-a-separate-venv.md) called MiniCPM-OCR-1.0 "the default Reader" and an
all-transformers stack "still the likely end state". It was written when every run with no options needed torch.
Now the default needs no torch, so the second venv costs only the runs with `--reader minicpm` or `--judge`. An
all-transformers stack would now have to move the default path itself (PaddleOCR-VL through transformers'
`paddleocr_vl`, with the tuned pipeline steps rebuilt and its 19 errors on pages 38-58 matched again) only to make
the optional paths simpler. We no longer call it the likely end state. The separate torch venv of ADR-0001 stands,
for MiniCPM-OCR-1.0 and the Judge.
