# pdf_ocr: PDF to Markdown, JSON and CSV with PaddleOCR-VL

`pdf_ocr.py` converts a PDF into Markdown, a detailed JSON file and one CSV per table. It uses only
the local models in `models/`:

| Model | Role |
|---|---|
| `models/PP-DocLayoutV3` | finds the layout blocks on each page (text, titles, tables, figures, …) and their reading order |
| `models/PaddleOCR-VL-1.6` | reads each block: text, tables as HTML, formulas as LaTeX, and charts or seals if you ask |

It downloads nothing, and the model-hoster connectivity check is switched off.

## Setup

You need Python 3.13 and [uv](https://docs.astral.sh/uv/). `pyproject.toml` pins `paddlepaddle-gpu` 3.3.1
from the CUDA 13.0 Paddle index. It also pulls in `paddleocr[doc-parser]` and `opencc-python-reimplemented`,
which the Simplified to Traditional repair needs.

```bash
uv sync
```

If no CUDA GPU is found, the tool runs on the CPU, which works but is slow.

## Usage

```bash
uv run pdf_ocr.py input.pdf                         # -> output/<stem>/
uv run pdf_ocr.py input.pdf -o out --pages 1-3,7,10-
uv run pdf_ocr.py --help
```

The output goes to `output/<pdf stem>/` unless you pass `-o`:

```
<stem>.md      Markdown: headings, paragraphs, HTML tables, LaTeX formulas, <!-- page N --> markers
<stem>.json    per-page blocks with label, text, bbox, polygon, reading order, raw layout detections
imgs/          figures, charts and seals cropped from the pages (only if there are any)
csv/           one CSV per valid table: pNNN_tableKK.csv (only if there are any)
```

If any page fails, the run still finishes. The failure is recorded in the JSON (`source.pages_failed`)
and in the Markdown, and the exit code is 1.

### Options

| Option | Default | What it does |
|---|---|---|
| `-o, --output-dir` | `output/<stem>` | where to write the results |
| `--models-dir` | `./models` | folder containing `PP-DocLayoutV3/` and `PaddleOCR-VL-1.6/` |
| `--pages` | all | 1-based page selection, e.g. `1-3,7,10-` |
| `--device` | `gpu:0` if available | Paddle device, e.g. `cpu` |
| `--dpi` | 216 | page render resolution |
| `--layout-threshold` | 0.2 | layout score threshold; lower finds more blocks but too low invents boxes |
| `--max-new-tokens` | 8192 | decoding limit per block; raise it for very large tables |
| `--charts` | off | turn charts into data tables (see Caveats) |
| `--seals` | off | read the text on seals and stamps |
| `--ocr-images` | off | also OCR any text inside figures |
| `--no-merge-blocks` | | do not merge adjacent text blocks before OCR |
| `--no-recover` | | turn off the truncated-block re-read |
| `--script {auto,off}` | `auto` | Simplified to Traditional repair; `off` keeps the raw model output |
| `--md-ignore-labels` | header, footer, page number, aside | block labels left out of the Markdown (the JSON keeps everything) |
| `--no-merge-tables` | | keep a table that continues over a page break as separate per-page tables in the Markdown and CSV |
| `--no-csv` | | do not write `csv/` |
| `--no-page-markers` | | leave out the `<!-- page N -->` markers |
| `-v, --verbose` | | show the PaddleOCR / PaddleX logs |

## Output formats

### JSON

The JSON has these top-level keys:
- `source`: file name, sha256, page count, pages processed and pages that failed.
- `engine`: models, library versions and device.
- `settings`: every option that affects the output.
- `postprocessing`: what the clean-up steps changed.
- `csv_tables`: which tables were exported and which were skipped.
- `elapsed_seconds`.
- `pages`.

Each page lists its `blocks`, each with these fields:
- `id`, `order`, `label`, `content`
- `bbox` and `polygon`, in pixels of the rendered page. `scale_px_per_pt` converts them back to PDF points.
- `in_markdown`

A page also keeps its unfiltered `layout_detections` with scores. When a clean-up step changes a block,
the original text is kept next to it (`content_raw` for the script repair, `content_before_reread` for
the re-read). That way no edit is hidden.

The JSON is always faithful to each page. A table that runs over a page break stays split into its
per-page parts.

### Markdown

The Markdown is built from the same blocks, with page furniture (headers, footers, page numbers) left
out. A table that continues on the next page is merged into one table, and heading levels are
re-assigned across the whole document.

### CSV

Each `table` block becomes `csv/pNNN_tableKK.csv`, where `NNN` is the page the table starts on and `KK`
is its position among that page's tables. The CSVs follow the Markdown, so a table merged across pages
is one CSV, and `--no-merge-tables` gives one CSV per page part.

- **Only well-formed tables are exported.** The table's HTML must parse into a rectangular grid. A
  table is skipped if it has text outside cells, rows of different widths, holes, spans that run past
  the edge, or nested tables, or if the model wrote something that is not an HTML table. Each skipped
  table is reported on stderr and listed in `csv_tables.skipped` in the JSON, with the reason.
- **Merged cells** (`rowspan` / `colspan`) are expanded by repeating the cell text in every cell they
  cover. Every row then has every value, which suits filtering and pivoting.
- **Cell text:** `<br>` becomes a line break inside the cell, HTML entities are decoded and other
  whitespace is collapsed.
- **Encoding:** UTF-8 with a BOM, so Excel opens CJK text correctly.
- **Re-runs:** a run into the same folder first deletes the old `pNNN_tableKK.csv` files there. Other
  files in `csv/` are left alone.
- **JSON entries:** `csv_tables.exported` lists each file with its first page, block id, all the pages
  it covers, and its row and column counts.

On the sample PDF, all 11 table blocks were valid and became 6 CSVs, because two tables run across
several pages. One is the drug list on pages 14–16 (93 rows). The other is the ICD-10 code table on
pages 17–20 (78 rows). In each, PaddleOCR's merge drops the header row repeated at the top of each
continuation page.

## What was done for accuracy

The goal was the highest precision possible with these two models. Every default below was chosen by
scoring the output against the text layer of a 21-page Traditional Chinese insurance policy. The PDF's
own characters were assigned to layout blocks by position, so reading order does not affect the score.

| | Character error rate | Simplified chars in a Traditional doc |
|---|---|---|
| PaddleOCR-VL library defaults | 1.74 % | 188 |
| `pdf_ocr.py` | **0.40 %** | 18 |

Each change and the reason for it:

1. **Layout threshold 0.3 → 0.2.** At the default, the layout model silently dropped whole paragraphs.
   Lowering the threshold cut the characters lost to missing blocks from about 85 to about 25. At 0.1
   it starts inventing full-width boxes.
2. **Render at 216 DPI.** 288 DPI scored the same and is slower.
3. **Simplified → Traditional repair (`--script auto`).** The model drifts into Simplified forms when
   reading Traditional text (银行, 医院), and a higher resolution does not fix it. The repair runs only
   when the document is clearly Traditional: at least 30 Traditional-only characters, and no more than
   15 % of the script-specific characters Simplified. Even then it changes only characters that cannot
   be encoded in Big5 and that OpenCC maps to another form. A blanket OpenCC conversion would also
   rewrite valid Traditional characters such as 台→臺 and 后→後. Formula blocks are never touched.
4. **Truncated-block re-read.** Now and then the model returns a few characters for a whole paragraph,
   and it does so every time for the same input. The tool estimates from the ink how many glyphs a text
   block holds. If the recognised text has fewer than 35 % of that, the block is re-read from renders
   at 1.25× and 0.8×. The longer reading is kept only if it is clearly longer and still plausible. On
   the sample, this recovered a 77-character paragraph exactly.
5. **Table line breaks.** The model writes an in-cell line break as a literal `\n`. It is rewritten to
   `<br>`.
6. **Charts and seals are opt-in.** In a synthetic test, chart recognition invented data for a figure it
   took to be a chart. By default, charts and seals are saved as images instead.
7. **Cross-page restructuring only for the Markdown and CSV.** PaddleOCR's table merge moves a
   continued table's content into the first page's block and empties the later ones. Building the JSON
   from that would make whole pages look blank, so the JSON is built before the merge.
8. **Image names are prefixed with the page number.** PaddleOCR names crops by their bbox only, so crops
   from different pages could overwrite each other.
9. **`footnote` blocks stay in the Markdown.** PaddleOCR's default drops them, but they are real content.

The output is deterministic: running the same PDF twice gives identical Markdown and JSON content.

## Caveats

- The tuning was measured on one Traditional Chinese document. Other languages and layouts should work,
  but these numbers do not carry over to them.
- About 94 of the 23,390 characters in the sample are still wrong, and these are model errors:
  - a repeated 腫→腗 misread in the ICD tables
  - a few small blocks the layout model never detects
  - 干 / 于 left where 幹 / 於 was meant. They are valid Traditional characters, so the repair
    deliberately leaves them.
- The orientation and unwarping models are not in `models/`. Rotated or warped scans are therefore not
  corrected.
- If a PDF already has a real text layer, as the sample does, extracting that text directly is exact.
  OCR is what matters for scanned PDFs.
- Speed: about 3.5 minutes for the 21-page sample on an RTX 4070 Ti (12 GB), plus a few seconds to load
  the models.
- Some harmless messages still appear on stderr, such as `which: no ccache` and a "Bucketed
  engine_config" warning.
