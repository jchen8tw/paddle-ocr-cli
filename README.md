# pdf_ocr: PDF to Markdown, JSON and CSV with MiniCPM-OCR or PaddleOCR-VL, with a Score for every Block

`pdf_ocr.py` converts a PDF, or every PDF in a folder, into Markdown, a detailed JSON file and one CSV per
table. It uses only the local models in `models/`:

| Model | Role |
|---|---|
| `models/PP-DocLayoutV3` | the Layout model: finds the blocks on each page (text, titles, tables, figures, …) and their reading order |
| `models/MiniCPM-OCR-1.0` | the default Reader: reads each block, with tables as HTML, formulas as LaTeX, and charts or seals if you ask |
| `models/PaddleOCR-VL-1.6` | the other Reader, chosen with `--reader paddle` |
| `models/MiniCPM-OCR-Jev` | the Judge: rates every text, formula and table Reading from 0 to 1 (a Score), on by default; `--no-judge` turns it off |

It downloads nothing, and the model-hoster connectivity check is switched off.

It is still one command. MiniCPM needs torch and transformers, which cannot share a virtual environment
with paddlepaddle ([ADR-0001](docs/adr/0001-torch-models-in-a-separate-venv.md)). So the torch part lives in
a second uv project, `worker/`, which the main command starts as a background process for the length of
the run. You never start it yourself.

## Setup

You need Python 3.13 and [uv](https://docs.astral.sh/uv/). `pyproject.toml` pins `paddlepaddle-gpu` 3.3.1
from the CUDA 13.0 Paddle index. It also pulls in `paddleocr[doc-parser]` and `opencc-python-reimplemented`,
which the Simplified to Traditional repair needs.

```bash
uv sync                      # the main environment: the Layout model, PaddleOCR and the paddle Reader
uv sync --project worker     # the worker environment: torch + transformers for MiniCPM (about 5.6 GB, CUDA 13.2 wheels)
```

The worker environment is created in `worker/.venv`. It holds MiniCPM-OCR-1.0 and the Judge, so it is needed
unless you run `--reader paddle --no-judge`. If it is missing, the tool stops at once and tells you to run the
second command. The torch wheel from PyPI is built for CUDA 13.2, so the GPU driver must support it. The
two environments are separate processes, so their CUDA libraries never meet.

If no CUDA GPU is found, the tool runs on the CPU, which works but is slow. (The MiniCPM worker needs a CUDA GPU.)

`--models-dir` has to hold only the folders that your options use: `PP-DocLayoutV3` always, plus
`MiniCPM-OCR-1.0` for the default Reader or `PaddleOCR-VL-1.6` for `--reader paddle`, plus `MiniCPM-OCR-Jev`
unless you pass `--no-judge`. A missing folder is reported by name before anything is loaded.

**The Judge also needs three things from the machine, one of them a one-time download.** (Skip all of this with
`--no-judge`.) The tool checks them before it loads any model, and names what is missing:

```bash
uv run --project worker playwright install chromium   # once: headless Chromium for the renderer, ~650 MB in ~/.cache/ms-playwright
```

- **Chromium**, installed by the command above. Set `PLAYWRIGHT_BROWSERS_PATH` if you keep it elsewhere. On a
  fresh Linux machine `playwright install --with-deps chromium` also installs Chromium's system libraries (it
  needs root).
- **`node`** (Node.js) on `PATH`. The Judge's renderer screens each Reading's LaTeX with KaTeX in Node before
  drawing it; a Reading KaTeX cannot parse gets the verdict `precheck_reject`.
- **Noto Sans CJK** installed in the system (`fonts-noto-cjk`, `noto-fonts-cjk` or `google-noto-sans-cjk-fonts`,
  depending on the distribution; `fc-list | grep "Noto Sans CJK"` must find it). The renderer draws Latin
  letters, digits and CJK text with it.
- The **bundled font** `worker/fonts/DroidSansFallbackFull.ttf` ships in the repository (Apache-2.0, with its
  licence files) and fills the CJK glyphs Noto lacks. If it is missing the tool says so; `git checkout` restores it.

Why the system's Noto Sans CJK is needed even though the font is bundled, and why the renderer runs under its
own fontconfig, is explained in "The Judge" below.

Memory on a 12 GB GPU, measured with `nvidia-smi` (RTX 4070 Ti, 12,282 MiB, with about 2.4 GB used by the desktop
and a game during the measurements; a four-page run of a "stand-in", a digital-born Traditional Chinese proposal PDF used before the measurement document was available. Whole-run peaks on the 21 measured
pages are in "Tuning notes": 9,453 to 10,406 MiB with the Judge on, 5,408 MiB for MiniCPM alone):

| path | main process | worker | peak of the whole card |
|---|---|---|---|
| default: Layout model + MiniCPM + Jev | 874 MiB | 7,514 MiB | 10,963 MiB |
| `--reader paddle` (Judge on): Layout model + PaddleOCR-VL, Jev in the worker | 8,126 MiB at start-up, less later | 5,006 MiB | 11,393 MiB |

So the default options fit with all three models loaded at once. A very long table adds up to about 1.4 GB
while Jev scores it. `--reader paddle` alone holds about 8 GB, and PaddleOCR-VL plus Jev side by side do not fit
(about 12.5 GB before the desktop). To make them fit, the run releases PaddleOCR-VL's cached memory after
construction and after every page's read, and moves Jev's weights to the CPU while PaddleOCR-VL reads, and back
before scoring (about a second per page). If your card is shared with other heavy processes and the Judge still runs
out of memory, the page or Block that hit it is recorded (a failed page, or the verdict `forward_error` with the
reason) and the run ends with a line saying so and suggesting `--no-judge` or `--reader minicpm`.

## Usage

```bash
uv run pdf_ocr.py input.pdf                         # -> output/<stem>/
uv run pdf_ocr.py input.pdf -o out --pages 1-3,7,10-
uv run pdf_ocr.py scans/                            # every PDF under scans/ -> output/scans/
uv run pdf_ocr.py --help
```

The output of a PDF goes to `output/<pdf stem>/` unless you pass `-o`:

```
<stem>.md      Markdown: headings, paragraphs, HTML tables, LaTeX formulas, <!-- page N --> markers
<stem>.json    per-page blocks with label, text, bbox, polygon, reading order, raw layout detections
imgs/          figures, charts and seals cropped from the pages (only if there are any)
csv/           one CSV per valid table: pNNN_tableKK.csv (only if there are any)
```

If any page fails, the run still finishes. The failure is recorded in the JSON (`source.pages_failed`)
and in the Markdown, and the exit code is 1.

### A folder of PDFs

Pass a folder instead of a PDF and one run transcribes every PDF under it. Each PDF is a Document. The models
load once, then the Documents go through one at a time.

```bash
uv run pdf_ocr.py scans/            # -> output/scans/
uv run pdf_ocr.py scans/ -o results # -> results/
uv run pdf_ocr.py .                 # the current directory; the default output/<its name>/ may lie inside it
```

Finding the Documents:
- The whole tree is searched. A file is a Document when its name ends in `.pdf`, in any case (`.PDF`, `.Pdf`).
  The contents are not checked.
- A file or folder whose name starts with `.` is skipped, at any depth: `.git`, `.Trash`, and the `._report.pdf`
  files macOS leaves on other disks.
- A symlink to a PDF counts as that PDF. A symlinked folder is not entered, so a link loop or a link out of the
  tree can neither make the run endless nor pull in other PDFs.
- The same PDF at two paths is two Documents.
- The Documents go in sorted order of their path relative to the folder, folder by folder (`a/z.pdf`, then
  `a-b.pdf`, then `a.pdf`), so two runs of one folder behave alike.

The output root is `-o`, or `output/<name of the folder>/` (`.` gives the name of the current directory). A
Document at `<sub>/<name>.pdf` writes to `<root>/<sub>/<name>/`, with the files of a single PDF inside:

```
output/scans/
├── invoice/                  scans/invoice.pdf
│   ├── invoice.md
│   ├── invoice.json
│   └── csv/
└── 2024/q1/report/           scans/2024/q1/report.pdf
    ├── report.md
    ├── report.json
    ├── imgs/
    └── csv/
```

A Document's `.md`, `.json` and CSVs are the same as a run on that PDF alone gives, down to the byte, except
`elapsed_seconds`: it counts the Document's own time, without the model load that the Documents share.
`source.file` is the bare file name, as for a single PDF. `--pages` cannot be used with a folder, since the
Documents differ in length.

Before any model loads, and with nothing written, the run stops with an error when:
- the folder holds no Documents;
- two Documents would write to one output folder (`a.pdf` and `a.PDF` side by side);
- a Document's output folder would lie inside another's `imgs/` or `csv/` (`a.pdf` and `a/csv/b.pdf`).

The error lists the Documents involved; rename them. The same checks that a single PDF gets (models, worker
environment, the Judge's renderer) come next and also stop the run before the first Document.

A PDF that cannot be opened, or has no pages, is reported, counted as failed and gets no JSON, and the run goes
on. A Document whose pages fail is written as for a single PDF. The exit code is 1 if any Document or any page
failed, and 0 otherwise (a skipped Document is never a failure). On stderr, a folder run prints how many Documents it
found and how many of them are already done (see "Running a folder again"), a note for each of those, `model load: Ns`
once, a header `[i/n] <relative path> (<pages> pages)` before each processed Document's page lines and its `wrote ...`
line, and at the end the count and one line for each failure. `i` is the Document's place among all the Documents
found and `n` their number, whether or not the others are skipped. The notes of the skipped Documents come together
right after the `found` line, before anything is loaded, so the headers that follow skip those numbers:

```
found 4 Documents in scans (2 already done)
[1/4] 2023/old.pdf skipped: already done
[4/4] invoice.pdf skipped: already done
...
model load: 31.2s
[2/4] 2024/q1/report.pdf (12 pages)
...
[3/4] broken.pdf FAILED: cannot open scans/broken.pdf: Failed to load document (PDFium: Data format error).
4 Documents: 1 written, 2 skipped, 1 failed
  broken.pdf: cannot open scans/broken.pdf: Failed to load document (PDFium: Data format error).
```

A Document with failed pages is counted as written, and the lines after the count name its failed pages
(`2024/q1/report.pdf: pages 3, 7 failed (recorded in its JSON)`).

#### Running a folder again

A run on a folder skips the Documents that are already done. An interrupted run therefore picks up where it
stopped, and after you add PDFs to the folder only the new ones are processed. A Document is done when its
`<name>.json` in the output folder reads cleanly and all of these hold:
- `source.sha256` is the PDF's sha256: the PDF has not changed.
- `source.pages_failed` is empty: a Document with failed pages is tried again.
- `settings` equals the `settings` this run would write. Any option that shapes the output (`--reader`, `--dpi`,
  `--judge-threshold`, `--no-judge`, `--script`, `--no-page-markers`, `--no-csv`, ...) therefore makes every
  Document count as not done; options that do not (`-v`, `--device`, `--models-dir`) change nothing.
- The Engine's models are the same: `engine.layout_model`, `engine.reader` and, with the Judge, `engine.judge`.

Library versions and the device are not compared, so a `uv sync` or another GPU redoes nothing. A JSON that is
missing, cut off by a kill, not JSON, or missing one of the fields above makes the Document not done. A PDF that
could not be opened has no JSON, so the next run tries it again. A skipped Document is not touched.

The check reads the JSONs and hashes the PDFs, and nothing else. When every Document is done, the run prints the
summary and exits 0 without the setup checks and without loading a model, in seconds. The collisions of output
folders are still checked among all the Documents, the skipped ones included.

`--force` redoes every Document of the folder. Use it after a change the check cannot see, such as different
weights placed under the same model folder names. A single PDF is always transcribed and overwrites its output, and
`--force` with one is a usage error.

### Options

| Option | Default | What it does |
|---|---|---|
| `-o, --output-dir` | `output/<stem>` | where to write the results. For a folder it is the output root: `output/<folder name>` by default, with one subfolder per Document (see "A folder of PDFs") |
| `--models-dir` | `./models` | folder containing `PP-DocLayoutV3/` and the Reader's folder (`MiniCPM-OCR-1.0/`, or `PaddleOCR-VL-1.6/` with `--reader paddle`) |
| `--pages` | all | 1-based page selection, e.g. `1-3,7,10-`. Not with a folder |
| `--force` | | with a folder: transcribe every Document again, also the ones already done (see "Running a folder again"). Not with a single PDF |
| `--device` | `gpu:0` if available | Paddle device, e.g. `cpu` |
| `--reader {minicpm,paddle}` | `minicpm` | the model that reads each block: MiniCPM-OCR-1.0 (through the torch worker) or PaddleOCR-VL-1.6. With `paddle` the Markdown, the CSVs and the blocks in the JSON are the same as before MiniCPM was added, except that lone LaTeX symbols are written as their character (see "The MiniCPM-OCR-1.0 Reader") and, in a Traditional document, `≥`/`≤` as `≧`/`≦`; the JSON only gains a `reader` field in `engine` and `settings` |
| `--dpi` | 216 | page render resolution (144, the official SDK's value, was slightly worse and not faster with MiniCPM; see "Tuning notes") |
| `--layout-threshold` | 0.2 | layout score threshold; lower finds more blocks but too low invents boxes |
| `--max-new-tokens` | 8192 | decoding limit per block; raise it for very large tables. With MiniCPM it is the limit for tables only (see "The MiniCPM-OCR-1.0 Reader") |
| `--charts` | off | turn charts into data tables (see Caveats) |
| `--seals` | off | read the text on seals and stamps |
| `--ocr-images` | off | also OCR any text inside figures |
| `--no-merge-blocks` | | do not merge adjacent text blocks before OCR |
| `--no-judge` | | do not score blocks (no Judge, no `playwright`, `node` or Noto Sans CJK needed). The ink-based re-read of short blocks runs instead |
| `--judge-threshold FLOAT` | 0.5 | a number from 0 to 1 (anything else is rejected); blocks whose Score is below this are re-read (up to 10 Readings, see "Re-reading low-Score Blocks"); the ones still below it afterwards are flagged `below_threshold` in the JSON and counted per page on stderr. The Judge authors' own value, 0.98, flags and re-reads most Blocks for no gain; 0.5 is calibrated on one document (see "Tuning notes"). `0` keeps the Scores and re-reads nothing |
| `--no-recover` | | with `--no-judge`, turn off the truncated-block re-read. With the Judge on that re-read never runs |
| `--script {auto,off}` | `auto` | Simplified to Traditional repair, which also writes `≥`/`≤` as `≧`/`≦`; `off` keeps the raw model output |
| `--md-ignore-labels` | header, footer, page number, aside | block labels left out of the Markdown (the JSON keeps everything) |
| `--no-merge-tables` | | keep a table that continues over a page break as separate per-page tables in the Markdown and CSV |
| `--no-csv` | | do not write `csv/` |
| `--no-page-markers` | | leave out the `<!-- page N -->` markers |
| `-v, --verbose` | | show the PaddleOCR / PaddleX logs |

## Output formats

### JSON

The JSON has these top-level keys:
- `source`: file name, sha256, page count, pages processed and pages that failed.
- `engine`: the Layout model, the Reader and the Judge (`layout_model`, `reader`, `judge`), the library versions
  and the device. Whenever the worker runs (MiniCPM, or the Judge) it also has the `torch`, `transformers`,
  `playwright` and `pillow` versions.
- `settings`: every option that affects the output, including `reader`, `judge` and `judge_threshold` (null
  with `--no-judge`), `script` (`"auto"` or `"off"`, from `--script`) and `page_markers` (`false` with
  `--no-page-markers`). `postprocessing.script_repair` still records what the repair actually did. With the
  Judge, also `reread_schedule` (the four rounds, with whether this run's Reader can do each one: `supported`)
  and `max_readings_per_block`. A folder run compares `settings` and the model names of `engine` with its own to
  decide whether a Document is already done (see "Running a folder again").
- `postprocessing`: what the clean-up steps changed (whenever the worker runs, also `repetition_cuts`; with the
  Judge, a `judge` summary: `blocks_judged`, `blocks_reread`, `blocks_below_threshold` (still below the threshold
  after re-reading), `reread_stop_reasons` (a count per `stop_reason`), `readings_beyond_the_first` (every extra
  Reading made, duplicates included), a count per verdict, and how many blocks were judged before the document was
  known to be Traditional). `latex_symbols_unwrapped` counts the lone LaTeX symbols written as their character (see
  "The MiniCPM-OCR-1.0 Reader"). Whenever the worker runs (MiniCPM, or the Judge) it counts every Reading cleaned,
  re-reads included, as `repetition_cuts` does. With `--reader paddle --no-judge` it counts the symbols in the
  written blocks. In `script_repair`, `applied` is true when the document was decided Traditional, and
  `blocks_changed` counts the blocks the repair changed, whether for a Simplified character or only for a `≥`/`≤`
  glyph. `big5_symbols_changed` counts the `≥` and `≤` written as `≧` and `≦`.
- `csv_tables`: which tables were exported and which were skipped.
- `elapsed_seconds`: for a single PDF, counted from the start of the model load. In a folder run, the Document's
  own time, without the shared model load.
- `pages`.

Each page lists its `blocks`, each with these fields:
- `id`, `order`, `label`, `content`
- `bbox` and `polygon`, in pixels of the rendered page. `scale_px_per_pt` converts them back to PDF points.
- `in_markdown`
- with the Judge, on every text, formula and table block: `score` (the Judge's 0-1 rating of the written text, or
  `null`), `verdict` and `below_threshold`; charts, seals and figures have none of these (see "The Judge")
- on a block that was re-read (its first Reading scored below the threshold): `readings`, `stop_reason` and, if
  the Reader could not do some rounds, `skipped_rounds`. `content`, `score`, `verdict` and `below_threshold`
  are then those of the winning Reading. A block that passed on its first Reading has none of the three (see
  "Re-reading low-Score Blocks")

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
- **Side-by-side panels are stacked.** A long two-column table is often printed as several panels on a
  page, so its header repeats (年度 | 係數 | 年度 | 係數 | 年度 | 係數). Left as is, that header has
  duplicate column names, and anything that keys rows by name (`csv.DictReader`, most converters and
  importers) keeps only the last panel. When the first row is an exact repeat of a header whose cells
  are all different, the panels are stacked one after another into a single table, and dash or empty
  rows that pad the end of a panel are dropped. The Markdown and the JSON keep the printed layout.
- **Cell text:** `<br>` becomes a line break inside the cell, HTML entities are decoded and other
  whitespace is collapsed.
- **Encoding:** UTF-8 with a BOM, so Excel opens CJK text correctly.
- **Re-runs:** a run into the same folder first deletes the old `pNNN_tableKK.csv` files there. Other
  files in `csv/` are left alone.
- **JSON entries:** `csv_tables.exported` lists each file with its first page, block id, all the pages
  it covers, and its row and column counts (plus `panels` when side-by-side panels were stacked).

On the original sample PDF (PaddleOCR-VL; not available any more), all 11 table blocks were valid and became 6 CSVs, because two tables run across
several pages. One is the drug list on pages 14–16 (93 rows). The other is the ICD-10 code table on
pages 17–20 (78 rows). In each, PaddleOCR's merge drops the header row repeated at the top of each
continuation page. On pages 38-58 of the measurement document all the tables were valid and gave 58 CSVs.

## Tuning notes

Everything in this section was measured on one document with the commands below, on one RTX 4070 Ti (12,282 MiB;
the desktop uses about 1,000 MiB of it, which is included in every VRAM figure). One run per configuration, except
where it says otherwise. **These numbers do not come from the original 21-page sample** (sha256 `92edbf79...`),
which is not available any more, so they cannot be compared with the 0.40 % figure in "Earlier tuning" below.

### The measurement document

| | |
|---|---|
| file | `台灣人壽解約金 115.09.23.pdf` (not in the repository; PDFs are gitignored) |
| sha256 | `1fd73a59c0fb8c1ca33ce8603ca8c0a5b39ff374898730731431d86473236aa7` |
| pages | 75 in all, **pages 38 to 58 (21 pages)** in every run below |
| reference | the PDF's own text layer: 16,586 scored characters (16,628 before display formulas were left out) |

It is a Traditional Chinese insurance disclosure (surrender-value tables) with a full text layer, and **the 21
pages are table-heavy**: by scored characters, tables are 64 % (10,588), text paragraphs 20 % (3,256), footnotes
11 % (1,771), figure titles 4 %, headers and page numbers 2 %. Pages 38 to 46 are mostly clause text, a list of
product names and small tables; pages 47 to 58 are 63 to 88 % table. The tables are rate tables, mostly digits.
That limits what the document can show: **both Readers got all 10,588 table characters right** (the harness
scores a table's text, not its row and column structure), so nothing here compares table structure, and the
differences between the Readers come from about 5,000 characters of text and footnotes. Only 12 Blocks out of 320
contain an error (35 wrong characters in all), so a difference of a few characters is within what one more
document could change. The 75-page PDF was used because it has a text layer. The 4-page brochure supplied with
it has none and is only used for a qualitative run (below).

### How the error rate is computed

`ocr_eval.py` (see "Testing") assigns the text layer's characters to the layout Blocks by position and compares
each Block's Reading with them by edit distance. Two things were fixed in it while measuring, because the raw
number moved by far more than any Reader's reading:

- **The kind of dash.** The document's text layer draws the dash of every empty table cell as a full-width
  hyphen (`－`), which prints as one short bar. PaddleOCR-VL writes an em dash (`—`) for it, MiniCPM writes a
  hyphen or, in some re-reads, an em dash. That is 125 of PaddleOCR-VL's 245 raw errors, and also the 63 to 68
  extra raw errors of the 0.9 to 0.98 runs below (at 0.9 the Judge scored the em-dash Reading higher in 11 of
  11 such Blocks, by 0.003 to 0.095). None is a misread, so hyphen, en dash, em dash, minus and full-width hyphen
  are now folded together.
- **Display formulas.** A display formula is LaTeX in the Reading and typeset glyphs in the text layer, so 101
  to 132 "errors" came from one 42-character formula on page 38. Such Blocks are now skipped (and counted as
  skipped), like figures. Inline formulas in text Blocks stay scored.

The tables below show the strict number (before the fix, what `ocr_eval.py` printed at the time) next to the
fixed one, so the effect is visible. The fixed one is what `tests/baseline.json` and the end-to-end test use.

### Readers compared

Pages 38-58. "Strict" is the unfixed harness, "CER" the fixed one. Wall time is the whole CLI run, model loading
included.

| configuration | strict errors (CER) | CER (errors / 16,586) | wall | peak VRAM |
|---|---|---|---|---|
| PaddleOCR-VL, `--no-judge` | 245 (1.47 %) | **0.115 %** (19) | 188 s | 9,657 MiB |
| MiniCPM, `--no-judge` | 195 (1.17 %) | 0.211 % (35) | 234 s | 5,408 MiB |
| PaddleOCR-VL, Judge on (defaults, threshold 0.5) | 245 (1.47 %) | 0.115 % (19) | 311 s | 10,070 MiB |
| MiniCPM, Judge on, nothing re-read (`--judge-threshold 0`) | 195 (1.17 %) | 0.211 % (35) | 315 s | 9,453 MiB |
| **MiniCPM, Judge on (defaults, threshold 0.5)** | 193 (1.16 %) | **0.199 %** (33) | 389 s and 375 s (two runs) | 9,540 and 9,996 MiB |

Per label (fixed harness), wrong characters, PaddleOCR-VL against MiniCPM without the Judge: text 9 against 21
(of 3,256), footnotes 9 against 13 (of 1,771), tables 0 against 0 (of 10,588), titles, headers and page numbers 0
against 0. Per page: PaddleOCR-VL has errors on pages 38 (9), 43 (1) and 52 (9); MiniCPM on pages 38 (17), 39, 40,
43 (1), 48, 51, 52 (9) and 54. Page 38 (the formula and the inline `≧` symbols) carries 17 of MiniCPM's 35 errors;
page 52 carries 9 of each Reader's, and they are the same 9 (see below); page 43's one error is the text-layer
character that lies in no Block. Without pages 38 and 52 it is 1 error against 9.

**On this document PaddleOCR-VL reads more characters right than MiniCPM** (19 against 35 wrong). MiniCPM's lead
in the strict column is only the dash convention. The two Readers' texts differ in 10 Blocks out of 320, and
PaddleOCR-VL is right or better in all 10:

- **MiniCPM loses on rare Traditional characters.** 躉 became 蔓, 薈 or (after a re-read) 薑 in three footnotes,
  鑽 became 鑛, 閣 became 閤, 績 became 續, 寶 became 實, 屆 became 居: 9 wrong characters in 8 Blocks, all read
  correctly by PaddleOCR-VL. The same 躉 weakness shows on the brochure below.
- **MiniCPM writes symbols as LaTeX.** On page 38, 16 of its 17 errors (PaddleOCR-VL: 9) were LaTeX in three text
  Blocks. One Block held a lone symbol, `(n) \( \geq \) 10` for the printed `(n)≧10` (6 errors). One held an inline
  formula, `$ t \geq 10 $` for `t≧10` (6). One held `$ t= $` and `$ t $` around plain letters (4). PaddleOCR-VL
  wrote the first as `≥` (1 error, the wrong glyph) and wrapped the other two in `$` as well. The official SDK's
  LaTeX repair does not fix this. It fixes only formula syntax, in formula Blocks, and has no table of symbols. The
  tool now writes a lone symbol as its character, and `≥`/`≤` as `≧`/`≦` in a Traditional document
  ([issue #7](https://github.com/jchen8tw/paddle-ocr-cli/issues/7); see "The MiniCPM-OCR-1.0 Reader"). That fixes
  the first Block for both Readers. The other two hold more than one symbol and are left as they are. The counts
  in this section were measured before that step; "After issue #7" below has the new ones.
- **MiniCPM wins on Simplified drift.** PaddleOCR-VL wrote 16 Simplified characters (资 别 项) into the
  Traditional text, in 16 Blocks, all fixed by `--script auto`; without that repair they would be 16 more errors.
  MiniCPM wrote none in these runs (0 repairs), and 1 at 144 DPI.
- **MiniCPM needs less memory when it reads alone** (5,408 MiB against 9,657 MiB) but is 24 % slower (234 s
  against 188 s).

**After issue #7** (lone LaTeX symbols written as their character, `≧`/`≦` in a Traditional document). With each
Reader's defaults (Judge on, threshold 0.5, 216 DPI), MiniCPM has 27 wrong characters (0.163 %, was 33) and
PaddleOCR-VL 18 (0.109 %, was 19). One run each, before and after, 2026-11-11. Of the 321 Blocks, the only one whose
text changed is page 38's `(n)≧10`, for both Readers. Page 38 goes from 17 to 11 errors for MiniCPM and from 9 to 8
for PaddleOCR-VL, and no page has more errors than before. What is left on page 38 is the inline math around
`t≧10` and `t=` described above (10 errors for MiniCPM, 8 for PaddleOCR-VL) and 閣 read as 閤. The Judge's Score of
MiniCPM's changed Block went from 0.79 to 0.92. Wall time did not change (373 s and 305 s).
- **Dropped Blocks: none.** Neither Reader returned an empty Reading for a Block with text, and 1 text-layer
  character lies in no Block (the layout model is shared). One footnote on page 52 loses its last phrase
  (`保額：新台幣10萬`) under both Readers although the Block's box covers it; the cause was not found.
- **Not seen here:** `〔 〕` read as `[ ]`, and the 腫→腗 misread (the document has no ICD tables).

### Judge Scores against errors

The Judge was run on every Block with the re-read switched off (`--judge-threshold 0`), so the Scores belong to
the first Reading. Of the 320 scored Blocks (the display formula excluded), 12 contain an error.

- The Scores separate wrong from right, **but weakly and on few cases**: the median Score of the 12 wrong Blocks is
  0.55, of the 308 right ones 0.91, and the chance that a wrong Block scores below a right one is 0.83 (an AUC
  from 12 wrong Blocks, so wide error bars). The five lowest Scores (0.004 to 0.24) are all rare-character
  misreads. The Blocks with markup errors (`$`, `\geq`) score 0.77 to 0.88, the dropped phrase 0.92: not
  separable from correct text.
- **Many correct Blocks score low.** 38 right Blocks score below 0.5, most of them one-line product names such as
  `險種：台灣人壽…` (0.2 to 0.5). The Score depends on the label: headers 0.998, page numbers 0.994, figure titles
  median 0.52, tables 0.89, text 0.84.
- For PaddleOCR-VL the four wrong Blocks scored 0.77 to 0.92 (none below 0.5); four cases say nothing about
  separation.
- **The higher-Scored of the two Readers per Block** (offline, from the two Judge runs) gives 24 wrong characters
  in Blocks, between PaddleOCR-VL alone (18) and MiniCPM alone (34): it picks the better Reading in 5 of the 10
  Blocks where the Readers differ. For the rare-character Blocks the Judge scores both Readings low (0.02 to 0.35).

### Threshold calibration (MiniCPM, Judge on)

321 judged Blocks. "Re-read" is the share of Blocks whose first Score was below the threshold; "still below" is the
share still below it after up to 10 Readings (these are flagged `below_threshold`). The run without re-reading (0)
is the time reference.

| `--judge-threshold` | Blocks re-read | still below | extra Readings | strict errors | CER (errors) | wall | peak VRAM |
|---|---|---|---|---|---|---|---|
| 0 (Scores only) | 0 | 0 | 0 | 195 | 0.211 % (35) | 315 s | 9,453 MiB |
| **0.5 (new default)** | 44 (13.7 %) | 43 (13.4 %) | 396 | 193 | **0.199 % (33)** | 389 s (+23 %) | 9,540 MiB |
| 0.7 | 72 (22.4 %) | 71 (22.1 %) | 648 | 195 | 0.199 % (33) | 473 s (+50 %) | 9,985 MiB |
| 0.9 | 155 (48.3 %) | 155 (48.3 %) | 1,395 | 258 | 0.199 % (33) | 848 s (+169 %) | 10,406 MiB |
| 0.95 | 240 (74.8 %) | 240 (74.8 %) | 2,160 | 263 | 0.199 % (33) | 965 s (+206 %) | 10,089 MiB |
| 0.98 (the Judge authors' value, the old default) | 267 (83.2 %) | 267 (83.2 %) | 2,403 | 263 | 0.199 % (33) | 1,090 s (+246 %) | 10,277 MiB |

What re-reading did, in full: of the 44 Blocks read again at 0.5, **two changed**. The 鑽→鑛 misread became right
(2 errors fewer; Score 0.08 to 0.28, still below 0.5, so still flagged). The 躉→薈 misread became 躉→薑: still wrong,
but scored **0.87**, so it passed and **lost its flag**. A re-read can replace a flagged wrong Reading with an
unflagged wrong one. At 0.7 one more Block changed, only in the kind of dash; from 0.9 up at least 11 more
did, which is the whole difference of the strict column. No threshold changed more than those 2 characters,
no Block ever reached 0.98 (`exhausted` for every Block still below), and at 0.98 the 2,403 extra Readings
contained only 38 texts that had not been read before.

The default is **0.5**: the lowest value tried (other than 0) and the cheapest one at which re-reading fixed anything
(+23 % time, 14 % of Blocks flagged); every higher value made the same fix at a higher cost. That is not a strong calibration: it rests on one fixed Block, and
with 12 wrong Blocks another pick would be as defensible. The Judge authors' 0.98 is not an option on this evidence:
it re-reads and flags 83 % of the Blocks and takes 3.5 times as long for nothing more. The flag is a weak review
queue: at 0.5 it holds 6 of the 12 wrong Blocks among 44 Blocks (14 %); at 0.9, 11 of 12 among 154 (48 %); at 0.95
and above all 12, among 75 to 83 % of all Blocks. For PaddleOCR-VL (only the 1.25x round can run) the default
re-read 46 Blocks (14 %) and none changed: 46 extra Readings, all `exhausted`, the same 19 errors.

### Render DPI

| `--dpi` | CER (errors) | Blocks re-read at 0.5 | Simplified repairs | wall | peak VRAM |
|---|---|---|---|---|---|
| **216 (default, kept)** | 0.199 % (33) | 44 | 0 | 389 s and 375 s | 9,540 and 9,996 MiB |
| 144 (what the official SDK renders at) | 0.241 % (40) | 56 | 1 | 470 s | 9,723 MiB |

144 DPI is slightly worse (7 more wrong characters, one Simplified drift, 2 text-layer characters outside every
Block against 1) and, in this run, not faster: the Judge scored more Blocks below 0.5 (56 against 44), and the
re-reads cost more than the smaller images save. One run each, and the layout model sees a different image, so the
Block sets differ (320 against 321 Blocks): a small difference, not a measured cause. The default stays 216.

### Repetition cuts, script repairs, time and memory

- **Repetition cuts: 0** in every MiniCPM run on the 21 pages and on the brochure. The loop cut was therefore not
  exercised by this document (its tests use synthetic loops). **Script repairs on MiniCPM output: 0** (1 character
  at 144 DPI); on PaddleOCR-VL output 16 characters. Since issue #7 the repair also writes one `≥` as `≧` on page
  38 for each Reader (`big5_symbols_changed`: 1), and `latex_symbols_unwrapped` is 1 for MiniCPM and 0 for
  PaddleOCR-VL, which wrote that `≥` directly.
- **Time.** The Judge adds 81 s to the 21 pages with MiniCPM (234 s to 315 s, 3.9 s per page) and 122 s with
  PaddleOCR-VL (188 s to 311 s, including its 46 re-reads). Default MiniCPM takes 375 to 389 s, about 18 s per page.
- **GPU memory.** The whole card peaked at 9,453 to 10,406 MiB with the Judge on, 5,408 MiB for MiniCPM without the
  Judge and 9,657 MiB for PaddleOCR-VL without it. No run hit a Reader or Judge memory error. The higher
  thresholds peaked a little higher, probably from 4-sample rounds on tables (not checked).

### Determinism

The default configuration was run twice, at different times (`--judge-threshold 0.5` given explicitly, then the plain
default). The Markdown, all 58 CSVs and the JSON, **Scores and re-reads included, were identical** apart from
`elapsed_seconds`. That holds for one machine and one GPU (see "The Judge" for the caveat).

### A brochure without a text layer

`台灣人壽金利樂利率變動型養老保險_商品DM_20260529適用.pdf` (4 pages, no text layer, sha256
`204d75673097d0673386a982d256c5ef533513365a19d4336356a15d9652206c`) was run once with the defaults (MiniCPM, Judge
on, threshold 0.5, 216 DPI). Nothing exists to compute an error rate against, so this is qualitative. It
finished with exit code 0 in 214 s (the longest page 91 s, 40 Blocks), peak card memory 10,778 MiB, no crash. 102
Blocks were judged; 25 were re-read and 22 were left below 0.5 and flagged. Pictures (cover photo, icons, logos)
were saved as images and not read, no chart data was invented (charts are opt-in), and 5 tables were exported to
CSV. I read the Markdown of pages 1 to 3 and compared only the page 3 heading with the page image; the two rate
tables look plausible and I did not check their numbers. What was wrong: the title 躉繳 on page 3 was read as 薹繳 (and 躉 as 臺 in the
text beside it; the same character MiniCPM misses above, and the Judge flagged that title at 0.42), and a numbered
clause "三、…" on page 2 became a `##` heading because the layout model labelled it a title (Score 0.96). Text inside
pictures, such as the title 範例說明 on page 3, is not read by default (`--ocr-images`).

### What these measurements cannot say

One document, one language, 21 of its 75 pages, one GPU. The table characters (64 % of the reference) are right
for both Readers, so the comparison rests on about 5,000 characters and 12 wrong Blocks. The original sample may
rank the Readers differently, and a scanned document was not measured at all. Table structure, merged cells and
reading order are not scored by `ocr_eval.py`. Scores are not promised to be the same on another GPU.

Follow-ups these measurements suggest are filed as issues, not done here: the symbols written as LaTeX (#7, done
since), best-of-N or disagreement flags across Readers (#8), the re-read clearing a wrong Block's flag (#9),
per-label thresholds (#10), re-measuring the Reader choice on more documents (#11), scoring table structure and
formulas (#12), the dropped footnote phrase on page 52 (#13) and the brochure's heading and picture text (#14).

## Design choices

### Earlier tuning on the original sample (PaddleOCR-VL)

**The numbers and steps in this part were measured with PaddleOCR-VL-1.6 (`--reader paddle`) on the original
21-page sample, which is not available any more, with an earlier throwaway harness. They are not comparable with the
measurements above.** Steps marked "both" live in the pipeline and apply to both Readers.

The goal was the highest precision possible with these models. The PDF's own characters were assigned to layout
blocks by position, so reading order does not affect the score.

| | Character error rate | Simplified chars in a Traditional doc |
|---|---|---|
| PaddleOCR-VL library defaults | 1.74 % | 188 |
| `pdf_ocr.py` | **0.40 %** | 18 |

Each change and the reason for it:

1. **Layout threshold 0.3 → 0.2** (both). At the default, the layout model silently dropped whole paragraphs.
   Lowering the threshold cut the characters lost to missing blocks from about 85 to about 25. At 0.1
   it starts inventing full-width boxes.
2. **Render at 216 DPI** (both). 288 DPI scored the same and is slower; 144 is a little worse with MiniCPM (see
   "Render DPI").
3. **Simplified → Traditional repair (`--script auto`)** (both). PaddleOCR-VL drifts into Simplified forms when
   reading Traditional text (银行, 医院), and a higher resolution does not fix it (MiniCPM drifted far less, see
   above). The repair runs only when the document is clearly Traditional: at least 30 Traditional-only
   characters, and no more than 15 % of the script-specific characters Simplified. Even then it changes only
   characters that cannot be encoded in Big5 and that OpenCC maps to another form. A blanket OpenCC conversion
   would also rewrite valid Traditional characters such as 台→臺 and 后→後. Formula blocks are never touched.
   The same Big5 rule picks the glyph of `≥` and `≤`. Neither is in Big5, so in a document decided Traditional the
   repair writes them as `≧` and `≦`, the Big5 forms that the measurement document prints. It does so also in a
   block that has no Simplified character.
4. **Truncated-block re-read** (both; with the Judge on it is replaced by the Judge's re-reading). Now and then
   PaddleOCR-VL returns a few characters for a whole paragraph, and it does so every time for the same input. The
   tool estimates from the ink how many glyphs a text block holds. If the recognised text has fewer than 35 % of
   that, the block is re-read from renders at 1.25× and 0.8×. The longer reading is kept only if it is clearly
   longer and still plausible. On the sample, this recovered a 77-character paragraph exactly.
5. **Table line breaks.** PaddleOCR-VL writes an in-cell line break as a literal `\n`. It is rewritten to
   `<br>` (MiniCPM: see below).
6. **Charts and seals are opt-in** (both). In a synthetic test, PaddleOCR-VL's chart recognition invented data for a
   figure it took to be a chart. By default, charts and seals are saved as images instead.
7. **Cross-page restructuring only for the Markdown and CSV** (both). PaddleOCR's table merge moves a continued
   table's content into the first page's block and empties the later ones. Building the JSON from that would make
   whole pages look blank, so the JSON is built before the merge.
8. **Image names are prefixed with the page number.** PaddleOCR names crops by their bbox only, so crops from
   different pages could overwrite each other.
9. **`footnote` blocks stay in the Markdown.** PaddleOCR's default drops them, but they are real content.

With PaddleOCR-VL the output is deterministic: running the same PDF twice gives identical Markdown and JSON content
(the same was confirmed for MiniCPM with the Judge, above).

### The MiniCPM-OCR-1.0 Reader

MiniCPM-OCR-1.0 is the default Reader. The Layout model, block merging, reading order, cross-page table
merging and the Markdown, JSON and CSV outputs are the same for both Readers, because the PaddleOCR
pipeline still does all of that. It hands each cropped block to the worker instead of to PaddleOCR-VL
(see "How the worker is wired in" below). The worker reads one block at a time:

- It loads the model as a native transformers image-text-to-text model (no remote code) in bf16.
- The prompt depends on the block label: `Text Recognition:` for text-like blocks and inline formulas,
  `Table Recognition:`, `Formula Recognition:` for display formulas, `Chart to Table:` and
  `Seal Recognition:`. The image comes before the text, and thinking is off.
- Decoding is greedy with `no_repeat_ngram_size=100`.
- The token limit depends on the block: 4096 for text, 512 for titles, headers and footers, 2048 for formulas
  and references, 1024 for footnotes, 256 for seals and page numbers, 4096 for charts. For tables it is
  `--max-new-tokens`. A block that hits its limit cannot run away.
- Charts and seals are still opt-in (`--charts`, `--seals`).

The raw Reading is then cleaned in this order:

1. **Repetition cut.** A Reading that falls into a loop is cut back (ported from the official MiniCPM-OCR
   SDK). The JSON counts the cuts in `postprocessing.repetition_cuts`.
2. **Lone symbols.** In a text block, a printed symbol that the Reader wrote as a LaTeX command becomes its
   character: `$\geq$`, `$ \geq $`, `\(\geq\)` and a bare `\geq` all become `≥`, with no space added. The step
   runs after the pipeline's rewrite of `\(...\)` into ` $ ... $ `, and drops the space that rewrite puts before
   the symbol. The commands are those in `minicpm_reader.LATEX_SYMBOLS`, such as `\geq`, `\leq`, `\geqq`, `\neq`,
   `\times`, `\div`, `\pm`, `\approx`, `\%`, `^\circ` and `\rightarrow`. A command counts only when no letter
   follows it, so `\left(` and `\top` stay as they are. A math span that holds more than the symbol (`$x \geq 5$`)
   and display math stay too, and so do formula, `formula_number`, `algorithm`, table, chart and seal blocks. The
   JSON counts the symbols in `postprocessing.latex_symbols_unwrapped`. PaddleOCR-VL's Readings get the same step.
   The bridge does it when the Judge runs, and `pdf_ocr.py` does it after the pipeline with `--no-judge`.
3. **OTSL to HTML.** MiniCPM writes tables in OTSL. They are converted with the same converter PaddleX uses.
4. **Table line breaks.** MiniCPM writes an in-cell line break as a *real* newline (PaddleOCR-VL writes a
   literal `\n`). Both become `<br>`, so the CSV keeps the line break inside the cell. This happens inside
   the Reader, so `postprocessing.tables_with_line_breaks_fixed` (the count of the literal `\n` fix) stays 0
   with MiniCPM.
5. **Simplified to Traditional repair**, as for PaddleOCR-VL, with the `≧`/`≦` glyphs. It is document-level and
   runs after all pages, which is why it comes after the Reader and not inside it.

The truncated-block re-read (`--no-recover`) also works with MiniCPM, but only without the Judge (`--no-judge`).

What differs from PaddleOCR-VL in the text. Seen on a digital-born proposal PDF used before the measurement
document (not seen on it, where full-width forms are normalised away and nothing of the kind showed up in the
errors): MiniCPM keeps the printed spacing around Latin letters and digits (`表 1-1`, `OCR 及`), writes `[...]` for
the printed `〔...〕`, and uses `(...)` for full-width brackets inside table cells. Seen on the measurement
document: it writes inline symbols such as `≧` as LaTeX (`$\geq$`, which the clean-up above unwraps), misreads
some rare Traditional characters (躉, 鑽, 閣, 績, 寶, 屆) that PaddleOCR-VL reads correctly, and does not drift
into Simplified characters the way PaddleOCR-VL does. It can also misread the odd character the repair cannot fix:
on the proposal, `爰` came out as the Simplified `爱`, which the repair turned into `愛`. Numbers are in
"Tuning notes".

The measurements of this Reader against PaddleOCR-VL are in "Tuning notes" above.

### The Judge

The Judge is MiniCPM-OCR Jev. It does not read; it looks at the Block's image and at an image of the Reading
rendered with Chromium and KaTeX, and answers how likely the Reading is to be correct. That is the **Score**, from 0
to 1. It is on by default and works with either Reader. A Block whose Score is below `--judge-threshold` is read
again, up to nine more times, and the best Reading is kept (next section).

- **What is judged.** Text, titles, footnotes, references (category `text`), inline and display formulas
  (`formula`) and tables (`table`). Charts, seals and figures are never judged (the Judge was not trained on
  them): they carry no `score`, `verdict` or `below_threshold`.
- **Per Block in the JSON.** `score` is the Judge's number for the *written* text. `verdict` says whether it is
  real. `below_threshold` is true when the Block should be looked at.

  | `verdict` | meaning | `score` | `below_threshold` |
  |---|---|---|---|
  | `scored` | a real Score | 0-1 | `score < --judge-threshold` |
  | `precheck_reject` | KaTeX cannot parse the Reading's LaTeX: the Reading is unusable | null | true |
  | `render_error` | the renderer failed: the Reading is unusable | null | true |
  | `missing_original_image` | the renderer got no readable crop: the Reading is unusable | null | true |
  | `encode_error` | the images could not be encoded: scoring never happened | null | true |
  | `forward_error` | the model (or the worker) failed: scoring never happened | null | true |

  The official SDK reports 0.0 for the first three and -1.0 for the last two; those are *not* Scores, so they
  are `null` here, and a failed Block is flagged rather than given a fake low Score. `verdict_detail` carries the
  reason. A failure on one Block never stops the run: the Reading is kept and the failure recorded. A page whose
  Judge call fails entirely gets `forward_error` on all its judged Blocks.
- **On stderr**, each page's line says how many of its judged Blocks are below the threshold (after re-reading)
  and how many were re-read, e.g. `[  2/21] page 39: 23 blocks, 20.6s, 4 of 23 judged below the threshold, 4 re-read` (measurement document,
  threshold 0.5), with the number not scored in brackets when there are some. The Markdown contains no Scores or markers.
- **Scores are per page.** The JSON is built before the cross-page table merge, so a Score never covers only half
  of a merged table.
- **Same text as written, give or take one space.** Before a Reading is judged it gets the same clean-up it gets for the output:
  the repetition cut, `\(...\)` and `\[...\]` rewritten to ` $ ... $ ` / ` $$ ... $$ ` (the pipeline does that
  anyway, afterwards; with the delimiters rewritten a Reading scored 0.9895 against 0.9951 for the raw `\(...\)`
  form on the SDK's text example, and both render as math), lone LaTeX symbols in text blocks written as their
  character, OTSL converted to HTML for tables, line breaks inside table cells turned into `<br>`, and the
  Simplified to Traditional repair (below). Raw OTSL scored about 0.0005 for a correct table, which is why tables
  go to the Judge as HTML. The one difference: the pipeline's rewrite turns the
  model's `\[ x \]` into `$$  x  $$` (two spaces, since the model already spaces its formulas), and the real Judge
  rated that 0.73 where `$$ x $$`, which renders the same in the Markdown, got 0.97 (SDK formula example). So the Judge
  is shown a formula with single spaces inside `$`/`$$`; the written text keeps what the pipeline writes.
- **One Chromium launch per batch.** The pipeline hands the Reader a page's Blocks in one call (one call per
  pixel-budget group, which on the stand-in was always one per page: four pages, four score requests), and the
  Readings of that call are rendered in one batch (a launch costs about 0.8 s) and scored one at a time (batching
  changes the Scores slightly). On the measurement document the Judge adds 3.9 s per page with MiniCPM (see
  "Tuning notes").

**The repair is decided while reading, so the Judge sees repaired text.** The Simplified to Traditional repair is
document-level: it needs the whole document to decide the document is Traditional (at least 30 Traditional-only
characters and at most 15 % Simplified). The Judge works per Block, so the decision is made on the Readings seen so
far, in the order the pipeline hands the Blocks over, and stays made: once the rule holds, every later Reading is repaired before it is judged.
The Reading that completes the evidence is repaired too. From then on the Judge also sees `≥` and `≤` as `≧` and
`≦`. Same input in the same order always gives the same decision. The document-level repair still runs afterwards
on the original Readings, exactly as before, so the written text is unchanged by the Judge. Two consequences, both
recorded in the JSON:

- A Block read *before* the decision was judged unrepaired, so its Score may describe the pre-repair text. Those
  Blocks have `judged_before_script_decision: true`. On the stand-in this was the first four Blocks of the first
  page; with a non-Traditional document the decision never happens and nothing is repaired.
- If the document-level repair ends up changing a Block differently from the sticky one (for example the sticky
  decision came early and the full survey disagrees), the block has `judged_text`, the text that was actually
  scored, next to `content`, the text that was written.

### Re-reading low-Score Blocks

A Block whose Score is below `--judge-threshold` is **re-read**, and the Reading with the highest Score is kept. The
schedule is fixed; each step runs only while the best Score so far is below the threshold:

| round | how the Block is read | Readings |
|---|---|---|
| 1 | greedy, the Block's own crop (the normal first Reading) | 1 |
| 2 | greedy, from a 1.25x render of the page | 1 |
| 3 | sampled, temperature 0.3 (top-k 20, top-p 0.8), seeded | 4 |
| 4 | sampled, temperature 0.7 (top-k 20, top-p 0.8), seeded | 4 |

- **At most 10 Readings per Block**, duplicates included. A text that has been read before is **scored only once**
  (MiniCPM's sampled Readings are often identical, and so is the 1.25x one for most Blocks).
- **The highest `scored` Reading wins**, not the latest; a tie goes to the earliest. Readings with another verdict
  (`precheck_reject`, `render_error`, ...) never win. If none was scored the first Reading is kept.
- **Every Reading is cleaned, repaired and judged exactly like the first** (same clean-up, the same sticky
  Simplified to Traditional repair, the same formula spacing for the Judge). A re-read does not change the repair
  decision: only the first Reading of a Block counts towards it. The Judge always compares with the Block's own
  crop, whatever scale the Reading was made from.
- **In the JSON**, a re-read Block has `readings`, one entry per *distinct* text: `content` (the text as it was
  judged), `score`, `verdict`, and how it was produced: `round`, `decode` (`greedy` or `temperature`), `temperature`
  and `scale` (plus `scale_source` for a scaled Reading); `duplicates` counts the later Readings that gave the
  same text. `stop_reason` says why the re-reading
  ended: `passed` (a Reading reached the threshold), `exhausted` (all rounds done, none did), `no_scored` (all
  rounds done and no Reading could be scored; the first Reading is kept), `judge_error` (the Judge failed on a
  Reading, or on the whole batch, part-way: the best Reading so far is kept). If the Judge fails on the *first*
  Reading nothing is re-read (there is no Score to improve on) and the Block just has `verdict: forward_error`.
  A fifth value, `reader_error`, is not in the schedule's spec: if the Reader fails on a re-read round (for example
  the GPU runs out of memory on a 4-sample table) the Block keeps its best Reading so far and the page is not
  lost; `stop_detail` has the message. A Block that passed on its first Reading has none of these fields and cost
  no extra Reading.
- **Seeds are deterministic.** The seed of a round is a hash (blake2b, not Python's `hash()`) of the document's
  sha256, the page, the Block's index on its page and the round, so it is the same in every process, and a Block's
  seed does not depend on which other Blocks are on the page. The worker seeds torch immediately before the one
  `generate` call that draws the 4 samples. Two runs of the same PDF on the same machine gave identical JSON, re-reads
  included (`tests/test_e2e_reread.py`). As with Scores, this is not promised across GPUs or library versions.
- **What each Reader can do.** MiniCPM-OCR-1.0 does all four rounds. The local PaddleOCR-VL ignores temperature and
  top_p (PaddleX says so: "not supported by the local model"), so it is greedy and cannot honestly sample: with
  `--reader paddle` only round 2 (the 1.25x input) is run, rounds 3 and 4 are skipped and listed in the Block's
  `skipped_rounds` (and flagged `supported: false` in `settings.reread_schedule`) instead of being faked with
  repeats. PaddleOCR-VL resizes every crop to its own pixel budget, so round 2 often gives the same text again.
- **Round 2 reads a 1.25x render of the page.** The page is rendered again at 1.25 times the `--dpi`, and the
  Block's box is cut from it, so the Reader gets new pixels. That is only done when the image the pipeline gave the
  Reader is exactly the plain crop of the page's own render at that box (checked pixel for pixel): a merged Block,
  a formula with its margins cut, a table with figure placeholders painted in, or a Block cut with a polygon mask is
  not, and its own crop is enlarged by 1.25 (Lanczos) instead. Each Reading says which one it came from
  (`scale_source`: `page_render` or `crop_resample`), and `postprocessing.judge.scaled_images_by_source` counts them.
  On the measurement document the page render was used for all 44 re-read Blocks at threshold 0.5 (MiniCPM); on a stand-in
  proposal PDF for 60 of 61 (MiniCPM) and 23 of 24 (paddle). It did not help much: see "Tuning notes".
- **Batched per page.** Round by round, only the Blocks still below the threshold are read, and each round is one
  batch of Reader requests and **one** Judge call (one Chromium launch) for the call's new Readings, not one per
  Block. The pipeline hands the Reader one call per pixel-budget group of a page, so this is per call.
  A sampled round costs one worker request per Block (4 samples decoded together, with only the last position's
  logits computed at prefill, which keeps a 4-sample table read inside the card's memory next to the Judge).
- **Cost, measured on the measurement document** (pages 38-58, 321 judged Blocks, MiniCPM Reader, RTX 4070 Ti; the
  table is in "Tuning notes"). At the old default of 0.98, 267 of the 321 Blocks (83 %) were re-read and none reached
  0.98; the run took 1,090 s against 315 s when nothing is re-read (`--judge-threshold 0`), 3.5 times as long, and
  the error rate was the same as at 0.5. The 2,403 extra Readings contained 38 texts not read before. **At the
  default of 0.5, 44 Blocks (14 %) were re-read, the run took 389 s (+23 %), and two Blocks changed**, one for the
  better and one to another wrong Reading with a high Score that cleared its flag. With the paddle Reader the
  1.25x round gave the same text for all 46 re-read Blocks. The schedule, as specified, spends most of its time on
  duplicates.
- **GPU memory.** With the Layout model, MiniCPM and Jev loaded, `nvidia-smi` (whole card, 1 s samples, about 1 GB of
  it other desktop processes) peaked at 9,453 to 10,406 MiB of 12,282 MiB over the threshold runs on the measurement
  document (10,778 MiB on the brochure), and at 10,070 MiB for the paddle run. The tables in those runs were small (319 to 380 characters of HTML). On the largest table crop available (1369 x
  1091 px, 2561 prompt tokens, 239 generated tokens) the worker alone went from 8.2 GB (both models idle) to 9.8 GB
  while decoding 4 seeded samples at 0.7, and the second call with the same seed gave identical texts. **The worst
  case, 4 samples of a table that runs to the 8192-token limit, was not run.** The 4-sample rounds rely on the worker
  computing only the last position's logits at prefill (`worker/minicpm_model.py`); without it a 4-sample table Reading
  ran out of memory in the spike. A Reader error in a re-read round (including running out of memory) ends that Block
  with `reader_error` and keeps its best Reading, and the end-of-run hint about GPU memory is shown.

**Scores are not bit-for-bit reproducible across machines**, and the authors say thresholds must be calibrated on
your own data. On one machine and one GPU the same PDF gives identical Scores (the end-to-end test runs it twice
and compares the JSON).

**The Judge authors' threshold of 0.98 flags most Blocks; the default is 0.5.** 0.98 is the authors' uncalibrated value
(`tau`). On the measurement document 83 % of the Blocks score below it, the correct ones included, because a
correct Reading usually scores well under 0.98 (median 0.91). 0.5 flags 14 % of the Blocks and holds half of the
wrong ones; 0.9 flags 48 % and holds 11 of 12 (see "Judge Scores against errors" and the threshold table in
"Tuning notes"). `--judge-threshold` changes the flag and the re-reading; the Score itself does not depend on it.

**Why the renderer needs the system's Noto Sans CJK, and how it is made reproducible.** The renderer is the
authors' own, vendored unchanged in `worker/render/` ([ADR-0002](docs/adr/0002-judge-uses-the-official-renderer.md)
and its amendment). The first plan was to bundle DroidSansFallbackFull *instead of* the system's Noto Sans CJK.
That does not work: Droid has no Latin letters, digits or Greek, so English text renders as empty boxes. In the
authors' setup Latin and digits come from Noto Sans CJK and Droid only fills CJK gaps. So the bundled Droid font
works *behind* the system's Noto Sans CJK. The renderer runs under a private fontconfig that lists only those two,
with the rendering rules (slight hinting, antialiasing, LCD filter) written out; they are not left to the machine,
because a file with the font folders alone changed the Scores (table example 0.025 instead of 0.032, the authors
observed 0.041). With the rules, Scores on the SDK's three reference examples deviate from the authors' by 0.0017 on
average and 0.009 at most (their own expectation is about 0.005). The full comparison is in the ADR.

### How the worker is wired in

The main command builds the PaddleOCR pipeline with the Layout model only (PaddleOCR-VL's weights are not
loaded for `--reader minicpm`). Then it replaces the pipeline's per-block recognizer with a bridge that
sends each crop to the worker as a PNG file in a temporary folder and reads the answer back as one line of
JSON. The bridge hands a page's Blocks to `block_reader.py`, "read a Block": a plain module (no GPU, no paddle)
that takes the Blocks, a Reader and a Judge, cleans and repairs each Reading, has the page judged in one batch,
re-reads the Blocks below the threshold round by round (the schedule above), and returns the final text with its
Score, verdict and flag. A Reader tells it which rounds it supports. The worker has two jobs over the same
protocol: `read` (MiniCPM; greedy, or `decode: {mode: sample, temperature, top_k, top_p, n, seed}` for `n` seeded
samples) and `score` (render and judge a batch of Readings). With `--reader paddle` the bridge calls
PaddleOCR-VL's own model as the Reader and the worker only judges, so the worker loads no MiniCPM. The worker is a long-lived process: the model is loaded once, and the main command stops it at the
end of the run, whatever happens. The code is in `reader_bridge.py` (the hook), `worker_client.py` (the
process and protocol) and `worker/` (the torch side).

The hook uses three private PaddleX names. A start-up check verifies them and stops with a clear message
if a PaddleX upgrade has moved them (`uv sync` restores the pinned versions), and a test runs the hook
against the real pipeline.

If the worker crashes or stops answering, the page it was reading is recorded in `source.pages_failed`
(exit code 1) with the worker's last messages, and the next page starts a fresh worker. Every wait has a
timeout, so a stuck worker cannot hang the run.

## Caveats

- **The numbers are from one document.** The defaults were measured on pages 38 to 58 of one Traditional Chinese
  insurance disclosure (table-heavy, 12 Blocks with errors; "Tuning notes"), not on the original sample, and the
  threshold default in particular rests on a single corrected Block. Other languages and layouts should work, but
  these numbers do not carry over, and a scanned document was not measured at all. Re-check `--judge-threshold`
  on your own documents (`ocr_eval.py` needs a text layer; the JSON's `score` and `below_threshold` do not).
- **On the measurement document MiniCPM, the default Reader, read fewer characters right than PaddleOCR-VL**
  (27 against 18 wrong of 16,586 with the defaults), mostly rare Traditional characters (躉, 鑽, 閣, 績, 寶, 屆) and
  short inline math written as LaTeX around printed text (`$ t \geq 10 $` for `t≧10`). It needs about half the GPU memory when it reads alone and does not drift into Simplified
  characters. `--reader paddle` is a reasonable choice for a Traditional document where those characters matter.
- **The Judge is a weak error detector.** At the default 0.5 it flags 14 % of the Blocks and about half of the
  Blocks with an error; a correct Reading often scores low (one-line product names, figure titles), and a re-read can
  replace a flagged wrong Reading by a wrong one with a high Score (躉 became 薑 at 0.87) that is no longer flagged.
  Treat `below_threshold` as a hint, not a verdict. Scores are only as comparable as the machine that made them.
- The Judge adds three setup items (Chromium, `node`, Noto Sans CJK) and, with MiniCPM, 3.9 s per page; with the
  default threshold the re-reads add about 23 % more run time (375 to 389 s against 315 s for the 21 pages). At the
  authors' 0.98 most Blocks are re-read and the run takes 3.5 times as long. `--no-judge` avoids all of it
  (234 s); `--judge-threshold 0` keeps the Scores and turns the re-reading off.
- `--reader minicpm` and the Judge need the worker environment (`uv sync --project worker`) and a CUDA GPU. The worker's
  torch wheel targets CUDA 13.2; a driver that supports only CUDA 13.0 needs a different torch index in
  `worker/pyproject.toml`. This was not tested.
- A page of a table-heavy PDF takes about 18 s with the defaults (MiniCPM, Judge on) on an RTX 4070 Ti (12 GB), and
  the card peaked at 9,540 to 10,000 MiB (about 1,000 MiB of it the desktop) on the measurement document, 10,778 MiB
  on a four-page brochure. The worst case, a table of 8,192 tokens read as 4 samples next to the Judge, was not run.
- **Paddle-only (original sample, `--reader paddle`):** about 94 of the 23,390 characters in the original sample were
  still wrong with PaddleOCR-VL, all model errors: a repeated 腫→腗 misread in the ICD tables, a few small blocks the
  layout model never detects, and 干 / 于 left where 幹 / 於 was meant (valid Traditional characters, so the repair
  deliberately leaves them). Whether MiniCPM repeats the 腫→腗 misread was not measured (the measurement document has
  no ICD tables). PaddleOCR-VL also drifts into Simplified characters (16 in the 21 measured pages, all repaired by
  `--script auto`), and its local model ignores sampling, so only one re-read round can run with it.
- The orientation and unwarping models are not in `models/`. Rotated or warped scans are therefore not
  corrected.
- If a PDF already has a real text layer, as the measurement document does, extracting that text directly is exact.
  OCR is what matters for scanned PDFs.
- Text inside pictures is not read unless you pass `--ocr-images`; on the brochure that left out a heading that is part of
  an image.
- Some harmless messages still appear on stderr, such as `which: no ccache` and a "Bucketed
  engine_config" warning.

## Testing

```bash
uv sync                      # includes the pytest dev group
uv run pytest                # fast tests; the tests that need a PDF skip without one
# also run the CLI on pages 38-58 of the measurement document, with both Readers (GPU + models, about an hour):
PDF_OCR_SAMPLE_PDF="/path/to/台灣人壽解約金 115.09.23.pdf" uv run pytest
```

The end-to-end tests run on the pages in `tests/baseline.json` (38-58) of the document the baseline was measured on. Set
`PDF_OCR_PAGES=42-45` for a shorter run; the error-rate check is then skipped, because the recorded bound belongs
to those pages, unless you also set `PDF_OCR_CER_BOUND`. The check is likewise skipped for any other PDF.

The fast tests need neither a GPU nor models: "read a Block" with a fake Reader and a fake Judge (repair happens
before judging, unjudged labels are skipped, failure verdicts, `--no-judge` and the ink re-read; and the re-read
schedule in `tests/test_reread.py`: stop at the first Reading that passes, the order of the rounds and the 10-Reading
cap, the highest Score winning, unscorable verdicts never winning, duplicates scored once, a Judge failure part-way,
unsupported rounds, seeds), the re-read rounds through the bridges (`tests/test_reread_bridge.py`), the sticky
repair, the Judge's verdicts with a fake renderer and model, the renderer's setup checks and fontconfig, the
bridges, the JSON fields, the prompt and token limit of each block label, the
clean-up of a Reading, the worker's request loop, the client's start-up, timeout and crash handling (against
a stand-in worker, `tests/fake_worker.py`), the bridge and the hook, the `--reader` options, and folder runs: which
Documents a folder holds and where each one writes (`tests/test_folder_plan.py`), which of them are already done
(`tests/test_folder_done.py`), the `settings` and the Engine's model names known from the options alone
(`tests/test_options_only.py`), what is refused before any model loads, and the loop over the Documents with a fake
Engine, skipping, an output root inside the folder and a worker crash included (`tests/test_cli_folder.py`).

The tests marked `gpu` need `PDF_OCR_SAMPLE_PDF` (a PDF with a real text layer; except the reference-example
test), the models, the worker environment (the Judge runs in it for either Reader) and the Judge's setup:

- `tests/test_hook_regression.py` runs the hook on a real page with the real Layout model and a fake Reader.
  It needs no Reader weights and little GPU memory, and it fails when a PaddleX upgrade moves the private
  names the hook uses.
- `tests/test_worker_crash.py` runs the whole CLI with a stand-in worker that crashes: the page it was reading
  is recorded as failed, and the next page gets a new worker.
- `tests/test_folder_run.py` runs the CLI on a folder with the default Reader and the Judge at
  `--judge-threshold 0.98`: two 2-page PDFs cut from the sample (one in a subfolder) and a corrupt PDF. Each good
  Document's Markdown, JSON (apart from `elapsed_seconds`) and CSVs must equal a run on that PDF alone, the corrupt
  file must be reported, the models load once and the exit code is 1. Then it deletes the corrupt PDF, which is
  never done, and runs the folder again into the same output: both Documents are skipped, no model loads and the
  exit code is 0; and once more with `--force`, which redoes both with the output they had. It runs one CLI process
  after another, so it takes the time of five runs.
- `tests/test_e2e.py` runs the CLI twice with the Judge on, plus once with `--no-judge` and once with
  `--judge-threshold 0` (Judge on, nothing re-read, whose Markdown must equal the `--no-judge` one), for each
  Reader, on a few pages. It checks the output structure (the same for both Readers), that every judged Block has a `score`, a
  `verdict` and a `below_threshold` and charts, seals and figures have none, that stderr counts the Blocks below
  the threshold per page, that the Markdown has no Scores, that `--no-judge` gives no Scores and records
  `judge: false`, that the character error rate against the text layer stays at or below the bound in
  `tests/baseline.json`, and that both runs give identical Markdown and JSON, Scores included.
- `tests/test_e2e_reread.py` runs the CLI twice with `--judge-threshold 0.9999`, which forces Blocks through the
  whole schedule, on `PDF_OCR_REREAD_PAGES` (default `4-5`), for each Reader. It checks that the two runs give
  identical JSON and Markdown, re-reads included, that every re-read Block has its `readings`, a `stop_reason`, the
  winner's `content`/`score` (the highest `scored` Reading), at most 10 Readings, that Blocks that were not re-read
  have none of these, that `postprocessing` and the stderr lines count the re-read Blocks, and for `--reader paddle`
  that rounds 3 and 4 are recorded as skipped.
- `tests/test_judge_reference.py` needs no sample PDF: it starts the real worker, scores the SDK's three
  reference examples and compares with the Scores the authors observed (within 0.015), and checks that unparsable
  LaTeX and a missing crop get `precheck_reject` and `missing_original_image` while the rest of the batch is scored.
  It needs the worker environment, Chromium, `node` and Noto Sans CJK.

`PDF_OCR_PAGES` (default: the `pages` in `tests/baseline.json`, `38-58`), `PDF_OCR_MODELS_DIR`,
`PDF_OCR_CER_BOUND`, `PDF_OCR_READERS` (default `minicpm,paddle`) and `PDF_OCR_WORKER_ENV` (the worker's
virtual environment; default `worker/.venv`) override the pages, the models folder, the bound, the Readers
to run and the worker environment. `PDF_OCR_WORKER_ENV` also works for the CLI itself.

`ocr_eval.py` is the error-rate harness. The PDF's characters are assigned to Blocks by position, so
reading order does not matter. It also runs standalone on a finished run and prints the error rate per
page and per Block label:

```bash
uv run ocr_eval.py input.pdf [output/<stem>/<stem>.json] [--report scores.json]
```

Text-layer characters that lie in no Block count as errors; figures, charts and seals are not scored, and neither are
display formulas (LaTeX against typeset glyphs; they are reported as skipped); whitespace, HTML markup, full/half-width
forms, the kind of dash and table-of-contents dot leaders are ignored. The score looks at a table's text, not at its
structure.

`tests/baseline.json` holds the recorded baseline: the document (name, sha256, pages), the error rate measured for each
Reader with its default options, and the bound the end-to-end test enforces (`minicpm` 0.25 %, `paddle` 0.20 %:
the measured rate plus 0.05 percentage points, rounded up to the next 0.05). The measured rates, after issue #7, are
0.163 % (27 of 16,586 characters) for MiniCPM and 0.109 % (18) for PaddleOCR-VL. The rule gives the same bounds as it
did for the earlier 0.199 % and 0.115 %. **This is not the original 21-page sample**, which
is not available; the numbers are not comparable with the 0.40 % in "Earlier tuning". To reproduce them:

```bash
PDF="/path/to/台灣人壽解約金 115.09.23.pdf"        # sha256 1fd73a59c0fb8c1ca33ce8603ca8c0a5b39ff374898730731431d86473236aa7
uv run pdf_ocr.py "$PDF" -o out --pages 38-58                  # MiniCPM, Judge on, threshold 0.5, 216 DPI
uv run ocr_eval.py "$PDF" "out/台灣人壽解約金 115.09.23.json"
uv run pdf_ocr.py "$PDF" -o out_paddle --pages 38-58 --reader paddle
uv run ocr_eval.py "$PDF" "out_paddle/台灣人壽解約金 115.09.23.json"
```

The end-to-end test fails if a Reader's error rate on those pages rises above its bound; it does not fail if it improves.
