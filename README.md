# pdf_ocr: PDF to Markdown, JSON and CSV with MiniCPM-OCR or PaddleOCR-VL

`pdf_ocr.py` converts a PDF into Markdown, a detailed JSON file and one CSV per table. It uses only
the local models in `models/`:

| Model | Role |
|---|---|
| `models/PP-DocLayoutV3` | the Layout model: finds the blocks on each page (text, titles, tables, figures, …) and their reading order |
| `models/MiniCPM-OCR-1.0` | the default Reader: reads each block, with tables as HTML, formulas as LaTeX, and charts or seals if you ask |
| `models/PaddleOCR-VL-1.6` | the other Reader, chosen with `--reader paddle` |

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

The worker environment is created in `worker/.venv`. It is needed only for the default `--reader minicpm`;
with `--reader paddle` you can skip it. If it is missing, the tool stops at once and tells you to run the
second command. The torch wheel from PyPI is built for CUDA 13.2, so the GPU driver must support it. The
two environments are separate processes, so their CUDA libraries never meet.

If no CUDA GPU is found, the tool runs on the CPU, which works but is slow. (The MiniCPM worker needs a CUDA GPU.)

`--models-dir` has to hold only the folders that your options use: `PP-DocLayoutV3` always, plus
`MiniCPM-OCR-1.0` for the default Reader or `PaddleOCR-VL-1.6` for `--reader paddle`. A missing folder is
reported by name before anything is loaded.

On a 12 GB GPU the default path (the Layout model and MiniCPM) is small: measured with `nvidia-smi`, about 0.9 GB
for the main process and about 2.7 GB for the worker once the model is loaded (more while it reads a long table).
`--reader paddle` holds about 8 GB.

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
| `--models-dir` | `./models` | folder containing `PP-DocLayoutV3/` and the Reader's folder (`MiniCPM-OCR-1.0/`, or `PaddleOCR-VL-1.6/` with `--reader paddle`) |
| `--pages` | all | 1-based page selection, e.g. `1-3,7,10-` |
| `--device` | `gpu:0` if available | Paddle device, e.g. `cpu` |
| `--reader {minicpm,paddle}` | `minicpm` | the model that reads each block: MiniCPM-OCR-1.0 (through the torch worker) or PaddleOCR-VL-1.6. With `paddle` the Markdown, the CSVs and the blocks in the JSON are the same as before MiniCPM was added; the JSON only gains a `reader` field in `engine` and `settings` |
| `--dpi` | 216 | page render resolution |
| `--layout-threshold` | 0.2 | layout score threshold; lower finds more blocks but too low invents boxes |
| `--max-new-tokens` | 8192 | decoding limit per block; raise it for very large tables. With MiniCPM it is the limit for tables only (see below) |
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
- `engine`: the Layout model and the Reader (`layout_model`, `reader`), the library versions and the device.
  With MiniCPM it also has the `torch` and `transformers` versions of the worker.
- `settings`: every option that affects the output, including `reader`.
- `postprocessing`: what the clean-up steps changed (with MiniCPM, also `repetition_cuts`).
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

On the sample PDF, all 11 table blocks were valid and became 6 CSVs, because two tables run across
several pages. One is the drug list on pages 14–16 (93 rows). The other is the ICD-10 code table on
pages 17–20 (78 rows). In each, PaddleOCR's merge drops the header row repeated at the top of each
continuation page.

## What was done for accuracy

**The numbers and the steps in this section were measured with PaddleOCR-VL-1.6 (`--reader paddle`).**
The MiniCPM Reader is described after them.

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
2. **OTSL to HTML.** MiniCPM writes tables in OTSL. They are converted with the same converter PaddleX uses.
3. **Table line breaks.** MiniCPM writes an in-cell line break as a *real* newline (PaddleOCR-VL writes a
   literal `\n`). Both become `<br>`, so the CSV keeps the line break inside the cell. This happens inside
   the Reader, so `postprocessing.tables_with_line_breaks_fixed` (the count of the literal `\n` fix) stays 0
   with MiniCPM.
4. **Simplified to Traditional repair**, as for PaddleOCR-VL. It is document-level and runs after all
   pages, which is why it comes after the Reader and not inside it.

The truncated-block re-read (`--no-recover`) also works with MiniCPM.

What differs from PaddleOCR-VL in the text: MiniCPM keeps the printed spacing around Latin letters and
digits (`表 1-1`, `OCR 及`), writes `[...]` for the printed `〔...〕`, and uses `(...)` for full-width
brackets inside table cells. These cost error-rate points when the text layer is the reference, though they
are cosmetic. It also misreads the odd character the repair cannot fix: `爰` came out as the Simplified `爱`,
which the repair turned into `愛`.

**Measurement (a stand-in, not the sample).** The 21-page insurance policy that the numbers above come from
was not available when MiniCPM was added. For a first check, pages 3 to 12 of a different Traditional
Chinese proposal PDF (digital-born, 5,961 text-layer characters, 10 pages) were run with `ocr_eval.py`:

| Reader | errors | character error rate |
|---|---|---|
| PaddleOCR-VL-1.6 | 0 | 0.00 % |
| MiniCPM-OCR-1.0 | 10 | 0.17 % |

This is only a smoke test of the harness and the Reader on a stand-in document. It is not a baseline and it
does not say which Reader is better on the sample. The measurement on the sample is a later ticket.
Roughly the same time per page for both Readers (about 65 s for these 10 pages, including the
model load).

### How the worker is wired in

The main command builds the PaddleOCR pipeline with the Layout model only (PaddleOCR-VL's weights are not
loaded for `--reader minicpm`). Then it replaces the pipeline's per-block recognizer with a bridge that
sends each crop to the worker as a PNG file in a temporary folder and reads the answer back as one line of
JSON. The worker is a long-lived process: the model is loaded once, and the main command stops it at the
end of the run, whatever happens. The code is in `reader_bridge.py` (the hook), `worker_client.py` (the
process and protocol) and `worker/` (the torch side).

The hook uses three private PaddleX names. A start-up check verifies them and stops with a clear message
if a PaddleX upgrade has moved them (`uv sync` restores the pinned versions), and a test runs the hook
against the real pipeline.

If the worker crashes or stops answering, the page it was reading is recorded in `source.pages_failed`
(exit code 1) with the worker's last messages, and the next page starts a fresh worker. Every wait has a
timeout, so a stuck worker cannot hang the run.

## Caveats

- The tuning was measured on one Traditional Chinese document, with PaddleOCR-VL. Other languages and
  layouts should work, but these numbers do not carry over to them. MiniCPM-OCR-1.0 became the default
  before it was measured on that document; see the stand-in measurement above.
- `--reader minicpm` needs the worker environment (`uv sync --project worker`) and a CUDA GPU. The worker's
  torch wheel targets CUDA 13.2; a driver that supports only CUDA 13.0 needs a different torch index in
  `worker/pyproject.toml`. This was not tested.
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

## Testing

```bash
uv sync                      # includes the pytest dev group
uv run pytest                # fast tests; the tests that need a PDF skip without one
PDF_OCR_SAMPLE_PDF=/path/to/policy.pdf uv run pytest          # also run the CLI on pages 1-3 (GPU + models)
```

The fast tests need neither a GPU nor models: the prompt and token limit of each block label, the
clean-up of a Reading, the worker's request loop, the client's start-up, timeout and crash handling (against
a stand-in worker, `tests/fake_worker.py`), the bridge and the hook, and the `--reader` options.

The tests marked `gpu` need `PDF_OCR_SAMPLE_PDF` (a PDF with a real text layer), the models and, for
MiniCPM, the worker environment:

- `tests/test_hook_regression.py` runs the hook on a real page with the real Layout model and a fake Reader.
  It needs no Reader weights and little GPU memory, and it fails when a PaddleX upgrade moves the private
  names the hook uses.
- `tests/test_worker_crash.py` runs the whole CLI with a stand-in worker that crashes: the page it was reading
  is recorded as failed, and the next page gets a new worker.
- `tests/test_e2e.py` runs the CLI twice, once per Reader, on a few pages. It checks the output structure (the
  same for both Readers), that the character error rate against the text layer stays at or below the bound
  in `tests/baseline.json`, and that both runs give identical Markdown and JSON.

`PDF_OCR_PAGES` (default: the `pages` in `tests/baseline.json`, now `1-3`), `PDF_OCR_MODELS_DIR`,
`PDF_OCR_CER_BOUND`, `PDF_OCR_READERS` (default `minicpm,paddle`) and `PDF_OCR_WORKER_ENV` (the worker's
virtual environment; default `worker/.venv`) override the pages, the models folder, the bound, the Readers
to run and the worker environment. `PDF_OCR_WORKER_ENV` also works for the CLI itself.

`ocr_eval.py` is the error-rate harness. The PDF's characters are assigned to Blocks by position, so
reading order does not matter. It also runs standalone on a finished run and prints the error rate per
page and per Block label:

```bash
uv run ocr_eval.py input.pdf [output/<stem>/<stem>.json] [--report scores.json]
```

Text-layer characters that lie in no Block count as errors; figures, charts and seals are not scored;
whitespace, HTML markup, full/half-width forms and table-of-contents dot leaders are ignored.

The bound in `tests/baseline.json` is currently a **placeholder** (5 %), not a measurement. The baseline
of PaddleOCR-VL-1.6 on the 21-page sample still has to be measured with `ocr_eval.py` and recorded there,
on exactly the pages in `baseline.json` (the 0.40 % above came from an earlier throwaway harness with
different normalisation and the whole document, so it is not this tool's baseline).
